"""Unit tests for ``GCSFileSystem.compose_many``.

The GCS JSON transport is replaced by an in-process fake that implements the
parts of the objects resource / objects.compose semantics the tests rely on
(exact generation preconditions, 32-source cap, ordered concatenation,
CRC32C responses and exact-version deletes), so no emulator or real GCS
project is required.
"""

import asyncio

import crcmod
import pytest

from gcsfs.core import (
    GCS_COMPOSE_MAX_SOURCE_OBJECTS,
    GCS_MAX_OBJECT_SIZE,
    GCSFileSystem,
    _crc32c_combine,
    _crc32c_to_b64,
    build_compose_plan,
)
from gcsfs.retry import ChecksumError, HttpError

BUCKET = "test-bucket"
DEST = f"{BUCKET}/destination"
_CRC_FN = crcmod.mkCrcFun(0x11EDC6F41, initCrc=0, xorOut=0xFFFFFFFF, rev=True)
WRONG_CRC = "Zm9vAAAA"


def b64_crc(data):
    return _crc32c_to_b64(_CRC_FN(data))


def source_data(i):
    # Deterministic, non-empty, ordered payload (i is the input position).
    return bytes([i % 256]) * (1 + (i % 5))


class FakeGCS:
    """Minimal, stateful fake of the GCS JSON API behind ``_call``."""

    def __init__(self):
        # name -> {"gen", "data", "crc32c", "storageClass", "forced_size"}
        self.objects = {}
        self.requests = []
        self.compose_calls = []
        self.compose_attempts = []
        self.delete_calls = []
        self.successful_deletes = []
        self.in_flight = 0
        self.max_in_flight = 0
        # Hooks for fault injection.
        self.compose_hook = None
        self.crc_overrides = {}
        self.delete_hook = None
        self.gate = None
        self.gate_after = None

    def seed(self, name, data, gen=1, storage_class="STANDARD", forced_size=None):
        self.objects[name] = {
            "gen": str(gen),
            "data": data,
            "crc32c": b64_crc(data),
            "storageClass": storage_class,
            "forced_size": forced_size,
        }

    def replace(self, name, data):
        entry = self.objects[name]
        self.objects[name] = {
            "gen": str(int(entry["gen"]) + 1),
            "data": data,
            "crc32c": b64_crc(data),
            "storageClass": entry["storageClass"],
            "forced_size": None,
        }

    def resource(self, name):
        entry = self.objects[name]
        size = (
            entry["forced_size"]
            if entry["forced_size"] is not None
            else len(entry["data"])
        )
        return {
            "name": name,
            "bucket": BUCKET,
            "generation": entry["gen"],
            "size": str(size),
            "crc32c": entry["crc32c"],
            "storageClass": entry["storageClass"],
            "componentCount": entry.get("componentCount", 1),
        }

    async def __call__(
        self, method, template, *args, json_out=False, info_out=False, **kwargs
    ):
        self.requests.append((method, template, args, kwargs))
        if method == "GET":
            name = args[1]
            generation = kwargs.get("generation")
            if name not in self.objects:
                raise FileNotFoundError(name)
            if generation is not None and self.objects[name]["gen"] != str(generation):
                raise FileNotFoundError(name)
            return self.resource(name)
        if method == "POST" and template.endswith("/compose"):
            return await self._compose(args[0], args[1], kwargs)
        if method == "DELETE":
            return await self._delete(args[0], args[1], kwargs)
        raise AssertionError(f"unexpected request: {method} {template}")

    async def _compose(self, bucket, dest_name, kwargs):
        body = kwargs["json"]
        params = {
            k: v for k, v in kwargs.items() if k not in ("json", "headers", "json_out")
        }
        self.compose_attempts.append(
            {"dest": dest_name, "body": body, "params": params}
        )
        # Fault injection runs first, emulating events that precede validation.
        if self.compose_hook is not None:
            injected = self.compose_hook(dest_name, body, params)
            if isinstance(injected, BaseException):
                raise injected
        gated = self.gate is not None and (
            self.gate_after is None or len(self.compose_calls) >= self.gate_after
        )
        if gated:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            try:
                await self.gate.wait()
            finally:
                self.in_flight -= 1

        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if_match = kwargs.get("ifGenerationMatch")
            exists = dest_name in self.objects
            if if_match == "0":
                if exists:
                    raise FileExistsError(dest_name)
            elif if_match is not None and (
                not exists or self.objects[dest_name]["gen"] != str(if_match)
            ):
                raise FileExistsError(dest_name)

            source_objects = body["sourceObjects"]
            if not 1 <= len(source_objects) <= GCS_COMPOSE_MAX_SOURCE_OBJECTS:
                raise AssertionError(
                    f"server-side compose cap violated: {len(source_objects)}"
                )
            data = b""
            component_count = 0
            for source in source_objects:
                name = source["name"]
                if name not in self.objects:
                    raise FileNotFoundError(name)
                # Legacy merge sends unpinned sources; pinned sources must
                # satisfy both the generation selector and its precondition.
                if "generation" in source:
                    live_generation = self.objects[name]["gen"]
                    if source.get("generation") != live_generation:
                        raise FileExistsError(name)
                    preconditions = source.get("objectPreconditions", {})
                    if preconditions.get("ifGenerationMatch") != live_generation:
                        raise FileExistsError(name)
                data += self.objects[name]["data"]
                component_count += self.objects[name].get("componentCount", 1)

            new_generation = str(
                max((int(entry["gen"]) for entry in self.objects.values()), default=0)
                + 1
            )
            crc32c = self.crc_overrides.get(dest_name, b64_crc(data))
            resource = {
                "name": dest_name,
                "bucket": bucket,
                "generation": new_generation,
                "size": str(len(data)),
                "crc32c": crc32c,
                "storageClass": "STANDARD",
                "componentCount": component_count,
            }
            self.objects[dest_name] = {
                "gen": new_generation,
                "data": data,
                "crc32c": crc32c,
                "storageClass": "STANDARD",
                "componentCount": component_count,
            }
            self.compose_calls.append(
                {
                    "dest": dest_name,
                    "body": body,
                    "params": params,
                    "resource": resource,
                }
            )
            return resource
        finally:
            self.in_flight -= 1

    async def _delete(self, bucket, name, kwargs):
        generation = kwargs.get("generation")
        self.delete_calls.append((name, generation))
        if generation is None:
            raise AssertionError("cleanup issued a delete without an exact generation")
        if self.delete_hook is not None:
            injected = self.delete_hook(name, generation)
            if isinstance(injected, BaseException):
                raise injected
        if name not in self.objects:
            raise FileNotFoundError(name)
        if self.objects[name]["gen"] != str(generation):
            raise FileExistsError(name)
        del self.objects[name]
        self.successful_deletes.append((name, generation))
        return {}, b""


@pytest.fixture
def harness():
    fs = GCSFileSystem(asynchronous=True, token="anon", skip_instance_cache=True)
    fake = FakeGCS()

    async def fake_call(method, template, *args, **kwargs):
        return await fake(method, template, *args, **kwargs)

    fs._call = fake_call
    return fs, fake


def seed_sources(fake, n, prefix="src"):
    paths = []
    for i in range(n):
        name = f"{prefix}-{i:04d}"
        fake.seed(name, source_data(i), gen=1000 + i)
        paths.append(f"{BUCKET}/{name}")
    return paths


def layer_shape(plan):
    return [len(layer) for layer in plan.layers]


def assert_conditions_pinned(fake, plan):
    for call in fake.compose_calls:
        for source in call["body"]["sourceObjects"]:
            assert "generation" in source
            assert source["objectPreconditions"] == {
                "ifGenerationMatch": source["generation"]
            }
        assert len(call["body"]["sourceObjects"]) <= GCS_COMPOSE_MAX_SOURCE_OBJECTS
        if call["dest"] == plan.destination_name:
            assert call["params"]["ifGenerationMatch"] is not None
        else:
            assert call["params"]["ifGenerationMatch"] == "0"
            assert call["dest"].startswith(plan.temp_prefix)
        assert "md5Hash" not in call["body"]


def assert_cleanup_exact(fake, result):
    ledger = {(t.name, t.generation) for t in result.created_temps}
    attempts = set(fake.delete_calls)
    removed = set(fake.successful_deletes)
    # Only ledger versions are ever targeted, each with its exact generation.
    assert attempts <= ledger
    assert len(fake.delete_calls) == len(attempts)
    for name, generation in fake.delete_calls:
        assert (name, generation) in ledger
    # Every version whose delete did not fail is actually gone.
    assert removed == {(t.name, t.generation) for t in result.deleted_temps}
    for name, _generation in removed:
        assert name not in fake.objects


# ---------------------------------------------------------------------------
# Pure planning / CRC helpers
# ---------------------------------------------------------------------------


def test_crc32c_combine_matches_direct_crc():
    import os

    from gcsfs.core import ComposeSource  # noqa: F401

    for _ in range(25):
        parts = [os.urandom(1 + os.urandom(1)[0] % 64) for _ in range(6)]
        combined = _CRC_FN(b"".join(parts))
        acc = None
        for part in parts:
            component = _CRC_FN(part)
            acc = (
                component if acc is None else _crc32c_combine(acc, component, len(part))
            )
        assert acc == combined


def test_plan_shapes_and_32_cap():
    from gcsfs.core import ComposeSource

    def make(n):
        return tuple(
            ComposeSource(
                path=f"{BUCKET}/k{i}",
                bucket=BUCKET,
                name=f"k{i}",
                generation=str(i + 1),
                size=1,
                crc32c=b64_crc(b"x"),
                storage_class="STANDARD",
            )
            for i in range(n)
        )

    assert layer_shape(build_compose_plan(make(1), BUCKET, "d", "t/", 32, "o")) == [1]
    assert layer_shape(build_compose_plan(make(32), BUCKET, "d", "t/", 32, "o")) == [1]
    assert layer_shape(build_compose_plan(make(33), BUCKET, "d", "t/", 32, "o")) == [
        2,
        1,
    ]
    plan = build_compose_plan(make(1025), BUCKET, "d", "t/", 32, "o")
    assert layer_shape(plan) == [33, 2, 1]
    for layer in plan.layers:
        for step in layer:
            assert 1 <= len(step.refs) <= 32
    names = [step.destination_name for layer in plan.layers[:-1] for step in layer]
    assert len(names) == len(set(names))
    assert all(name.startswith("t/o/") for name in names)
    assert plan.layers[-1][0].is_final
    assert all(not step.is_final for layer in plan.layers[:-1] for step in layer)


def test_plan_rejects_bad_chunk_size_and_empty_inputs():
    from gcsfs.core import ComposeSource

    src = (
        ComposeSource(
            path=f"{BUCKET}/k",
            bucket=BUCKET,
            name="k",
            generation="1",
            size=1,
            crc32c=b64_crc(b"x"),
            storage_class="STANDARD",
        ),
    )
    with pytest.raises(ValueError):
        build_compose_plan(src, BUCKET, "d", "t/", 33, "o")
    with pytest.raises(ValueError):
        build_compose_plan((), BUCKET, "d", "t/", 32, "o")


# ---------------------------------------------------------------------------
# plan_only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_only_33_sources_issues_no_writes(harness):
    fs, fake = harness
    paths = seed_sources(fake, 33)
    plan = await fs._compose_many(DEST, paths, plan_only=True)
    assert layer_shape(plan) == [2, 1]
    assert plan.total_size == sum(len(source_data(i)) for i in range(33))
    # No compose, no delete, no destination probe.
    assert fake.compose_calls == []
    assert fake.delete_calls == []
    assert all(method != "POST" for method, *_ in fake.requests)
    # Sources were pinned from the (mocked) service metadata, in order.
    assert [s.generation for s in plan.sources] == [str(1000 + i) for i in range(33)]
    assert plan.layers[-1][0].if_generation_match is None


@pytest.mark.asyncio
async def test_plan_only_1025_sources(harness):
    fs, fake = harness
    paths = seed_sources(fake, 1025)
    plan = await fs._compose_many(DEST, paths, plan_only=True)
    assert layer_shape(plan) == [33, 2, 1]
    assert fake.compose_calls == []
    assert fake.delete_calls == []


# ---------------------------------------------------------------------------
# End-to-end layered execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compose_many_33_end_to_end(harness):
    fs, fake = harness
    paths = seed_sources(fake, 33)
    result = await fs._compose_many(DEST, paths)

    assert result.success
    assert layer_shape(result.plan) == [2, 1]
    # Two temp composes (32 + 1 sources) then one final compose of the two temps.
    posts = fake.compose_calls
    assert [len(call["body"]["sourceObjects"]) for call in posts] == [32, 1, 2]
    expected = b"".join(source_data(i) for i in range(33))
    assert fake.objects[result.plan.destination_name]["data"] == expected
    assert result.destination_size == len(expected)
    # Destination did not exist: explicit create-only precondition.
    assert posts[-1]["params"]["ifGenerationMatch"] == "0"
    assert_conditions_pinned(fake, result.plan)
    assert len(result.created_temps) == 2
    assert_cleanup_exact(fake, result)
    assert result.uncleaned == []


@pytest.mark.asyncio
async def test_compose_many_1025_sources(harness):
    fs, fake = harness
    paths = seed_sources(fake, 1025)
    result = await fs._compose_many(DEST, paths)

    assert result.success
    assert layer_shape(result.plan) == [33, 2, 1]
    posts = fake.compose_calls
    assert len(posts) == 36
    for call in posts:
        assert 1 <= len(call["body"]["sourceObjects"]) <= 32
    expected = b"".join(source_data(i) for i in range(1025))
    assert fake.objects[result.plan.destination_name]["data"] == expected
    assert result.destination_crc32c == b64_crc(expected)
    assert result.destination_size == len(expected)
    assert_conditions_pinned(fake, result.plan)
    # 35 intermediate objects, all deleted by their exact returned generations.
    assert len(result.created_temps) == 35
    assert_cleanup_exact(fake, result)
    assert result.uncleaned == []
    assert result.plan.destination_name in fake.objects


@pytest.mark.asyncio
async def test_duplicate_sources_keep_duplicate_bytes(harness):
    fs, fake = harness
    fake.seed("dup", b"abc", gen=77)
    fake.seed("src-0000", source_data(0), gen=1000)
    paths = [
        f"{BUCKET}/dup",
        f"{BUCKET}/src-0000",
        f"{BUCKET}/dup",
        f"{BUCKET}/dup",
    ]
    result = await fs._compose_many(DEST, paths)

    assert result.success
    assert fake.objects[result.plan.destination_name]["data"] == (
        b"abc" + source_data(0) + b"abc" + b"abc"
    )
    assert [s.name for s in result.plan.sources] == [
        "dup",
        "src-0000",
        "dup",
        "dup",
    ]
    assert all(s.generation == "77" for s in result.plan.sources if s.name == "dup")
    first_body = fake.compose_calls[0]["body"]
    dup_entries = [s for s in first_body["sourceObjects"] if s["name"] == "dup"]
    assert len(dup_entries) == 3
    assert all(
        entry["generation"] == "77"
        and entry["objectPreconditions"] == {"ifGenerationMatch": "77"}
        for entry in dup_entries
    )


# ---------------------------------------------------------------------------
# Failure semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_source_replaced_midway_fails_and_cleans_temps(harness):
    fs, fake = harness
    paths = seed_sources(fake, 33)

    def hook(dest_name, body, params):
        # Another process replaces source 0 after it was pinned at gen 1000.
        fake.replace("src-0000", b"TAMPERED")
        return None

    fake.compose_hook = hook
    with pytest.raises(FileExistsError) as excinfo:
        await fs._compose_many(DEST, paths)

    result = excinfo.value.compose_many_result
    assert result.plan.destination_name not in fake.objects
    assert result.destination_generation is None
    assert_cleanup_exact(fake, result)
    assert result.uncleaned == []


@pytest.mark.asyncio
async def test_temp_name_conflict_does_not_overwrite(harness):
    fs, fake = harness
    paths = seed_sources(fake, 33)
    fake.compose_hook = lambda dest_name, body, params: FileExistsError(dest_name)

    with pytest.raises(FileExistsError):
        # Serial execution makes the first failing step deterministic; the
        # remaining layer steps must never be scheduled.
        await fs._compose_many(DEST, paths, concurrency=1)

    # Exactly one compose attempt: no blind retry that drops the precondition.
    assert len(fake.compose_attempts) == 1
    assert fake.compose_attempts[0]["params"]["ifGenerationMatch"] == "0"
    assert fake.compose_calls == []
    assert fake.delete_calls == []
    assert "destination" not in fake.objects


@pytest.mark.asyncio
async def test_cleanup_failure_keeps_inspectable_uncleaned_list(harness):
    fs, fake = harness
    paths = seed_sources(fake, 33)

    def delete_hook(name, generation):
        if name.endswith("part0"):
            return HttpError({"code": 403, "message": "injected cleanup failure"})
        return None

    fake.delete_hook = delete_hook
    result = await fs._compose_many(DEST, paths)

    assert result.success
    assert len(result.uncleaned) == 1
    stuck, message = result.uncleaned[0]
    assert stuck.name.endswith("part0")
    assert "injected cleanup failure" in message
    # The stuck object is still live; every other temp was deleted exactly.
    assert stuck.name in fake.objects
    removed_names = {name for name, _ in fake.successful_deletes}
    assert stuck.name not in removed_names
    ledger = {t.name for t in result.created_temps}
    attempted_names = {name for name, _ in fake.delete_calls}
    assert attempted_names == ledger
    # No prefix/recursive delete: every call targeted an exact ledger version.
    for name, generation in fake.delete_calls:
        assert name in ledger and generation is not None
    assert result.plan.destination_name in fake.objects


@pytest.mark.asyncio
async def test_final_target_conflict_does_not_overwrite_new_object(harness):
    fs, fake = harness
    paths = seed_sources(fake, 2)
    fake.seed("destination", b"OLD", gen=1)

    def hook(dest_name, body, params):
        if dest_name == "destination":
            assert params["ifGenerationMatch"] == "1"
            # The live object moved on before the final compose landed.
            fake.replace("destination", b"NEW")
            return FileExistsError(dest_name)
        return None

    fake.compose_hook = hook
    with pytest.raises(FileExistsError) as excinfo:
        await fs._compose_many(DEST, paths)

    result = excinfo.value.compose_many_result
    # The newer object survives untouched; compose never overwrote it.
    assert fake.objects["destination"]["data"] == b"NEW"
    assert fake.objects["destination"]["gen"] == "2"
    assert result.destination_generation is None
    assert_cleanup_exact(fake, result)


@pytest.mark.asyncio
async def test_explicit_destination_generation(harness):
    fs, fake = harness
    paths = seed_sources(fake, 2)

    # Create-only on an absent destination.
    result = await fs._compose_many(DEST, paths, if_generation_match=0)
    assert result.success
    assert fake.compose_calls[-1]["params"]["ifGenerationMatch"] == "0"

    # Pinning the current generation allows an overwrite.
    current = fake.objects["destination"]["gen"]
    result = await fs._compose_many(DEST, paths, if_generation_match=int(current))
    assert result.success
    assert fake.compose_calls[-1]["params"]["ifGenerationMatch"] == current

    # A stale explicit generation fails without publishing anything new.
    before = fake.objects["destination"]["gen"]
    with pytest.raises(FileExistsError) as excinfo:
        await fs._compose_many(DEST, paths, if_generation_match=999)
    result = excinfo.value.compose_many_result
    assert result.plan.destination_name in fake.objects
    assert fake.objects["destination"]["gen"] == before


@pytest.mark.asyncio
async def test_path_bucket_and_generation_are_not_ignored(harness):
    fs, fake = harness
    fake.seed("k0", source_data(0), gen=7)
    fake.seed("k1", source_data(1), gen=8)
    # The filesystem is NOT version-aware, yet a generation on the path must be
    # honoured instead of becoming part of the object name.
    assert fs.version_aware is False
    plan = await fs._compose_many(
        DEST,
        [f"{BUCKET}/k0#7", f"{BUCKET}/k1?generation=8"],
        plan_only=True,
    )
    assert plan.sources[0].generation == "7"
    assert plan.sources[1].generation == "8"
    gets = [call for call in fake.requests if call[0] == "GET"]
    assert gets[0][3]["generation"] == "7"
    assert gets[1][3]["generation"] == "8"

    # A bucket on an input path is rejected, never silently swapped.
    with pytest.raises(ValueError, match="destination bucket"):
        await fs._compose_many(
            DEST, [f"{BUCKET}/k0", "other-bucket/k1"], plan_only=True
        )
    # A generation on the destination path is rejected rather than ignored.
    with pytest.raises(ValueError, match="specific object generation"):
        await fs._compose_many(f"{DEST}#3", [f"{BUCKET}/k0"], plan_only=True)


@pytest.mark.asyncio
async def test_empty_sources_and_total_size_overflow(harness):
    fs, fake = harness
    fake.seed("empty", b"", gen=1)
    fake.seed("ok", source_data(0), gen=2)
    with pytest.raises(ValueError, match="0-byte"):
        await fs._compose_many(DEST, [f"{BUCKET}/empty", f"{BUCKET}/ok"])
    assert fake.compose_calls == []

    big = FakeGCS()

    async def big_call(method, template, *args, **kwargs):
        return await big(method, template, *args, **kwargs)

    fs._call = big_call
    big.seed("big1", b"x", forced_size=GCS_MAX_OBJECT_SIZE)
    big.seed("big2", b"x", forced_size=GCS_MAX_OBJECT_SIZE)
    with pytest.raises(ValueError, match="5 TiB"):
        await fs._compose_many(DEST, [f"{BUCKET}/big1", f"{BUCKET}/big2"])
    assert big.compose_calls == []


@pytest.mark.asyncio
async def test_storage_class_mismatch_rejected(harness):
    fs, fake = harness
    fake.seed("std", b"a", storage_class="STANDARD")
    fake.seed("cold", b"b", storage_class="COLDLINE")
    with pytest.raises(ValueError, match="storage class"):
        await fs._compose_many(DEST, [f"{BUCKET}/std", f"{BUCKET}/cold"])
    assert fake.compose_calls == []


# ---------------------------------------------------------------------------
# CRC32C handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_crc32c_mismatch_on_temp_unpublishes_nothing(harness):
    fs, fake = harness
    paths = seed_sources(fake, 33)

    def hook(dest_name, body, params):
        # Corrupt the CRC of the very first temporary object at run time, so
        # the name (including the operation id) is the one actually used.
        if dest_name != "destination" and not fake.crc_overrides:
            fake.crc_overrides[dest_name] = WRONG_CRC
        return None

    fake.compose_hook = hook
    with pytest.raises(ChecksumError) as excinfo:
        await fs._compose_many(DEST, paths)
    result = excinfo.value.compose_many_result
    assert "destination" not in fake.objects
    # The corrupt temp itself is in the ledger and gets deleted exactly.
    assert_cleanup_exact(fake, result)


@pytest.mark.asyncio
async def test_crc32c_mismatch_on_final_removes_destination(harness):
    fs, fake = harness
    paths = seed_sources(fake, 2)
    fake.crc_overrides["destination"] = WRONG_CRC
    with pytest.raises(ChecksumError) as excinfo:
        await fs._compose_many(DEST, paths)
    result = excinfo.value.compose_many_result
    assert result.destination_generation is None
    assert "destination" not in fake.objects
    # The unpublished destination version was removed using its exact gen.
    assert any(name == "destination" for name, _ in fake.delete_calls)
    assert result.uncleaned == []


@pytest.mark.asyncio
async def test_check_crc32c_disabled_needs_no_digest(harness):
    fs, fake = harness
    fake.seed("no-crc-1", b"hello")
    fake.seed("no-crc-2", b"world")
    for entry in fake.objects.values():
        entry["crc32c"] = None
    result = await fs._compose_many(
        DEST,
        [f"{BUCKET}/no-crc-1", f"{BUCKET}/no-crc-2"],
        check_crc32c=False,
    )
    assert result.success
    assert fake.objects["destination"]["data"] == b"helloworld"


# ---------------------------------------------------------------------------
# Concurrency, cancellation and legacy merge compatibility
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrency_bounded_to_two(harness):
    fs, fake = harness
    paths = seed_sources(fake, 64)
    fake.gate = asyncio.Event()
    task = asyncio.ensure_future(fs._compose_many(DEST, paths, concurrency=2))
    await asyncio.sleep(0.05)
    assert 1 <= fake.max_in_flight <= 2
    fake.gate.set()
    result = await task
    assert result.success
    assert fake.max_in_flight <= 2


@pytest.mark.asyncio
async def test_concurrency_bounded_to_one(harness):
    fs, fake = harness
    paths = seed_sources(fake, 64)
    fake.gate = asyncio.Event()
    task = asyncio.ensure_future(fs._compose_many(DEST, paths, concurrency=1))
    await asyncio.sleep(0.05)
    assert fake.max_in_flight == 1
    fake.gate.set()
    result = await task
    assert result.success
    assert fake.max_in_flight == 1


@pytest.mark.asyncio
async def test_cancellation_cleans_up_issued_composes(harness):
    fs, fake = harness
    paths = seed_sources(fake, 33)
    fake.gate = asyncio.Event()
    fake.gate_after = 2
    task = asyncio.ensure_future(fs._compose_many(DEST, paths, concurrency=1))
    # Wait until two composes have been issued and recorded.
    while len(fake.compose_calls) < 2:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # Both created temps are removed by their exact returned generations.
    assert len(fake.delete_calls) == 2
    created = {
        (call["dest"], call["resource"]["generation"]) for call in fake.compose_calls
    }
    assert set(fake.delete_calls) == created
    assert "destination" not in fake.objects


@pytest.mark.asyncio
async def test_legacy_merge_shape_and_32_guard(harness):
    fs, fake = harness
    fake.seed("a", b"aa")
    fake.seed("b", b"bb")
    await fs._merge(DEST, [f"{BUCKET}/a", f"{BUCKET}/b"])
    call = fake.compose_calls[0]
    # Legacy request shape is unchanged.
    assert call["body"] == {
        "sourceObjects": [{"name": "a"}, {"name": "b"}],
        "kind": "storage#composeRequest",
        "destination": {"name": "destination", "bucket": BUCKET},
    }
    assert "ifGenerationMatch" not in call["params"]
    assert fake.objects["destination"]["data"] == b"aabb"

    with pytest.raises(ValueError, match="compose_many"):
        await fs._merge(DEST, [f"{BUCKET}/a"] * 33)
    # The oversized request never reached the (fake) server.
    assert len(fake.compose_calls) == 1


@pytest.mark.asyncio
async def test_invalid_arguments(harness):
    fs, fake = harness
    paths = seed_sources(fake, 2)
    with pytest.raises(ValueError):
        await fs._compose_many(DEST, paths, concurrency=0)
    with pytest.raises(ValueError):
        await fs._compose_many(DEST, paths, max_sources_per_compose=0)
    with pytest.raises(ValueError):
        await fs._compose_many(DEST, paths, max_sources_per_compose=33)
    with pytest.raises(ValueError):
        await fs._compose_many(DEST, paths, if_generation_match=-1)
    with pytest.raises(ValueError):
        await fs._compose_many(DEST, [])
    with pytest.raises(ValueError):
        await fs._compose_many(BUCKET, paths)
