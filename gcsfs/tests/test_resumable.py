"""
Tests for checkpoint-backed resumable uploads (gcsfs.resumable).

The GCS resumable protocol is exercised against an in-memory fake server that
implements the real wire contract: 308 ``Range`` acknowledgements, status
queries with ``Content-Range: bytes */<size>``, 404/410 expiry and final
200/201 responses. No emulator or network access is required.
"""

import asyncio
import hashlib
import io
import json
import logging
import os
import re
import stat as stat_module
from types import SimpleNamespace

import aiohttp
import pytest

import gcsfs.core
import gcsfs.resumable
from gcsfs.core import GCS_MIN_BLOCK_SIZE, GCSFileSystem
from gcsfs.resumable import (
    STATE_COMPLETE,
    STATE_FINAL_UNCERTAIN,
    STATE_UPLOADING,
    ResumableUploadCheckpointError,
    ResumableUploadFinalUncertain,
    ResumableUploadSessionExpired,
    ResumableUploadSourceChanged,
    ResumableUploadStateError,
    ResumableUploadTargetMismatch,
    cancel_resumable_upload,
    load_checkpoint,
    resumable_upload_file,
    resumable_upload_status,
)

BUCKET = "resumable-bucket"
KEY = "object/key.bin"
RPATH = f"gs://{BUCKET}/{KEY}"
SESSION_URL = (
    "https://storage.example.com/upload/storage/v1/b/"
    f"{BUCKET}/o?uploadType=resumable&upload_id=session-abc-1234567890"
    "&ifGenerationMatch=0"
)
UPLOAD_ID = "session-abc-1234567890"


# ---------------------------------------------------------------------------
# Fake GCS resumable upload server
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status, headers=None, body=b""):
        self.status = status
        self.headers = dict(headers or {})
        self._body = body

    async def read(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, server):
        self.server = server

    def request(self, method, url, headers=None, data=None, params=None, timeout=None):
        # Mirrors aiohttp: request() returns an async context manager directly.
        return self.server.handle(method, url, dict(headers or {}), data, params or {})


class FakeResumableServer:
    """Stateful fake implementing the GCS resumable upload wire protocol."""

    def __init__(self):
        self.sessions = {}
        self.events = []  # (method, url, content_range, body_length, params)
        self.deletes = []
        self.initiations = []

    def create_session(self, url, total, **flags):
        self.sessions[url] = {
            "total": total,
            "received": bytearray(),
            "done": False,
            "flags": flags,
            "data_posts": 0,
            "generation": "9001",
        }
        return url

    def session(self, url):
        return self.sessions[url]

    def _resolve(self, url):
        # GCS answers on the session URL regardless of the precondition query.
        if url in self.sessions:
            return url
        target = gcsfs.core._strip_generation_match(url)
        for candidate in self.sessions:
            if gcsfs.core._strip_generation_match(candidate) == target:
                return candidate
        return None

    def handle(self, method, url, headers, data, params):
        content_range = headers.get("Content-Range")
        body = b"" if data is None else data
        self.events.append((method, url, content_range, len(body), params))

        requested_url = url
        resolved = self._resolve(url)
        if resolved is None:
            return FakeResponse(404)
        session = self.sessions[resolved]

        if method == "DELETE":
            self.deletes.append(requested_url)
            del self.sessions[resolved]
            return FakeResponse(204)

        flags = session["flags"]
        if flags.get("expire"):
            return FakeResponse(int(flags["expire"]))

        # Empty-body requests are status queries (or zero-byte finalization).
        if content_range.startswith("bytes */"):
            if session["total"] == 0:
                session["done"] = True
                return FakeResponse(200, {}, self._object_body(session))
            if session["done"]:
                return FakeResponse(200, {}, self._object_body(session))
            if flags.get("no_range_on_query"):
                flags["no_range_on_query"] = False
                # A 308 without Range declares that the server holds nothing.
                session["received"] = bytearray()
                return FakeResponse(308, {})
            if session["received"]:
                return FakeResponse(
                    308, {"Range": f"bytes=0-{len(session['received']) - 1}"}
                )
            return FakeResponse(308, {})

        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\*|\d+)", content_range)
        assert match, f"unexpected Content-Range: {content_range}"
        offset = int(match.group(1))
        end = int(match.group(2))
        is_final = match.group(3) != "*"
        assert end == offset + len(body) - 1
        session["data_posts"] += 1

        if is_final and flags.get("complete_then_disconnect"):
            self._place(session, offset, body)
            if len(session["received"]) >= session["total"]:
                session["done"] = True
            raise aiohttp.ClientConnectionError("response lost after finalize")

        if is_final and flags.get("disconnect_final"):
            raise aiohttp.ClientConnectionError("dropped final request")

        if flags.get("disconnect_first_n"):
            flags["disconnect_first_n"] -= 1
            raise aiohttp.ClientConnectionError("dropped request")

        # A consumed (completed) session rejects further chunk POSTs, like GCS.
        if session["done"]:
            return FakeResponse(404)

        if flags.get("partial_first") and not flags.get("_partial_done"):
            flags["_partial_done"] = True
            accept = int(flags["partial_first"])
            self._place(session, offset, body[:accept])
            return FakeResponse(308, {"Range": f"bytes=0-{offset + accept - 1}"})

        if flags.get("no_range_once") and session["data_posts"] == 2:
            flags["no_range_once"] = False
            # The server lost its state: the no-Range 308 declares zero bytes.
            session["received"] = bytearray()
            return FakeResponse(308, {})

        if flags.get("stall"):
            remaining, at = flags["stall"]
            flags["stall"] = (remaining - 1, at)
            if remaining - 1 <= 0:
                del flags["stall"]
            self._place(session, offset, body)
            return FakeResponse(308, {"Range": f"bytes=0-{at - 1}"})

        if flags.get("always_stall") is not None:
            at = int(flags["always_stall"])
            self._place(session, offset, body)
            return FakeResponse(308, {"Range": f"bytes=0-{at - 1}"})

        self._place(session, offset, body)
        if is_final and len(session["received"]) >= session["total"]:
            session["done"] = True
            return FakeResponse(200, {}, self._object_body(session))
        return FakeResponse(308, {"Range": f"bytes=0-{len(session['received']) - 1}"})

    @staticmethod
    def _place(session, offset, body):
        received = session["received"]
        assert offset <= len(received), "gap in resumable upload byte stream"
        if offset == len(received):
            received.extend(body)
        else:
            received[offset : offset + len(body)] = body

    @staticmethod
    def _object_body(session):
        return json.dumps(
            {
                "name": KEY,
                "bucket": BUCKET,
                "generation": session["generation"],
                "size": str(session["total"]),
            }
        ).encode()

    def data_requests(self):
        return [event for event in self.events if event[0] == "POST" and event[2]]

    def status_queries(self):
        return [
            event
            for event in self.events
            if event[0] == "POST" and event[2].startswith("bytes */")
        ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    real_sleep = asyncio.sleep

    async def instant_sleep(delay):
        await real_sleep(0)

    monkeypatch.setattr(gcsfs.resumable.asyncio, "sleep", instant_sleep)


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    server = FakeResumableServer()
    fs = GCSFileSystem(token="anon", skip_instance_cache=True)
    fs._session = FakeSession(server)

    async def fake_initiate(
        fs_,
        bucket,
        key,
        content_type,
        metadata,
        fixed_key_metadata,
        mode,
        kms_key_name,
        if_generation_match,
    ):
        server.initiations.append(
            {
                "bucket": bucket,
                "key": key,
                "content_type": content_type,
                "metadata": metadata,
                "fixed_key_metadata": fixed_key_metadata,
                "mode": mode,
                "kms_key_name": kms_key_name,
                "if_generation_match": if_generation_match,
            }
        )
        return SESSION_URL

    monkeypatch.setattr(gcsfs.core, "_http_initiate_upload", fake_initiate)

    source = tmp_path / "source.bin"
    checkpoint = tmp_path / "private" / "upload.ckpt"

    def make_fs(**kwargs):
        other = GCSFileSystem(token="anon", skip_instance_cache=True, **kwargs)
        other._session = FakeSession(server)
        return other

    return SimpleNamespace(
        fs=fs,
        server=server,
        source=source,
        checkpoint=checkpoint,
        make_fs=make_fs,
        tmp_path=tmp_path,
    )


def write_source(path, size, seed=0):
    data = bytes((i + seed) % 251 for i in range(size))
    with open(path, "wb") as handle:
        handle.write(data)
    return data


def read_checkpoint(path):
    return load_checkpoint(str(path))


def upload_kwargs(**overrides):
    base = {"chunksize": GCS_MIN_BLOCK_SIZE, "max_attempts": 3}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Happy path and wire correctness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_upload_content_ranges_final_bytes_and_checkpoint(ctx):
    size = 3 * GCS_MIN_BLOCK_SIZE + 12345
    data = write_source(ctx.source, size)
    ctx.server.create_session(SESSION_URL, size)

    result = await resumable_upload_file(
        ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
    )

    assert result == {
        "state": STATE_COMPLETE,
        "confirmed_offset": size,
        "total_size": size,
        "generation": "9001",
        "checkpoint_path": os.path.abspath(str(ctx.checkpoint)),
    }
    assert bytes(ctx.server.session(SESSION_URL)["received"]) == data
    assert len(ctx.server.initiations) == 1

    posts = ctx.server.data_requests()
    # Non-final chunks are exact 256 KiB multiples with unknown total size.
    for event in posts[:-1]:
        assert event[3] % GCS_MIN_BLOCK_SIZE == 0
        assert event[2].endswith("/*")
    # The final chunk carries the total size and may be unaligned.
    final_length = posts[-1][3]
    assert final_length == 12345
    assert posts[-1][2] == f"bytes {size - final_length}-{size - 1}/{size}"

    payload = read_checkpoint(ctx.checkpoint)
    assert payload["state"] == STATE_COMPLETE
    assert payload["confirmed_offset"] == size
    assert payload["source"]["prefix_digest"] == hashlib.sha256(data).hexdigest()
    assert payload["session_url"] == SESSION_URL
    # The session URL lives only in the private checkpoint.
    raw = ctx.checkpoint.read_bytes()
    assert SESSION_URL.encode() in raw
    mode = stat_module.S_IMODE(os.stat(ctx.checkpoint).st_mode)
    if os.name == "posix":
        # Owner-only permissions are expressible via chmod on POSIX.
        assert mode == 0o600
    else:
        # Windows stat cannot represent group/other bits; the file still must
        # not carry any executable bit.
        assert mode & 0o111 == 0


@pytest.mark.asyncio
async def test_empty_file_uses_zero_byte_finalization(ctx):
    data = write_source(ctx.source, 0)
    ctx.server.create_session(SESSION_URL, 0)

    result = await resumable_upload_file(
        ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
    )

    assert result["state"] == STATE_COMPLETE
    assert result["confirmed_offset"] == 0
    posts = ctx.server.events
    assert len(posts) == 1
    method, _, content_range, body_length, _ = posts[0]
    assert method == "POST"
    assert content_range == "bytes */0"
    assert body_length == 0
    assert ctx.server.session(SESSION_URL)["done"] is True
    assert bytes(ctx.server.session(SESSION_URL)["received"]) == data


@pytest.mark.asyncio
async def test_partial_receive_continues_from_server_range(ctx):
    size = 3 * GCS_MIN_BLOCK_SIZE
    data = write_source(ctx.source, size)
    partial = GCS_MIN_BLOCK_SIZE // 2
    ctx.server.create_session(SESSION_URL, size, partial_first=partial)

    result = await resumable_upload_file(
        ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
    )

    assert result["state"] == STATE_COMPLETE
    assert bytes(ctx.server.session(SESSION_URL)["received"]) == data

    posts = ctx.server.data_requests()
    # First chunk was only partially accepted.
    assert posts[0][2] == f"bytes 0-{GCS_MIN_BLOCK_SIZE - 1}/*"
    # The next request starts at the server-confirmed offset, not local sent.
    assert posts[1][2] == f"bytes {partial}-{partial + GCS_MIN_BLOCK_SIZE - 1}/*"
    # Every request is contiguous from a 256 KiB-aligned acknowledgement.
    assert posts[1][3] == GCS_MIN_BLOCK_SIZE


@pytest.mark.asyncio
async def test_308_without_range_rewinds_to_zero(ctx):
    size = 3 * GCS_MIN_BLOCK_SIZE
    data = write_source(ctx.source, size)
    ctx.server.create_session(SESSION_URL, size, no_range_once=True)

    result = await resumable_upload_file(
        ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
    )

    assert result["state"] == STATE_COMPLETE
    assert bytes(ctx.server.session(SESSION_URL)["received"]) == data
    posts = ctx.server.data_requests()
    ranges = [event[2] for event in posts]
    # First chunk acknowledged normally; the second 308 carried no Range.
    assert ranges[0].startswith("bytes 0-")
    assert ranges[1].startswith(f"bytes {GCS_MIN_BLOCK_SIZE}-")
    # Recovery rewinds to offset zero and re-sends the first chunk.
    assert ranges[2].startswith("bytes 0-")
    payload = read_checkpoint(ctx.checkpoint)
    assert payload["confirmed_offset"] == size


@pytest.mark.asyncio
async def test_duplicate_range_confirmation_is_idempotent_and_bounded(ctx):
    size = 3 * GCS_MIN_BLOCK_SIZE
    data = write_source(ctx.source, size)
    # Report the first 256 KiB as confirmed twice before making progress.
    ctx.server.create_session(SESSION_URL, size, stall=(2, GCS_MIN_BLOCK_SIZE))

    result = await resumable_upload_file(
        ctx.fs,
        str(ctx.source),
        RPATH,
        str(ctx.checkpoint),
        **upload_kwargs(max_attempts=6),
    )

    assert result["state"] == STATE_COMPLETE
    assert bytes(ctx.server.session(SESSION_URL)["received"]) == data
    ranges = [event[2] for event in ctx.server.data_requests()]
    # The stalled chunk was re-sent with the identical Content-Range after the
    # duplicate acknowledgement, and the confirmed offset never went backwards.
    assert ranges[1] == ranges[2]
    assert all(int(rng.split(" ")[1].split("-")[0]) >= 0 for rng in ranges)
    payload = read_checkpoint(ctx.checkpoint)
    assert payload["confirmed_offset"] == size


@pytest.mark.asyncio
async def test_no_progress_is_eventually_bounded(ctx):
    size = 2 * GCS_MIN_BLOCK_SIZE
    write_source(ctx.source, size)
    ctx.server.create_session(SESSION_URL, size, always_stall=GCS_MIN_BLOCK_SIZE)

    with pytest.raises(
        gcsfs.resumable.ResumableUploadError, match="no confirmed progress"
    ):
        await resumable_upload_file(
            ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
        )

    # Checkpoint is retained at the last server-confirmed offset.
    payload = read_checkpoint(ctx.checkpoint)
    assert payload["state"] == STATE_UPLOADING
    assert payload["confirmed_offset"] == GCS_MIN_BLOCK_SIZE
    assert (
        payload["source"]["prefix_digest"]
        == hashlib.sha256(
            write_source(ctx.source, size)[:GCS_MIN_BLOCK_SIZE]
        ).hexdigest()
    )


# ---------------------------------------------------------------------------
# Restart behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restart_queries_status_before_resuming_without_new_session(ctx):
    size = 4 * GCS_MIN_BLOCK_SIZE
    data = write_source(ctx.source, size)
    ctx.server.create_session(SESSION_URL, size, always_stall=GCS_MIN_BLOCK_SIZE)

    with pytest.raises(gcsfs.resumable.ResumableUploadError):
        await resumable_upload_file(
            ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
        )
    events_after_first_run = len(ctx.server.events)
    assert len(ctx.server.initiations) == 1

    # Simulate a process restart: fresh call, same checkpoint, healthy server.
    del ctx.server.session(SESSION_URL)["flags"]["always_stall"]
    result = await resumable_upload_file(
        ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
    )

    assert result["state"] == STATE_COMPLETE
    assert len(ctx.server.initiations) == 1
    new_events = ctx.server.events[events_after_first_run:]
    # The very first post-restart call is a status query, never a session POST
    # rooted at a local "sent" length.
    assert new_events[0][2].startswith("bytes */")
    assert new_events[0][3] == 0
    assert bytes(ctx.server.session(SESSION_URL)["received"]) == data


@pytest.mark.asyncio
async def test_restart_complete_checkpoint_short_circuits(ctx):
    size = GCS_MIN_BLOCK_SIZE
    data = write_source(ctx.source, size)
    ctx.server.create_session(SESSION_URL, size)
    await resumable_upload_file(
        ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
    )
    events = len(ctx.server.events)

    result = await resumable_upload_file(
        ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
    )

    assert result["state"] == STATE_COMPLETE
    assert len(ctx.server.events) == events  # no network at all
    assert len(ctx.server.initiations) == 1
    assert bytes(ctx.server.session(SESSION_URL)["received"]) == data


@pytest.mark.asyncio
async def test_final_disconnect_uncertain_then_resume_resends(ctx):
    size = 2 * GCS_MIN_BLOCK_SIZE + 777
    data = write_source(ctx.source, size)
    ctx.server.create_session(SESSION_URL, size, disconnect_final=True)

    with pytest.raises(ResumableUploadFinalUncertain):
        await resumable_upload_file(
            ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
        )

    payload = read_checkpoint(ctx.checkpoint)
    assert payload["state"] == STATE_FINAL_UNCERTAIN
    assert ctx.server.deletes == []  # nothing is cancelled on uncertainty

    ctx.server.session(SESSION_URL)["flags"]["disconnect_final"] = False
    result = await resumable_upload_file(
        ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
    )

    assert result["state"] == STATE_COMPLETE
    assert bytes(ctx.server.session(SESSION_URL)["received"]) == data
    assert len(ctx.server.initiations) == 1
    assert ctx.server.deletes == []


@pytest.mark.asyncio
async def test_final_response_lost_after_server_completed_resumes_via_query(ctx):
    size = GCS_MIN_BLOCK_SIZE + 99
    data = write_source(ctx.source, size)
    ctx.server.create_session(SESSION_URL, size, complete_then_disconnect=True)

    with pytest.raises(ResumableUploadFinalUncertain):
        await resumable_upload_file(
            ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
        )

    # Restart: the status query reports the object already exists.
    result = await resumable_upload_file(
        ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
    )

    assert result["state"] == STATE_COMPLETE
    assert result["generation"] == "9001"
    assert bytes(ctx.server.session(SESSION_URL)["received"]) == data
    assert len(ctx.server.initiations) == 1
    assert ctx.server.deletes == []


@pytest.mark.asyncio
async def test_transport_retry_is_bounded_and_recovers(ctx):
    size = 2 * GCS_MIN_BLOCK_SIZE
    data = write_source(ctx.source, size)
    ctx.server.create_session(SESSION_URL, size, disconnect_first_n=1)

    result = await resumable_upload_file(
        ctx.fs,
        str(ctx.source),
        RPATH,
        str(ctx.checkpoint),
        **upload_kwargs(max_attempts=3),
    )

    assert result["state"] == STATE_COMPLETE
    assert bytes(ctx.server.session(SESSION_URL)["received"]) == data


# ---------------------------------------------------------------------------
# Identity and target validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_source_size_change_rejected_before_network(ctx):
    size = 2 * GCS_MIN_BLOCK_SIZE
    write_source(ctx.source, size)
    ctx.server.create_session(SESSION_URL, size, always_stall=GCS_MIN_BLOCK_SIZE)
    with pytest.raises(gcsfs.resumable.ResumableUploadError):
        await resumable_upload_file(
            ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
        )
    events_before = len(ctx.server.events)

    with open(ctx.source, "ab") as handle:
        handle.write(b"extra-bytes")

    with pytest.raises(ResumableUploadSourceChanged):
        await resumable_upload_file(
            ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
        )
    assert len(ctx.server.events) == events_before  # no network contact


@pytest.mark.asyncio
async def test_source_content_change_rejected_before_network(ctx):
    size = 2 * GCS_MIN_BLOCK_SIZE
    data = write_source(ctx.source, size)
    ctx.server.create_session(SESSION_URL, size, always_stall=GCS_MIN_BLOCK_SIZE)
    with pytest.raises(gcsfs.resumable.ResumableUploadError):
        await resumable_upload_file(
            ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
        )
    events_before = len(ctx.server.events)

    corrupted = bytearray(data)
    corrupted[10] ^= 0xFF
    with open(ctx.source, "wb") as handle:
        handle.write(corrupted)
    os.utime(ctx.source, ns=(1, 1))

    with pytest.raises(ResumableUploadSourceChanged):
        await resumable_upload_file(
            ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
        )
    assert len(ctx.server.events) == events_before


@pytest.mark.asyncio
async def test_target_parameter_mismatch_rejected_before_network(ctx):
    size = 2 * GCS_MIN_BLOCK_SIZE
    write_source(ctx.source, size)
    ctx.server.create_session(SESSION_URL, size)
    await resumable_upload_file(
        ctx.fs,
        str(ctx.source),
        RPATH,
        str(ctx.checkpoint),
        **upload_kwargs(content_type="text/plain", metadata={"a": "b"}),
    )
    events_before = len(ctx.server.events)

    with pytest.raises(ResumableUploadTargetMismatch):
        await resumable_upload_file(
            ctx.fs,
            str(ctx.source),
            RPATH,
            str(ctx.checkpoint),
            **upload_kwargs(content_type="application/json", metadata={"a": "b"}),
        )
    with pytest.raises(ResumableUploadTargetMismatch):
        await resumable_upload_file(
            ctx.fs,
            str(ctx.source),
            RPATH,
            str(ctx.checkpoint),
            **upload_kwargs(content_type="text/plain", metadata={"a": "c"}),
        )
    assert len(ctx.server.events) == events_before


@pytest.mark.asyncio
async def test_requester_pays_is_frozen_and_validated(ctx):
    size = GCS_MIN_BLOCK_SIZE
    data = write_source(ctx.source, size)
    rp_fs = ctx.make_fs(requester_pays="billed-project")
    ctx.server.create_session(SESSION_URL, size)

    result = await resumable_upload_file(
        rp_fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
    )
    assert result["state"] == STATE_COMPLETE
    assert bytes(ctx.server.session(SESSION_URL)["received"]) == data
    # userProject is attached to every session request.
    assert all(
        event[4].get("userProject") == "billed-project" for event in ctx.server.events
    )
    payload = read_checkpoint(ctx.checkpoint)
    assert payload["target"]["requester_pays"] == "billed-project"

    other_fs = ctx.make_fs(requester_pays=False)
    with pytest.raises(ResumableUploadTargetMismatch):
        await resumable_upload_file(
            other_fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
        )


@pytest.mark.asyncio
async def test_conditional_creation_params_frozen_at_initiation(ctx):
    size = GCS_MIN_BLOCK_SIZE + 5
    data = write_source(ctx.source, size)
    ctx.server.create_session(SESSION_URL, size)

    await resumable_upload_file(
        ctx.fs,
        str(ctx.source),
        RPATH,
        str(ctx.checkpoint),
        **upload_kwargs(
            mode="create",
            if_generation_match=0,
            content_type="text/csv",
            metadata={"label": "v1"},
            fixed_key_metadata={"cache_control": "no-cache"},
            kms_key_name="projects/p/locations/l/keyRings/r/cryptoKeys/k",
        ),
    )

    assert len(ctx.server.initiations) == 1
    initiated = ctx.server.initiations[0]
    assert initiated["mode"] == "create"
    assert initiated["if_generation_match"] == 0
    assert initiated["content_type"] == "text/csv"
    assert initiated["metadata"] == {"label": "v1"}
    assert initiated["fixed_key_metadata"] == {"cache_control": "no-cache"}
    assert initiated["kms_key_name"].endswith("cryptoKeys/k")

    payload = read_checkpoint(ctx.checkpoint)
    assert payload["target"]["if_generation_match"] == 0
    assert payload["target"]["content_type"] == "text/csv"
    assert bytes(ctx.server.session(SESSION_URL)["received"]) == data


@pytest.mark.asyncio
async def test_stream_input_is_rejected(ctx):
    with pytest.raises(TypeError, match="re-openable local file path"):
        await resumable_upload_file(
            ctx.fs, io.BytesIO(b"x"), RPATH, str(ctx.checkpoint)
        )


# ---------------------------------------------------------------------------
# Session expiry and checkpoint integrity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [404, 410])
@pytest.mark.asyncio
async def test_expired_session_does_not_create_replacement(ctx, status):
    size = 2 * GCS_MIN_BLOCK_SIZE
    write_source(ctx.source, size)
    ctx.server.create_session(SESSION_URL, size, always_stall=GCS_MIN_BLOCK_SIZE)
    with pytest.raises(gcsfs.resumable.ResumableUploadError):
        await resumable_upload_file(
            ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
        )

    ctx.server.session(SESSION_URL)["flags"] = {"expire": status}
    with pytest.raises(ResumableUploadSessionExpired, match=str(status)):
        await resumable_upload_file(
            ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
        )

    assert len(ctx.server.initiations) == 1  # no implicit replacement session
    assert ctx.checkpoint.exists()  # caller keeps the evidence


@pytest.mark.asyncio
async def test_corrupt_checkpoint_blocks_all_network(ctx):
    size = GCS_MIN_BLOCK_SIZE
    write_source(ctx.source, size)
    ctx.server.create_session(SESSION_URL, size)
    await resumable_upload_file(
        ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
    )

    raw = bytearray(ctx.checkpoint.read_bytes())
    raw[-5] ^= 0xFF
    ctx.checkpoint.write_bytes(bytes(raw))
    events_before = len(ctx.server.events)

    with pytest.raises(ResumableUploadCheckpointError):
        await resumable_upload_file(
            ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
        )
    assert len(ctx.server.events) == events_before

    ctx.checkpoint.write_bytes(b"{not valid json")
    with pytest.raises(ResumableUploadCheckpointError):
        load_checkpoint(str(ctx.checkpoint))


# ---------------------------------------------------------------------------
# Cancellation protocol
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_retain_keeps_session_and_checkpoint(ctx):
    size = 2 * GCS_MIN_BLOCK_SIZE
    write_source(ctx.source, size)
    ctx.server.create_session(SESSION_URL, size, always_stall=GCS_MIN_BLOCK_SIZE)
    with pytest.raises(gcsfs.resumable.ResumableUploadError):
        await resumable_upload_file(
            ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
        )

    result = await cancel_resumable_upload(ctx.fs, str(ctx.checkpoint), abandon=False)

    assert result["abandoned"] is False
    assert ctx.checkpoint.exists()
    assert ctx.server.deletes == []
    assert SESSION_URL in ctx.server.sessions


@pytest.mark.asyncio
async def test_cancel_abandon_deletes_only_this_session(ctx):
    size = 2 * GCS_MIN_BLOCK_SIZE
    write_source(ctx.source, size)
    ctx.server.create_session(SESSION_URL, size)
    await resumable_upload_file(
        ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
    )
    # A second, unrelated session must never be touched.
    other_url = SESSION_URL.replace(UPLOAD_ID, "session-other-999")
    ctx.server.create_session(other_url, 10)

    payload = read_checkpoint(ctx.checkpoint)
    # Reset to an in-progress state so a DELETE is actually required.
    payload["state"] = STATE_UPLOADING
    gcsfs.resumable.save_checkpoint(str(ctx.checkpoint), payload)

    result = await cancel_resumable_upload(ctx.fs, str(ctx.checkpoint), abandon=True)

    assert result["abandoned"] is True
    assert not ctx.checkpoint.exists()
    assert ctx.server.deletes == [gcsfs.core._strip_generation_match(SESSION_URL)]
    assert "ifGenerationMatch" not in ctx.server.deletes[0]
    assert SESSION_URL not in ctx.server.sessions
    assert other_url in ctx.server.sessions


@pytest.mark.asyncio
async def test_cancel_refused_while_final_status_uncertain(ctx):
    size = GCS_MIN_BLOCK_SIZE
    write_source(ctx.source, size)
    ctx.server.create_session(SESSION_URL, size, disconnect_final=True)
    with pytest.raises(ResumableUploadFinalUncertain):
        await resumable_upload_file(
            ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
        )

    with pytest.raises(ResumableUploadStateError):
        await cancel_resumable_upload(ctx.fs, str(ctx.checkpoint), abandon=True)
    assert ctx.server.deletes == []
    assert ctx.checkpoint.exists()


@pytest.mark.asyncio
async def test_cancel_completed_upload_removes_checkpoint_without_delete(ctx):
    size = GCS_MIN_BLOCK_SIZE
    write_source(ctx.source, size)
    ctx.server.create_session(SESSION_URL, size)
    await resumable_upload_file(
        ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
    )

    result = await cancel_resumable_upload(ctx.fs, str(ctx.checkpoint), abandon=True)

    assert result["abandoned"] is True
    assert not ctx.checkpoint.exists()
    assert ctx.server.deletes == []


# ---------------------------------------------------------------------------
# Status helper and secrecy guarantees
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resumable_upload_status_refreshes_confirmed_offset(ctx):
    size = 2 * GCS_MIN_BLOCK_SIZE
    data = write_source(ctx.source, size)
    ctx.server.create_session(SESSION_URL, size, always_stall=GCS_MIN_BLOCK_SIZE)
    with pytest.raises(gcsfs.resumable.ResumableUploadError):
        await resumable_upload_file(
            ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
        )

    # The stale 308 acks under-reported progress, but the server actually stored
    # both chunks. The status query reveals the real held length and the
    # checkpoint must catch up to it with a verifiable digest.
    result = await resumable_upload_status(ctx.fs, str(ctx.checkpoint))
    assert result["state"] == STATE_UPLOADING
    assert result["confirmed_offset"] == size
    payload = read_checkpoint(ctx.checkpoint)
    assert payload["confirmed_offset"] == size
    assert payload["source"]["prefix_digest"] == hashlib.sha256(data).hexdigest()
    # The first call of the refresh is an empty status query.
    assert ctx.server.status_queries()[-1][3] == 0


@pytest.mark.asyncio
async def test_session_url_never_in_logs_or_exception_text(ctx, caplog):
    size = 2 * GCS_MIN_BLOCK_SIZE
    write_source(ctx.source, size)
    ctx.server.create_session(SESSION_URL, size, disconnect_first_n=1)
    caplog.set_level(logging.DEBUG, logger="gcsfs.resumable")

    await resumable_upload_file(
        ctx.fs,
        str(ctx.source),
        RPATH,
        str(ctx.checkpoint),
        **upload_kwargs(max_attempts=3),
    )

    for record in caplog.records:
        assert UPLOAD_ID not in record.getMessage()
        assert SESSION_URL not in record.getMessage()

    # An in-progress checkpoint whose session has expired: exception text must
    # not leak the session URL.
    payload = read_checkpoint(ctx.checkpoint)
    payload["state"] = STATE_UPLOADING
    gcsfs.resumable.save_checkpoint(str(ctx.checkpoint), payload)
    ctx.server.session(SESSION_URL)["flags"] = {"expire": 410}
    with pytest.raises(Exception) as exc_info:
        await resumable_upload_file(
            ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
        )
    assert SESSION_URL not in str(exc_info.value)
    assert UPLOAD_ID not in str(exc_info.value)

    raw = bytearray(ctx.checkpoint.read_bytes())
    raw[-5] ^= 0xFF
    ctx.checkpoint.write_bytes(bytes(raw))
    with pytest.raises(Exception) as exc_info:
        await resumable_upload_file(
            ctx.fs, str(ctx.source), RPATH, str(ctx.checkpoint), **upload_kwargs()
        )
    assert SESSION_URL not in str(exc_info.value)
    assert UPLOAD_ID not in str(exc_info.value)


@pytest.mark.asyncio
async def test_sync_wrapper_runs_end_to_end(ctx):
    size = 2 * GCS_MIN_BLOCK_SIZE
    data = write_source(ctx.source, size)
    ctx.server.create_session(SESSION_URL, size)

    result = ctx.fs.resumable_upload_file(
        str(ctx.source),
        RPATH,
        str(ctx.checkpoint),
        chunksize=GCS_MIN_BLOCK_SIZE,
        max_attempts=3,
    )

    assert result["state"] == STATE_COMPLETE
    assert bytes(ctx.server.session(SESSION_URL)["received"]) == data
    assert (
        ctx.fs.resumable_upload_status(str(ctx.checkpoint))["state"] == STATE_COMPLETE
    )
    assert (
        ctx.fs.cancel_resumable_upload(str(ctx.checkpoint), abandon=False)["abandoned"]
        is False
    )
