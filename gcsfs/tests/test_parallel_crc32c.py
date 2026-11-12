"""Tests for the parallel CRC32C chunk-combination download mode.

These tests are fully offline: the network, disk writes and metadata are
injected fakes, which makes failure injection (bit flips, generation changes,
short reads and write errors) and concurrency accounting deterministic.
"""

import asyncio
import base64
import os
import random
import types
from unittest import mock

import pytest

from gcsfs.checkers import Crc32cChecker, crcmod
from gcsfs.concurrency import split_range
from gcsfs.core import GCSFileSystem
from gcsfs.parallel_download import (
    ParallelCrc32cDownloader,
    ShortReadError,
    validate_crc32c_download_details,
)
from gcsfs.retry import ChecksumError, GenerationMismatchError

PINNED_GENERATION = "123456"
OTHER_GENERATION = "987654"

pytestmark = pytest.mark.skipif(crcmod is None, reason="crcmod is not installed")


def serial_crc32c_b64(data: bytes) -> str:
    """Independent serial baseline using the existing GCS-convention checker."""
    checker = Crc32cChecker()
    checker.update(data)
    return base64.b64encode(checker.crc32c.digest()).decode()


class FakeFetcher:
    """Serves byte ranges from in-memory data with injectable failure modes."""

    def __init__(
        self,
        data,
        ranges,
        *,
        delay_seed=0,
        flip_index=None,
        short_read_index=None,
        generation_error_index=None,
        block_first=False,
    ):
        self.data = data
        self.ranges = ranges
        self.delay_seed = delay_seed
        self.flip_index = flip_index
        self.short_read_index = short_read_index
        self.generation_error_index = generation_error_index
        self.block_first = block_first
        self.calls = []
        self.in_flight = 0
        self.peak_in_flight = 0
        self._blocker = asyncio.Event()

    def release_blocker(self):
        self._blocker.set()

    async def __call__(self, start, end, generation):
        # Every request must carry the single pinned generation.
        assert generation == PINNED_GENERATION
        self.calls.append((start, end, generation))
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            index = next(i for i, (s, length) in enumerate(self.ranges) if s == start)
            if self.block_first and index == 0:
                await self._blocker.wait()

            # Index-dependent jitter forces later chunks to complete first.
            rng = random.Random(self.delay_seed + index)
            await asyncio.sleep(0.002 * (1 + rng.randrange(5)))

            if index == self.generation_error_index:
                raise GenerationMismatchError("b/o", PINNED_GENERATION)
            if index == self.short_read_index:
                return self.data[start : end - 1]  # one byte short
            if index == self.flip_index:
                corrupted = bytearray(self.data[start:end])
                corrupted[0] ^= 0xFF
                return bytes(corrupted)
            return self.data[start:end]
        finally:
            self.in_flight -= 1


class TargetHarness:
    """In-memory file target with pre-existing content and failure injection."""

    def __init__(self, tmp_path, size, *, old_content=b"pre-existing-target"):
        self.target = tmp_path / "object.bin"
        self.target.write_bytes(old_content)
        self.old_content = old_content
        self.buffer = bytearray(size)
        self.committed = False
        self.aborted = False
        self.write_error_offset = None

    def writer(self, offset, data):
        if offset == self.write_error_offset:
            raise OSError("simulated disk full")
        self.buffer[offset : offset + len(data)] = data

    def commit(self):
        self.target.write_bytes(bytes(self.buffer))
        self.committed = True

    def abort(self):
        self.aborted = True

    def assert_untouched(self):
        assert self.target.read_bytes() == self.old_content
        assert not self.committed


def build_downloader(
    data,
    ranges,
    fetcher,
    harness,
    *,
    concurrency,
    max_pending_writes,
    expected_crc32c=None,
    generation=PINNED_GENERATION,
):
    return ParallelCrc32cDownloader(
        fetcher=fetcher,
        writer=harness.writer,
        commit=harness.commit,
        abort=harness.abort,
        ranges=ranges,
        total_size=len(data),
        generation=generation,
        expected_crc32c=expected_crc32c or serial_crc32c_b64(data),
        concurrency=concurrency,
        max_pending_writes=max_pending_writes,
    )


def assert_calls_match_plan(fetcher, ranges):
    assert len(fetcher.calls) == len(ranges)
    planned = {(start, start + length) for start, length in ranges}
    seen = {(start, end) for start, end, _ in fetcher.calls}
    assert seen == planned
    assert {generation for _, _, generation in fetcher.calls} == {PINNED_GENERATION}


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", range(8))
async def test_random_splits_and_random_completion_order(tmp_path, seed):
    rng = random.Random(seed)
    size = rng.randint(1, 5000)
    data = os.urandom(size)
    concurrency = rng.randint(2, 6)
    chunk_size = rng.randint(1, max(1, size // 2))
    ranges = split_range(size, concurrency, max(1, chunk_size))

    fetcher = FakeFetcher(data, ranges, delay_seed=seed)
    harness = TargetHarness(tmp_path, size)
    downloader = build_downloader(
        data,
        ranges,
        fetcher,
        harness,
        concurrency=concurrency,
        max_pending_writes=concurrency,
    )

    await downloader.run()

    # The published file must be byte-identical without any in-memory join.
    assert harness.committed
    assert harness.target.read_bytes() == data
    assert not harness.aborted
    assert_calls_match_plan(fetcher, ranges)

    # Every interval got covered exactly once, in some completion order.
    assert sorted(downloader.stats.completion_order) == list(range(len(ranges)))
    assert downloader.stats.fetched_bytes == size
    assert downloader.stats.written_bytes == size

    # Across the parametrized seeds, completions really do arrive out of order.
    if seed == 0:
        # Re-run once with strongly inverted delays to prove ordering freedom.
        sub_dir = tmp_path / "sub"
        sub_dir.mkdir()
        inverted = FakeFetcher(data, ranges, delay_seed=100)
        inverted_harness = TargetHarness(sub_dir, size)
        second = build_downloader(
            data,
            ranges,
            inverted,
            inverted_harness,
            concurrency=concurrency,
            max_pending_writes=concurrency,
        )
        await second.run()
        shuffled_seen = (
            second.stats.completion_order != list(range(len(ranges)))
            or len(ranges) == 1
        )
        assert shuffled_seen


@pytest.mark.asyncio
async def test_bit_flip_in_one_chunk_fails_checksum_and_keeps_old_target(tmp_path):
    data = os.urandom(3000)
    ranges = split_range(len(data), 4, 300)
    fetcher = FakeFetcher(data, ranges, flip_index=2)
    harness = TargetHarness(tmp_path, len(data))
    downloader = build_downloader(
        data, ranges, fetcher, harness, concurrency=4, max_pending_writes=2
    )

    with pytest.raises(ChecksumError):
        await downloader.run()

    harness.assert_untouched()
    assert harness.aborted


@pytest.mark.asyncio
async def test_generation_change_is_rejected_and_target_kept(tmp_path):
    data = os.urandom(2000)
    ranges = split_range(len(data), 3, 256)
    fetcher = FakeFetcher(data, ranges, generation_error_index=1)
    harness = TargetHarness(tmp_path, len(data))
    downloader = build_downloader(
        data, ranges, fetcher, harness, concurrency=3, max_pending_writes=2
    )

    with pytest.raises(GenerationMismatchError):
        await downloader.run()

    harness.assert_untouched()
    # Every request that went out still used the pinned generation.
    assert {generation for _, _, generation in fetcher.calls} == {PINNED_GENERATION}


@pytest.mark.asyncio
async def test_short_read_is_not_a_valid_chunk(tmp_path):
    data = os.urandom(2000)
    ranges = split_range(len(data), 3, 256)
    fetcher = FakeFetcher(data, ranges, short_read_index=0)
    harness = TargetHarness(tmp_path, len(data))
    downloader = build_downloader(
        data, ranges, fetcher, harness, concurrency=3, max_pending_writes=2
    )

    with pytest.raises(ShortReadError):
        await downloader.run()

    harness.assert_untouched()
    assert 0 not in downloader.stats.completion_order


@pytest.mark.asyncio
async def test_write_failure_does_not_mark_chunk_complete(tmp_path):
    data = os.urandom(2000)
    ranges = split_range(len(data), 4, 256)
    failing_offset, failing_length = ranges[1]
    fetcher = FakeFetcher(data, ranges)
    harness = TargetHarness(tmp_path, len(data))
    harness.write_error_offset = failing_offset
    downloader = build_downloader(
        data, ranges, fetcher, harness, concurrency=4, max_pending_writes=2
    )

    with pytest.raises(OSError, match="simulated disk full"):
        await downloader.run()

    harness.assert_untouched()
    assert 1 not in downloader.stats.completion_order


@pytest.mark.asyncio
async def test_concurrency_buffer_and_task_caps_are_enforced(tmp_path):
    data = os.urandom(8000)
    concurrency = 2
    max_pending_writes = 2
    ranges = split_range(len(data), 8, 100)
    assert len(ranges) <= ParallelCrc32cDownloader.MAX_TASKS

    fetcher = FakeFetcher(data, ranges, delay_seed=3)
    harness = TargetHarness(tmp_path, len(data))
    downloader = build_downloader(
        data,
        ranges,
        fetcher,
        harness,
        concurrency=concurrency,
        max_pending_writes=max_pending_writes,
    )

    await downloader.run()

    assert fetcher.peak_in_flight <= concurrency
    assert downloader.stats.peak_in_flight_fetches <= concurrency
    # Buffers = fetch-slot holders awaiting a permit + permitted writes.
    assert downloader.stats.peak_buffered_chunks <= concurrency + max_pending_writes
    assert harness.target.read_bytes() == data

    # The task count hard cap cannot be bypassed through an oversized plan.
    with pytest.raises(ValueError, match="cap"):
        ParallelCrc32cDownloader(
            fetcher=fetcher,
            writer=harness.writer,
            commit=harness.commit,
            abort=harness.abort,
            ranges=[(i, 1) for i in range(ParallelCrc32cDownloader.MAX_TASKS + 1)],
            total_size=ParallelCrc32cDownloader.MAX_TASKS + 1,
            generation=PINNED_GENERATION,
            expected_crc32c="AAAAAA==",
            concurrency=concurrency,
            max_pending_writes=max_pending_writes,
        )


@pytest.mark.asyncio
async def test_cancellation_reclaims_tasks_and_keeps_target(tmp_path):
    data = os.urandom(2000)
    ranges = split_range(len(data), 3, 256)
    fetcher = FakeFetcher(data, ranges, block_first=True)
    harness = TargetHarness(tmp_path, len(data))
    downloader = build_downloader(
        data, ranges, fetcher, harness, concurrency=3, max_pending_writes=2
    )

    run_task = asyncio.create_task(downloader.run())
    await asyncio.sleep(0.05)
    run_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await run_task

    # All internal range tasks have been drained (no leaked tasks/threads).
    leftover = [
        t
        for t in asyncio.all_tasks()
        if t.get_name().startswith("crc32c-range-") and not t.done()
    ]
    assert leftover == []
    harness.assert_untouched()
    assert harness.aborted


def test_validate_details_explicit_rejection_policy(monkeypatch):
    # Missing crc dependency: refuse instead of downgrading to size checks.
    import gcsfs.parallel_download as parallel

    monkeypatch.setattr(parallel, "crcmod", None)
    with pytest.raises(ImportError, match="crcmod"):
        validate_crc32c_download_details({"crc32c": "x", "generation": "1", "size": 1})

    monkeypatch.undo()

    # Object without a whole-object checksum.
    with pytest.raises(ChecksumError, match="no crc32c checksum"):
        validate_crc32c_download_details({"generation": "1", "size": 1})

    # Metadata without a pinnable generation.
    with pytest.raises(ChecksumError, match="no generation"):
        validate_crc32c_download_details({"crc32c": "AAAAAA==", "size": 1})

    expected, generation = validate_crc32c_download_details(
        {"crc32c": "AAAAAA==", "generation": "7", "size": 0}
    )
    assert expected == "AAAAAA=="
    assert generation == "7"


@pytest.mark.asyncio
async def test_empty_object_publishes_empty_file(tmp_path):
    harness = TargetHarness(tmp_path, 0, old_content=b"")
    downloader = ParallelCrc32cDownloader(
        fetcher=mock.AsyncMock(),
        writer=harness.writer,
        commit=harness.commit,
        abort=harness.abort,
        ranges=[],
        total_size=0,
        generation=PINNED_GENERATION,
        expected_crc32c="AAAAAA==",
        concurrency=2,
        max_pending_writes=2,
    )
    await downloader.run()
    assert harness.committed
    assert harness.target.read_bytes() == b""

    # A wrong advertised checksum on an empty object still blocks publishing.
    bad_harness = TargetHarness(tmp_path, 0, old_content=b"")
    bad = ParallelCrc32cDownloader(
        fetcher=mock.AsyncMock(),
        writer=bad_harness.writer,
        commit=bad_harness.commit,
        abort=bad_harness.abort,
        ranges=[],
        total_size=0,
        generation=PINNED_GENERATION,
        expected_crc32c="DoesntMatter==",
        concurrency=2,
        max_pending_writes=2,
    )
    with pytest.raises(ChecksumError):
        await bad.run()
    assert not bad_harness.committed


class StubGCS:
    """Minimal stand-in for GCSFileSystem exercising the core wiring."""

    def __init__(
        self,
        data,
        details,
        *,
        served_generation=PINNED_GENERATION,
        flip_range=None,
    ):
        self.data = data
        self.details = details
        self.served_generation = served_generation
        self.flip_range = flip_range
        self.requests = []

    async def _info(self, path, **kwargs):
        return self.details

    def url(self, path, generation=None):
        return f"http://stub/download/{path}?generation={generation}"

    async def _process_limits(self, path, start, end):
        return f"bytes={start}-{end - 1}"

    async def _call(self, method, url, headers=None, **kwargs):
        self.requests.append((url, dict(headers or {})))
        rng = headers["Range"].removeprefix("bytes=")
        start, end_inclusive = (int(x) for x in rng.split("-"))
        payload = bytearray(self.data[start : end_inclusive + 1])
        if self.flip_range == start:
            payload[0] ^= 0xFF
        # The whole-object hash is echoed on every range response; it must
        # never be treated as the checksum of the returned byte range.
        response_headers = {
            "X-Goog-Generation": self.served_generation,
            "X-Goog-Hash": f"crc32c={serial_crc32c_b64(self.data)}",
        }
        return response_headers, bytes(payload)

    def _init_local_file(self, lpath, total_size):
        os.makedirs(os.path.dirname(lpath) or os.curdir, exist_ok=True)
        with open(lpath, "wb") as handle:
            handle.truncate(total_size)


@pytest.mark.asyncio
async def test_core_wiring_pins_generation_and_validates(tmp_path):
    data = os.urandom(4000)
    details = {
        "size": len(data),
        "generation": PINNED_GENERATION,
        "crc32c": serial_crc32c_b64(data),
    }
    stub = StubGCS(data, details)
    target = tmp_path / "downloaded.bin"

    await GCSFileSystem._get_file_crc32c_parallel(
        stub,
        "b/o",
        str(target),
        concurrency=3,
        chunk_size=512,
    )

    assert target.read_bytes() == data
    # Every range request targets the pinned generation and the ranges tile
    # the object without overlaps or holes.
    intervals = []
    for url, headers in stub.requests:
        assert f"generation={PINNED_GENERATION}" in url
        intervals.append(
            tuple(int(x) for x in headers["Range"].removeprefix("bytes=").split("-"))
        )
    intervals.sort()
    assert intervals[0][0] == 0
    assert intervals[-1][1] == len(data) - 1
    for (_, end_a), (start_b, _) in zip(intervals, intervals[1:]):
        assert start_b == end_a + 1


@pytest.mark.asyncio
async def test_core_wiring_rejects_changed_generation(tmp_path):
    data = os.urandom(2000)
    details = {
        "size": len(data),
        "generation": PINNED_GENERATION,
        "crc32c": serial_crc32c_b64(data),
    }
    stub = StubGCS(data, details, served_generation=OTHER_GENERATION)
    target = tmp_path / "downloaded.bin"
    target.write_bytes(b"old-version")

    with pytest.raises(GenerationMismatchError):
        await GCSFileSystem._get_file_crc32c_parallel(
            stub, "b/o", str(target), concurrency=2, chunk_size=512
        )

    assert target.read_bytes() == b"old-version"


@pytest.mark.asyncio
async def test_core_wiring_ignores_whole_object_hash_on_range_responses(tmp_path):
    data = os.urandom(2000)
    details = {
        "size": len(data),
        "generation": PINNED_GENERATION,
        "crc32c": serial_crc32c_b64(data),
    }
    # Corrupt the first range while the response still advertises the correct
    # whole-object hash: the local chunk checksum must win.
    stub = StubGCS(data, details, flip_range=0)
    target = tmp_path / "downloaded.bin"
    target.write_bytes(b"old-version")

    with pytest.raises(ChecksumError):
        await GCSFileSystem._get_file_crc32c_parallel(
            stub, "b/o", str(target), concurrency=2, chunk_size=512
        )

    assert target.read_bytes() == b"old-version"


@pytest.mark.asyncio
async def test_get_file_routing_preserves_target_on_validation_failure(tmp_path):
    target = tmp_path / "routed.bin"
    target.write_bytes(b"untouched")

    stub = types.SimpleNamespace()
    # Bind the real implementation without constructing a GCSFileSystem.
    stub._get_file_crc32c_parallel = types.MethodType(
        GCSFileSystem._get_file_crc32c_parallel, stub
    )

    async def fail_details(path, **kwargs):
        # Object carries no checksum: the explicit policy must reject.
        return {"size": 10, "generation": PINNED_GENERATION}

    stub._info = fail_details

    with pytest.raises(ChecksumError, match="no crc32c checksum"):
        await GCSFileSystem._get_file(
            stub, "b/o", str(target), crc32c_parallel=True, concurrency=2
        )

    assert target.read_bytes() == b"untouched"
