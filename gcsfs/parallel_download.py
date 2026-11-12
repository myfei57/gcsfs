"""
Parallel, out-of-order range download with CRC32C chunk combination.

The orchestrator in this module downloads a GCS object as a fixed set of
non-overlapping byte ranges and validates it against the whole-object
CRC32C published in the object metadata, without ever re-reading the data
sequentially or assembling it in memory:

* every range request is pinned to one frozen object ``generation`` and only
  the bytes actually returned for that exact interval are accepted;
* a raw CRC32C is computed per chunk on the received bytes and chunks are
  combined through the CRC32C polynomial algebra (see
  :mod:`gcsfs.checkers`), so completions may arrive in any order;
* short reads, overlapping/hole intervals, range errors, generation changes
  and write failures invalidate the download instead of publishing a file;
* the assembled bytes are written to a temporary file and the final target is
  only published through an atomic replace once every interval is covered and
  the combined checksum matches, leaving a pre-existing target untouched on
  any failure.

Concurrency, buffered chunks awaiting a write and the number of download
tasks are all bounded explicitly.
"""

import asyncio

from .checkers import Crc32cChunkHasher, Crc32cCombiner, crcmod
from .retry import ChecksumError


class ShortReadError(ChecksumError):
    """A range response did not return exactly the requested byte interval."""


def validate_crc32c_download_details(details):
    """Apply the explicit preconditions for parallel CRC32C validation.

    The mode never silently degrades to size-only verification: a missing
    ``crcmod`` dependency, an object without a whole-object ``crc32c``
    checksum or metadata without a pinnable ``generation`` are all rejected.
    """
    if crcmod is None:
        raise ImportError(
            "The python package `crcmod` is required for parallel CRC32C "
            "validation. This can be installed with `pip install gcsfs[crc]`"
        )

    expected_crc32c = details.get("crc32c")
    if not expected_crc32c:
        raise ChecksumError(
            "Object metadata has no crc32c checksum; parallel CRC32C "
            "validation was explicitly requested and will not silently fall "
            "back to size-only verification"
        )

    generation = details.get("generation")
    if not generation:
        raise ChecksumError(
            "Object metadata has no generation; parallel CRC32C validation "
            "requires pinning every range request to one object generation"
        )

    return expected_crc32c, generation


class _DownloadStats:
    """Instrumentation for concurrency and memory-bound guarantees."""

    def __init__(self):
        self.in_flight_fetches = 0
        self.peak_in_flight_fetches = 0
        self.buffered_chunks = 0
        self.peak_buffered_chunks = 0
        self.fetched_bytes = 0
        self.written_bytes = 0
        self.completion_order = []

    def enter_fetch(self):
        self.in_flight_fetches += 1
        self.peak_in_flight_fetches = max(
            self.peak_in_flight_fetches, self.in_flight_fetches
        )

    def exit_fetch(self):
        self.in_flight_fetches -= 1

    def enter_buffer(self):
        self.buffered_chunks += 1
        self.peak_buffered_chunks = max(self.peak_buffered_chunks, self.buffered_chunks)

    def exit_buffer(self):
        self.buffered_chunks -= 1


class ParallelCrc32cDownloader:
    """Download an object through validated, out-of-order parallel ranges.

    Parameters
    ----------
    fetcher: awaitable ``fetcher(start, end, generation) -> bytes``
        Returns exactly the bytes of the half-open interval ``[start, end)``
        read from the pinned ``generation``. It owns the network response
        body and must close/release it on error or cancellation.
    writer: callable ``writer(offset, data) -> None``
        Synchronous, thread-safe write of ``data`` at absolute ``offset`` of
        the temporary file; executed through :func:`asyncio.to_thread`.
    commit: callable ``commit() -> None``
        Publishes the temporary file as the final target (atomic replace).
    abort: callable ``abort() -> None``
        Releases file handles and removes the temporary file; never raises.
    ranges: list of ``(start, length)``
        Contiguous, non-overlapping plan covering exactly ``total_size``.
    total_size: int
        Whole-object byte size from the pinned metadata.
    generation:
        Object generation every request must read.
    expected_crc32c: str
        Whole-object GCS big-endian base64 CRC32C from the pinned metadata.
    concurrency: int
        Upper bound on simultaneously in-flight range requests.
    max_pending_writes: int
        Upper bound on fetched chunks whose bytes are buffered for writing.
    callback: optional fsspec progress callback.
    """

    #: Hard cap on the number of chunk tasks created for one download.
    MAX_TASKS = 32

    def __init__(
        self,
        *,
        fetcher,
        writer,
        commit,
        abort,
        ranges,
        total_size,
        generation,
        expected_crc32c,
        concurrency,
        max_pending_writes,
        callback=None,
    ):
        if not generation:
            raise ChecksumError("A pinned object generation is required")
        if not expected_crc32c:
            raise ChecksumError("A whole-object crc32c checksum is required")
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if max_pending_writes < 1:
            raise ValueError("max_pending_writes must be >= 1")
        if len(ranges) > self.MAX_TASKS:
            raise ValueError(
                f"Refusing to schedule {len(ranges)} chunk tasks; cap is "
                f"{self.MAX_TASKS}"
            )

        self._fetcher = fetcher
        self._writer = writer
        self._commit = commit
        self._abort = abort
        self.total_size = int(total_size)
        self.generation = generation
        self.expected_crc32c = expected_crc32c
        self.concurrency = concurrency
        self.max_pending_writes = max_pending_writes
        self.ranges = list(ranges)
        self.callback = callback

        # Rejects plans with overlaps, holes or intervals outside [0, size).
        self._combiner = Crc32cCombiner(self.ranges, self.total_size)
        self.stats = _DownloadStats()

    async def run(self):
        if self.callback is not None:
            self.callback.set_size(self.total_size)

        try:
            # Empty object: there are no ranges; validate the empty-input
            # checksum and publish an empty target atomically.
            if not self.ranges:
                self._combiner.validate_json_response({"crc32c": self.expected_crc32c})
                await asyncio.to_thread(self._commit)
                return

            fetch_slots = asyncio.Semaphore(self.concurrency)
            # Each permit represents one chunk's worth of buffered bytes
            # waiting on (or performing) a disk write.
            write_slots = asyncio.Semaphore(self.max_pending_writes)

            tasks = [
                asyncio.create_task(
                    self._process_chunk(index, fetch_slots, write_slots),
                    name=f"crc32c-range-{index}",
                )
                for index in range(len(self.ranges))
            ]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                # Cancel in-flight network requests so their response bodies
                # are released, then drain every task so to_thread writes are
                # reclaimed rather than leaked.
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

            # Full interval coverage and per-chunk validity are guaranteed here;
            # only publish if the polynomial-combined checksum matches the object.
            self._combiner.validate_json_response({"crc32c": self.expected_crc32c})
            await asyncio.to_thread(self._commit)
        except BaseException:
            # Any failure (transport, short read, write, checksum mismatch,
            # cancellation) rolls the temporary file back; the caller keeps
            # ownership of the final target.
            await asyncio.to_thread(self._abort)
            raise

    async def _process_chunk(self, index, fetch_slots, write_slots):
        start, length = self.ranges[index]
        end = start + length

        # Bound the number of simultaneous network responses. The write permit
        # is acquired while still holding the fetch slot, so the number of
        # chunks whose bytes are buffered (awaiting or performing a write) is
        # at most concurrency + max_pending_writes.
        async with fetch_slots:
            self.stats.enter_fetch()
            try:
                # Every request carries the same pinned generation.
                data = await self._fetcher(start, end, self.generation)
            finally:
                self.stats.exit_fetch()

            # A short read (or over-read) can never be a valid block. The
            # whole-object hash in range response headers is intentionally
            # never consulted: the chunk checksum is computed locally on the
            # bytes actually received.
            if (
                not isinstance(data, (bytes, bytearray, memoryview))
                or len(data) != length
            ):
                received = (
                    len(data) if isinstance(data, (bytes, bytearray, memoryview)) else 0
                )
                raise ShortReadError(
                    f"Range [{start}, {end}) returned {received} of {length} bytes"
                )

            hasher = Crc32cChunkHasher()
            hasher.update(bytes(data))
            raw_crc = hasher.raw_value
            self.stats.fetched_bytes += length

            self.stats.enter_buffer()
            try:
                await write_slots.acquire()
                try:
                    await asyncio.to_thread(self._writer, start, bytes(data))
                finally:
                    write_slots.release()
            finally:
                self.stats.exit_buffer()

        # The chunk is marked complete only after the write has succeeded;
        # a failed write never registers a (start, end, crc) block.
        self._combiner.add_chunk(index, start, end, raw_crc)
        self.stats.written_bytes += length
        self.stats.completion_order.append(index)
        if self.callback is not None:
            self.callback.relative_update(length)
