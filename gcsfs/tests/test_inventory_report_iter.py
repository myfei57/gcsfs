"""Tests for the streaming ``InventoryReport.iter_snapshot`` entry point.

The tests use in-memory fakes instead of a live GCS service:

* a fake filesystem routes report-config / listing / media / HEAD calls and
  records every request, so the "no live listing" guarantees are observable;
* a pluggable byte-chunk seam replaces the network read and lets tests split
  payloads at arbitrary byte boundaries (including inside multi-byte UTF-8
  characters) and observe response lifecycle on early exit;
* the stdlib ``csv`` module is used as an independent parsing baseline.
"""

import asyncio
import csv
import gzip
import io
import json
import tracemalloc
from datetime import date, datetime, timedelta

import pytest

from gcsfs.core import GCSFileSystem
from gcsfs.inventory_report import (
    DEFAULT_RECORD_QUEUE_SIZE,
    InventoryReport,
    InventoryReportError,
    InventoryReportIntegrityError,
    InventoryReportLimitError,
    InventoryReportParseError,
    InventorySnapshot,
    PinnedShard,
    _GzipChunker,
    _StreamingCsvParser,
)

SRC = "source-bucket"
DEST = "destination-bucket"
DEST_PATH = "inventory/reports"
REPORT_ID = "cfg-uuid"
LOCATION = "us-central1"
TARGET = date(2024, 1, 1)
SNAPSHOT_TIME = "2024-01-01T00:07:27Z"

DEFAULT_FIELDS = ["name", "size", "updated", "customField"]
DEFAULT_CSV = {
    "recordSeparator": "\n",
    "delimiter": ",",
    "headerRequired": True,
}


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def shard_base_name(index, target=TARGET, compressed=False, report_id=REPORT_ID):
    stamp = f"{target.isoformat()}T00:00"
    suffix = ".csv.gz" if compressed else ".csv"
    return f"{report_id}_{stamp}_{index}{suffix}"


def manifest_base_name(target=TARGET, report_id=REPORT_ID):
    return f"{report_id}_{target.isoformat()}T00:00_manifest.json"


def full_name(base_name):
    return f"{DEST_PATH}/{base_name}"


def raw_report_config(fields, csv_options=DEFAULT_CSV, *, source=SRC, dest=DEST):
    today = datetime.today().date()
    return {
        "frequencyOptions": {
            "startDate": {
                "day": 1,
                "month": 1,
                "year": today.year - 1,
            },
            "endDate": {
                "day": 31,
                "month": 12,
                "year": today.year + 1,
            },
        },
        "objectMetadataReportOptions": {
            "metadataFields": list(fields),
            "storageFilters": {"bucket": source},
            "storageDestinationOptions": {
                "bucket": dest,
                "destinationPath": DEST_PATH,
            },
        },
        "csvOptions": dict(csv_options),
    }


def manifest_doc(
    shard_names,
    fields,
    csv_options=DEFAULT_CSV,
    *,
    source=SRC,
    dest=DEST,
    target=TARGET,
    records_processed=None,
    report_id=REPORT_ID,
    overrides=None,
):
    doc = {
        "report_config": {
            "name": (f"projects/p/locations/{LOCATION}/reportConfigs/{report_id}"),
            "csvOptions": dict(csv_options),
            "objectMetadataReportOptions": {
                "metadataFields": list(fields),
                "storageFilters": {"bucket": source},
                "storageDestinationOptions": {
                    "bucket": dest,
                    "destinationPath": DEST_PATH,
                },
            },
        },
        "records_processed": records_processed,
        "snapshot_time": SNAPSHOT_TIME,
        "target_datetime": {
            "year": target.year,
            "month": target.month,
            "day": target.day,
        },
        "shard_count": len(shard_names),
        "report_shards_file_names": list(shard_names),
    }
    if overrides:
        doc.update(overrides)
    return doc


def object_metadata(name, payload, created_on, *, generation, content_encoding=None):
    item = {
        "name": name,
        "generation": str(generation),
        "size": str(len(payload)),
        "timeCreated": f"{created_on.isoformat()}T00:00:00.000Z",
    }
    if content_encoding:
        item["contentEncoding"] = content_encoding
    return item


def build_listing(
    shard_payloads,
    shard_names,
    manifest_payload,
    *,
    target=TARGET,
    generations=None,
    shard_dates=None,
    content_encodings=None,
    manifest_name=None,
    manifest_date=None,
):
    """Return the object listing under the destination path."""
    items = []
    generations = generations or [1000 + i for i in range(len(shard_payloads))]
    shard_dates = shard_dates or [target] * len(shard_payloads)
    content_encodings = content_encodings or [None] * len(shard_payloads)
    for index, (name, payload) in enumerate(zip(shard_names, shard_payloads)):
        items.append(
            object_metadata(
                full_name(name),
                payload,
                shard_dates[index],
                generation=generations[index],
                content_encoding=content_encodings[index],
            )
        )
    items.append(
        object_metadata(
            full_name(manifest_name or manifest_base_name(target)),
            manifest_payload,
            manifest_date or target,
            generation=9000,
        )
    )
    return items


def csv_payload(rows, fields, csv_options=DEFAULT_CSV, *, header=None):
    delim = csv_options["delimiter"]
    sep = csv_options["recordSeparator"]
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter=delim, lineterminator=sep)
    if header is not False and csv_options.get("headerRequired", False):
        writer.writerow(fields)
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def baseline_objects(payloads, fields, csv_options):
    """Independent stdlib-csv baseline over the raw shard payloads."""
    delim = csv_options["delimiter"]
    rows = []
    for payload in payloads:
        reader = csv.reader(
            io.StringIO(payload.decode("utf-8"), newline=""),
            delimiter=delim,
            quotechar='"',
        )
        shard_rows = list(reader)
        if csv_options.get("headerRequired", False):
            shard_rows = shard_rows[1:]
        rows.extend(shard_rows)
    return [dict(zip(fields, row)) for row in rows]


async def drain(snapshot):
    objects = []
    async with snapshot:
        async for obj in snapshot:
            objects.append(obj)
    return objects


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class StreamTracker:
    """Records open/close lifecycle of shard streams."""

    def __init__(self):
        self.open_shards = []
        self.closed_shards = []
        self.max_concurrent = 0
        self._concurrent = 0
        self.requested_generations = []

    def open(self, shard):
        self.open_shards.append(shard.name)
        self.requested_generations.append(shard.generation)
        self._concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self._concurrent)

    def close(self, shard):
        self.closed_shards.append(shard.name)
        self._concurrent -= 1


class ChunkSeam:
    """Stand-in for ``InventoryReport._iter_media_chunks``.

    Serves pinned bytes (captured at construction time) in fixed-size chunks
    and records lifecycle through a :class:`StreamTracker`.
    """

    def __init__(self, pinned_store, tracker, pinned_generations):
        self._store = dict(pinned_store)
        self._tracker = tracker
        self._pinned_generations = pinned_generations

    async def __call__(self, gcs_file_system, bucket, shard, chunk_size):
        assert (
            shard.generation == self._pinned_generations[shard.name]
        ), "media read must use the pinned shard generation"
        self._tracker.open(shard)
        try:
            data = self._store[(bucket, shard.name)]
            for offset in range(0, len(data), chunk_size):
                yield data[offset : offset + chunk_size]
        finally:
            self._tracker.close(shard)


class FakeFileSystem:
    """Minimal async GCS surface used by the inventory report code."""

    def __init__(
        self,
        raw_config,
        listing,
        store,
        *,
        head_store=None,
        head_delay=0.0,
    ):
        self.project = "project"
        self.requests_timeout = 5
        self.dircache = {}
        self._raw_config = raw_config
        self._listing = listing
        self._store = store
        self._head_store = head_store or {}
        self._head_delay = head_delay
        self.calls = []
        self.head_calls = []
        self.head_in_flight = 0
        self.head_max_in_flight = 0

    async def _set_session(self):
        return None

    def _format_path(self, path, args):
        return path.format(*args)

    def _get_params(self, kwargs):
        return {key: value for key, value in kwargs.items() if value is not None}

    def _get_headers(self, headers, **kwargs):
        return dict(headers or {})

    async def _call(self, method, path, *args, json_out=False, **kwargs):
        self.calls.append((method, path, args, kwargs))
        if "reportConfigs" in path:
            return self._raw_config
        if path == "b/{}/o":
            return {"items": list(self._listing)}
        if path == "b/{}/o/{}":
            bucket, name = args
            if kwargs.get("alt") == "media":
                return {}, self._store[(bucket, name)]
        raise AssertionError(f"unexpected call: {method} {path} {args} {kwargs}")

    async def _get_object(self, path):
        self.head_calls.append(path)
        self.head_in_flight += 1
        self.head_max_in_flight = max(self.head_max_in_flight, self.head_in_flight)
        try:
            if self._head_delay:
                await asyncio.sleep(self._head_delay)
            key = path.split("/", 1)[1]
            info = dict(self._head_store.get(key, {"size": 999}))
            info.setdefault("name", path)
            info.setdefault("type", "file")
            return info
        finally:
            self.head_in_flight -= 1

    # Reuse the real gcsfs object-resource conversions.
    def _process_object(self, bucket, object_metadata):
        return GCSFileSystem._process_object(self, bucket, object_metadata)

    def _parse_timestamp(self, timestamp):
        return GCSFileSystem._parse_timestamp(self, timestamp)


def make_snapshot_inputs(
    rows_by_shard,
    fields,
    csv_options=DEFAULT_CSV,
    *,
    compressed=False,
    target=TARGET,
    **listing_kwargs,
):
    """Build (fake fs args, shard names, raw payloads) for a report batch."""
    shard_names = [
        shard_base_name(i, target, compressed=compressed)
        for i in range(len(rows_by_shard))
    ]
    raw_payloads = [csv_payload(rows, fields, csv_options) for rows in rows_by_shard]
    if compressed:
        stored_payloads = [gzip.compress(payload, mtime=0) for payload in raw_payloads]
        encodings = ["gzip"] * len(raw_payloads)
    else:
        stored_payloads = raw_payloads
        encodings = [None] * len(raw_payloads)

    total_records = sum(len(rows) for rows in rows_by_shard)
    manifest = manifest_doc(
        shard_names,
        fields,
        csv_options,
        target=target,
        records_processed=total_records,
    )
    manifest_payload = json.dumps(manifest).encode("utf-8")
    listing = build_listing(
        stored_payloads,
        shard_names,
        manifest_payload,
        target=target,
        content_encodings=encodings,
        **listing_kwargs,
    )
    store = {
        (DEST, full_name(name)): payload
        for name, payload in zip(shard_names, stored_payloads)
    }
    store[(DEST, full_name(manifest_base_name(target)))] = manifest_payload
    config = raw_report_config(fields, csv_options)
    return {
        "config": config,
        "listing": listing,
        "store": store,
        "shard_names": shard_names,
        "raw_payloads": raw_payloads,
        "manifest": manifest,
        "total_records": total_records,
    }


def install_seam(monkeypatch, fs, tracker, pinned_generations):
    seam = ChunkSeam(fs._store, tracker, pinned_generations)
    monkeypatch.setattr(InventoryReport, "_iter_media_chunks", seam)
    return seam


def report_info(**overrides):
    info = {
        "location": LOCATION,
        "id": REPORT_ID,
        "use_snapshot_listing": True,
    }
    info.update(overrides)
    return info


# ---------------------------------------------------------------------------
# Parser unit tests
# ---------------------------------------------------------------------------


class TestStreamingCsvParser:
    def _parse(self, payload, csv_options, chunk_size, fields=DEFAULT_FIELDS):
        parser = _StreamingCsvParser(
            delimiter=csv_options["delimiter"],
            record_separator=csv_options["recordSeparator"],
            header_required=csv_options.get("headerRequired", False),
            expected_header=fields,
            max_record_bytes=None,
            shard_name="shard.csv",
        )
        events = []
        for offset in range(0, len(payload), chunk_size):
            events.extend(parser.feed(payload[offset : offset + chunk_size]))
        events.extend(parser.finish())
        return [event[0] for event in events]

    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 64, 4096])
    def test_quoted_delimiters_newlines_and_escaped_quotes(self, chunk_size):
        rows = [
            ["plain", "10", "2024-01-01T00:00:00Z", "x"],
            [
                "has,comma",
                "20",
                "2024-01-01T00:00:00Z",
                "line1\nline2",
            ],
            [
                'quote"inside',
                "30",
                "2024-01-01T00:00:00Z",
                "cr\r\nnl",
            ],
            ["", "40", "2024-01-01T00:00:00Z", "trailing,"],
        ]
        payload = csv_payload(rows, DEFAULT_FIELDS)
        parsed = self._parse(payload, DEFAULT_CSV, chunk_size)
        expected = list(csv.reader(io.StringIO(payload.decode(), newline="")))[1:]
        assert parsed == expected

    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 13])
    def test_crlf_separator_including_split_across_chunks(self, chunk_size):
        options = {
            "recordSeparator": "\r\n",
            "delimiter": ",",
            "headerRequired": False,
        }
        rows = [["a", "b"], ["c", "d"]]
        payload = csv_payload(rows, ["x", "y"], options, header=False)
        assert b"\r\n" in payload
        parsed = self._parse(payload, options, chunk_size, fields=["x", "y"])
        assert parsed == rows

    def test_crlf_inside_quotes_is_literal_content(self):
        options = {
            "recordSeparator": "\r\n",
            "delimiter": ",",
            "headerRequired": False,
        }
        payload = b'"line1\r\nline2",x\r\n'
        parsed = self._parse(payload, options, 1, fields=["a", "b"])
        assert parsed == [["line1\r\nline2", "x"]]

    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 4])
    def test_unicode_character_split_across_chunks(self, chunk_size):
        rows = [
            ["目录/文件-A", "10", "2024-01-01T00:00:00Z", "emoji-😀"],
            ["ar/مستند", "20", "2024-01-01T00:00:00Z", "日本語"],
        ]
        payload = csv_payload(rows, DEFAULT_FIELDS)
        parsed = self._parse(payload, DEFAULT_CSV, chunk_size)
        expected = list(csv.reader(io.StringIO(payload.decode(), newline="")))[1:]
        assert parsed == expected

    def test_unicode_split_at_every_byte_boundary(self):
        text = "中文-😀-Ελληνικά"
        rows = [[text, "1", "2024-01-01T00:00:00Z", "μ"]]
        payload = csv_payload(rows, DEFAULT_FIELDS)
        for split in range(1, len(payload)):
            parsed = self._parse(payload, DEFAULT_CSV, split)
            assert parsed == [[text, "1", "2024-01-01T00:00:00Z", "μ"]], split

    def test_header_mismatch_raises_parse_error(self):
        payload = b"wrong,header,fields,again\n"
        parser = _StreamingCsvParser(
            delimiter=",",
            record_separator="\n",
            header_required=True,
            expected_header=DEFAULT_FIELDS,
            max_record_bytes=None,
            shard_name="shard.csv",
        )
        with pytest.raises(InventoryReportParseError) as exc_info:
            parser.feed(payload)
            parser.finish()
        assert exc_info.value.shard == "shard.csv"

    def test_unterminated_quote_reports_location(self):
        parser = _StreamingCsvParser(
            delimiter=",",
            record_separator="\n",
            header_required=False,
            expected_header=None,
            max_record_bytes=None,
            shard_name="shard.csv",
        )
        parser.feed(b'"unterminated,rest\n')
        with pytest.raises(InventoryReportParseError, match="Unterminated quote"):
            parser.finish()

    def test_invalid_utf8_record_reports_location(self):
        parser = _StreamingCsvParser(
            delimiter=",",
            record_separator="\n",
            header_required=False,
            expected_header=None,
            max_record_bytes=None,
            shard_name="shard.csv",
        )
        with pytest.raises(InventoryReportParseError, match="UTF-8") as exc_info:
            parser.feed(b"\xff\xfe,ok\n")
            parser.finish()
        assert exc_info.value.record_index == 0
        assert exc_info.value.shard == "shard.csv"

    def test_record_byte_limit(self):
        parser = _StreamingCsvParser(
            delimiter=",",
            record_separator="\n",
            header_required=False,
            expected_header=None,
            max_record_bytes=8,
            shard_name="shard.csv",
        )
        with pytest.raises(InventoryReportLimitError) as exc_info:
            parser.feed(b"a" * 20 + b"\n")
        assert exc_info.value.shard == "shard.csv"
        assert exc_info.value.limit == 8

    def test_blank_lines_and_trailing_separator_emit_no_empty_records(self):
        parser = _StreamingCsvParser(
            delimiter=",",
            record_separator="\n",
            header_required=False,
            expected_header=None,
            max_record_bytes=None,
            shard_name="shard.csv",
        )
        events = parser.feed(b"a,b\n\nc,d\n")
        events.extend(parser.finish())
        assert [event[0] for event in events] == [["a", "b"], ["c", "d"]]

    def test_non_ascii_delimiter_rejected(self):
        with pytest.raises(ValueError, match="single ASCII"):
            _StreamingCsvParser(
                delimiter="中",
                record_separator="\n",
                header_required=False,
                expected_header=None,
                max_record_bytes=None,
                shard_name="shard.csv",
            )


class TestGzipChunker:
    def test_incremental_decompression_matches_gzip_module(self):
        payload = ("line,value\n" * 1000).encode()
        compressed = gzip.compress(payload, mtime=0)
        chunker = _GzipChunker(None, "shard.csv.gz")
        out = b""
        for split in range(0, len(compressed), 7):
            out += chunker.feed(compressed[split : split + 7])
        out += chunker.finish()
        assert out == payload

    def test_truncated_gzip_raises(self):
        compressed = gzip.compress(b"a,b\nc,d\n", mtime=0)
        chunker = _GzipChunker(None, "shard.csv.gz")
        chunker.feed(compressed[:-8])
        with pytest.raises(InventoryReportError, match="Truncated gzip"):
            chunker.finish()

    def test_garbage_gzip_raises_with_shard_name(self):
        chunker = _GzipChunker(None, "shard.csv.gz")
        with pytest.raises(InventoryReportError, match="shard.csv.gz"):
            chunker.feed(b"definitely not gzip")

    def test_decompression_budget_enforced(self):
        compressed = gzip.compress(b"x" * 1024, mtime=0)
        chunker = _GzipChunker(16, "shard.csv.gz")
        with pytest.raises(InventoryReportLimitError) as exc_info:
            chunker.feed(compressed)
        assert exc_info.value.limit == 16


# ---------------------------------------------------------------------------
# End-to-end snapshot tests
# ---------------------------------------------------------------------------


class TestIterSnapshot:
    @pytest.mark.asyncio
    async def test_multi_shard_snapshot_matches_csv_baseline(self, monkeypatch):
        rows = [
            [
                f"dir/object-{i}",
                str(100 + i),
                "2024-01-01T00:00:00Z",
                f"tag-{i % 3}",
            ]
            for i in range(25)
        ]
        split = [rows[:10], rows[10:20], rows[20:]]
        inputs = make_snapshot_inputs(split, DEFAULT_FIELDS)
        fs = FakeFileSystem(inputs["config"], inputs["listing"], inputs["store"])
        tracker = StreamTracker()
        generations = {
            full_name(name): str(1000 + i)
            for i, name in enumerate(inputs["shard_names"])
        }
        install_seam(monkeypatch, fs, tracker, generations)

        snapshot = await InventoryReport.iter_snapshot(fs, report_info(), chunk_size=11)
        assert isinstance(snapshot, InventorySnapshot)
        assert snapshot.snapshot_time == SNAPSHOT_TIME
        assert snapshot.target_datetime == TARGET.isoformat()
        assert snapshot.records_processed == 25
        assert len(snapshot.shards) == 3
        assert all(isinstance(shard, PinnedShard) for shard in snapshot.shards)
        assert [shard.shard_index for shard in snapshot.shards] == [0, 1, 2]
        assert snapshot.schema == tuple(DEFAULT_FIELDS)

        objects = await drain(snapshot)
        assert snapshot.completed is True

        baseline = baseline_objects(inputs["raw_payloads"], DEFAULT_FIELDS, DEFAULT_CSV)
        assert len(objects) == len(baseline) == 25
        for obj, base in zip(objects, baseline):
            assert obj["name"] == f"{SRC}/{base['name']}"
            assert obj["size"] == int(base["size"])
            assert obj["type"] == "file"
            # Unknown/extensible schema fields are preserved verbatim.
            assert obj["customField"] == base["customField"]
            assert "mtime" in obj

    @pytest.mark.asyncio
    async def test_gzip_shards_are_decompressed_incrementally(self, monkeypatch):
        rows = [[f"key-{i}", str(i), "2024-01-01T00:00:00Z", "z"] for i in range(30)]
        inputs = make_snapshot_inputs(
            [rows[:15], rows[15:]], DEFAULT_FIELDS, compressed=True
        )
        fs = FakeFileSystem(inputs["config"], inputs["listing"], inputs["store"])
        tracker = StreamTracker()
        generations = {
            full_name(name): str(1000 + i)
            for i, name in enumerate(inputs["shard_names"])
        }
        install_seam(monkeypatch, fs, tracker, generations)

        snapshot = await InventoryReport.iter_snapshot(fs, report_info(), chunk_size=13)
        assert all(shard.compressed for shard in snapshot.shards)
        objects = await drain(snapshot)
        baseline = baseline_objects(inputs["raw_payloads"], DEFAULT_FIELDS, DEFAULT_CSV)
        assert [obj["name"] for obj in objects] == [
            f"{SRC}/{row['name']}" for row in baseline
        ]
        assert tracker.closed_shards == tracker.open_shards

    @pytest.mark.asyncio
    async def test_corrupt_gzip_fails_with_shard_context(self, monkeypatch):
        rows = [["a", "1", "2024-01-01T00:00:00Z", "x"]]
        inputs = make_snapshot_inputs([rows], DEFAULT_FIELDS, compressed=True)
        corrupt_name = full_name(inputs["shard_names"][0])
        inputs["store"][(DEST, corrupt_name)] = inputs["store"][(DEST, corrupt_name)][
            :-8
        ]
        fs = FakeFileSystem(inputs["config"], inputs["listing"], inputs["store"])
        tracker = StreamTracker()
        install_seam(
            monkeypatch,
            fs,
            tracker,
            {corrupt_name: "1000"},
        )
        snapshot = await InventoryReport.iter_snapshot(fs, report_info())
        with pytest.raises(InventoryReportError) as exc_info:
            await drain(snapshot)
        assert corrupt_name in str(exc_info.value)
        assert tracker.closed_shards == tracker.open_shards

    @pytest.mark.asyncio
    async def test_huge_record_is_rejected_with_position(self, monkeypatch):
        rows = [["a" * 200, "1", "2024-01-01T00:00:00Z", "x"]]
        inputs = make_snapshot_inputs([rows], DEFAULT_FIELDS)
        fs = FakeFileSystem(inputs["config"], inputs["listing"], inputs["store"])
        install_seam(
            monkeypatch,
            fs,
            StreamTracker(),
            {full_name(inputs["shard_names"][0]): "1000"},
        )
        snapshot = await InventoryReport.iter_snapshot(
            fs, report_info(), chunk_size=5, max_record_bytes=32
        )
        with pytest.raises(InventoryReportLimitError) as exc_info:
            await drain(snapshot)
        error = exc_info.value
        assert error.shard == full_name(inputs["shard_names"][0])
        assert error.record_index == 0
        assert error.limit == 32

    @pytest.mark.asyncio
    async def test_decompressed_byte_budget_enforced(self, monkeypatch):
        rows = [["a" * 100, "1", "2024-01-01T00:00:00Z", "x"]]
        inputs = make_snapshot_inputs([rows], DEFAULT_FIELDS, compressed=True)
        fs = FakeFileSystem(inputs["config"], inputs["listing"], inputs["store"])
        install_seam(
            monkeypatch,
            fs,
            StreamTracker(),
            {full_name(inputs["shard_names"][0]): "1000"},
        )
        snapshot = await InventoryReport.iter_snapshot(
            fs, report_info(), max_decompressed_bytes=16
        )
        with pytest.raises(InventoryReportLimitError):
            await drain(snapshot)

    @pytest.mark.asyncio
    async def test_prefix_filter_uses_key_boundaries(self, monkeypatch):
        keys = [
            "dir/object-1",
            "dir/object-2",
            "dir",
            "dirx/object-3",
            "other/object-4",
        ]
        rows = [[key, "10", "2024-01-01T00:00:00Z", "x"] for key in keys]
        inputs = make_snapshot_inputs([rows], DEFAULT_FIELDS)
        fs = FakeFileSystem(inputs["config"], inputs["listing"], inputs["store"])
        install_seam(
            monkeypatch,
            fs,
            StreamTracker(),
            {full_name(inputs["shard_names"][0]): "1000"},
        )
        snapshot = await InventoryReport.iter_snapshot(fs, report_info(), prefix="dir")
        objects = await drain(snapshot)
        names = {obj["name"] for obj in objects}
        assert names == {
            f"{SRC}/dir/object-1",
            f"{SRC}/dir/object-2",
            f"{SRC}/dir",
        }

    @pytest.mark.asyncio
    async def test_name_only_mode_skips_processing(self, monkeypatch):
        rows = [
            ["key-1", "10", "2024-01-01T00:00:00Z", "x"],
            ["key-2", "20", "2024-01-01T00:00:00Z", "y"],
        ]
        inputs = make_snapshot_inputs([rows], DEFAULT_FIELDS)
        fs = FakeFileSystem(inputs["config"], inputs["listing"], inputs["store"])
        install_seam(
            monkeypatch,
            fs,
            StreamTracker(),
            {full_name(inputs["shard_names"][0]): "1000"},
        )
        snapshot = await InventoryReport.iter_snapshot(
            fs, report_info(use_snapshot_listing=False)
        )
        objects = await drain(snapshot)
        assert objects == [
            {"name": f"{SRC}/key-1"},
            {"name": f"{SRC}/key-2"},
        ]
        assert fs.head_calls == []

    @pytest.mark.asyncio
    async def test_custom_delimiter_and_crlf_separator(self, monkeypatch):
        fields = ["name", "size"]
        options = {
            "recordSeparator": "\r\n",
            "delimiter": "|",
            "headerRequired": True,
        }
        rows = [["a|b", "10"], ["c", "20"]]
        inputs = make_snapshot_inputs([rows], fields, options)
        fs = FakeFileSystem(inputs["config"], inputs["listing"], inputs["store"])
        install_seam(
            monkeypatch,
            fs,
            StreamTracker(),
            {full_name(inputs["shard_names"][0]): "1000"},
        )
        snapshot = await InventoryReport.iter_snapshot(fs, report_info(), chunk_size=3)
        objects = await drain(snapshot)
        baseline = baseline_objects(inputs["raw_payloads"], fields, options)
        assert [obj["name"] for obj in objects] == [
            f"{SRC}/{row['name']}" for row in baseline
        ]
        assert objects[0]["size"] == 10

    @pytest.mark.asyncio
    async def test_generation_and_size_follow_schema_types(self, monkeypatch):
        fields = ["name", "size", "generation"]
        rows = [["obj-1", "10", "7777"], ["obj-2", "20", "8888"]]
        inputs = make_snapshot_inputs([rows], fields)
        fs = FakeFileSystem(inputs["config"], inputs["listing"], inputs["store"])
        install_seam(
            monkeypatch,
            fs,
            StreamTracker(),
            {full_name(inputs["shard_names"][0]): "1000"},
        )
        snapshot = await InventoryReport.iter_snapshot(
            fs,
            {
                "location": LOCATION,
                "id": REPORT_ID,
                "use_snapshot_listing": True,
            },
        )
        objects = await drain(snapshot)
        assert objects[0]["size"] == 10
        assert isinstance(objects[0]["size"], int)
        # GCS generations are identifiers and stay strings.
        assert objects[0]["generation"] == "7777"
        assert objects[1]["generation"] == "8888"


class TestIntegrity:
    @pytest.mark.asyncio
    async def test_missing_manifest_refuses_to_guess(self):
        inputs = make_snapshot_inputs(
            [[["a", "1", "2024-01-01T00:00:00Z", "x"]]], DEFAULT_FIELDS
        )
        listing = [
            item
            for item in inputs["listing"]
            if not item["name"].endswith("_manifest.json")
        ]
        fs = FakeFileSystem(inputs["config"], listing, inputs["store"])
        with pytest.raises(InventoryReportIntegrityError, match="manifest"):
            await InventoryReport.iter_snapshot(fs, report_info())

    @pytest.mark.asyncio
    async def test_missing_shard_fails_before_any_record(self, monkeypatch):
        inputs = make_snapshot_inputs(
            [
                [["a", "1", "2024-01-01T00:00:00Z", "x"]],
                [["b", "2", "2024-01-01T00:00:00Z", "y"]],
            ],
            DEFAULT_FIELDS,
        )
        missing = full_name(inputs["shard_names"][1])
        listing = [item for item in inputs["listing"] if item["name"] != missing]
        fs = FakeFileSystem(inputs["config"], listing, inputs["store"])
        install_seam(
            monkeypatch,
            fs,
            StreamTracker(),
            {full_name(inputs["shard_names"][0]): "1000"},
        )
        with pytest.raises(InventoryReportIntegrityError, match="missing"):
            await InventoryReport.iter_snapshot(fs, report_info())

    @pytest.mark.asyncio
    async def test_shard_count_mismatch_fails(self):
        inputs = make_snapshot_inputs(
            [[["a", "1", "2024-01-01T00:00:00Z", "x"]]], DEFAULT_FIELDS
        )
        inputs["manifest"]["shard_count"] = 5
        manifest_payload = json.dumps(inputs["manifest"]).encode()
        listing = build_listing(
            inputs["raw_payloads"],
            inputs["shard_names"],
            manifest_payload,
        )
        store = dict(inputs["store"])
        manifest_name = full_name(manifest_base_name())
        store[(DEST, manifest_name)] = manifest_payload
        fs = FakeFileSystem(inputs["config"], listing, store)
        with pytest.raises(InventoryReportIntegrityError, match="shard_count"):
            await InventoryReport.iter_snapshot(fs, report_info())

    @pytest.mark.asyncio
    async def test_shard_from_different_report_time_fails(self):
        other_day = TARGET + timedelta(days=1)
        inputs = make_snapshot_inputs(
            [
                [["a", "1", "2024-01-01T00:00:00Z", "x"]],
                [["b", "2", "2024-01-01T00:00:00Z", "y"]],
            ],
            DEFAULT_FIELDS,
            shard_dates=[TARGET, other_day],
        )
        fs = FakeFileSystem(inputs["config"], inputs["listing"], inputs["store"])
        with pytest.raises(InventoryReportIntegrityError, match="report times"):
            await InventoryReport.iter_snapshot(fs, report_info())

    @pytest.mark.asyncio
    async def test_schema_drift_between_manifest_and_config_fails(self):
        inputs = make_snapshot_inputs(
            [[["a", "1", "2024-01-01T00:00:00Z", "x"]]], DEFAULT_FIELDS
        )
        drifted_fields = ["name", "size", "updated"]
        inputs["manifest"]["report_config"]["objectMetadataReportOptions"][
            "metadataFields"
        ] = drifted_fields
        manifest_payload = json.dumps(inputs["manifest"]).encode()
        listing = build_listing(
            inputs["raw_payloads"],
            inputs["shard_names"],
            manifest_payload,
        )
        store = dict(inputs["store"])
        store[(DEST, full_name(manifest_base_name()))] = manifest_payload
        fs = FakeFileSystem(inputs["config"], listing, store)
        with pytest.raises(InventoryReportIntegrityError, match="schema drift"):
            await InventoryReport.iter_snapshot(fs, report_info())

    @pytest.mark.asyncio
    async def test_csv_option_drift_fails(self):
        inputs = make_snapshot_inputs(
            [[["a", "1", "2024-01-01T00:00:00Z", "x"]]], DEFAULT_FIELDS
        )
        inputs["manifest"]["report_config"]["csvOptions"]["delimiter"] = "|"
        manifest_payload = json.dumps(inputs["manifest"]).encode()
        listing = build_listing(
            inputs["raw_payloads"],
            inputs["shard_names"],
            manifest_payload,
        )
        store = dict(inputs["store"])
        store[(DEST, full_name(manifest_base_name()))] = manifest_payload
        fs = FakeFileSystem(inputs["config"], listing, store)
        with pytest.raises(InventoryReportIntegrityError, match="delimiter"):
            await InventoryReport.iter_snapshot(fs, report_info())

    @pytest.mark.asyncio
    async def test_wrong_report_config_id_fails(self):
        inputs = make_snapshot_inputs(
            [[["a", "1", "2024-01-01T00:00:00Z", "x"]]], DEFAULT_FIELDS
        )
        inputs["manifest"]["report_config"][
            "name"
        ] = "projects/p/locations/l/reportConfigs/some-other-id"
        manifest_payload = json.dumps(inputs["manifest"]).encode()
        listing = build_listing(
            inputs["raw_payloads"],
            inputs["shard_names"],
            manifest_payload,
        )
        store = dict(inputs["store"])
        store[(DEST, full_name(manifest_base_name()))] = manifest_payload
        fs = FakeFileSystem(inputs["config"], listing, store)
        with pytest.raises(InventoryReportIntegrityError, match="report config id"):
            await InventoryReport.iter_snapshot(fs, report_info())

    @pytest.mark.asyncio
    async def test_records_processed_mismatch_fails_after_parsing(self, monkeypatch):
        inputs = make_snapshot_inputs(
            [[["a", "1", "2024-01-01T00:00:00Z", "x"]]], DEFAULT_FIELDS
        )
        inputs["manifest"]["records_processed"] = 99
        manifest_payload = json.dumps(inputs["manifest"]).encode()
        listing = build_listing(
            inputs["raw_payloads"],
            inputs["shard_names"],
            manifest_payload,
        )
        store = dict(inputs["store"])
        store[(DEST, full_name(manifest_base_name()))] = manifest_payload
        fs = FakeFileSystem(inputs["config"], listing, store)
        install_seam(
            monkeypatch,
            fs,
            StreamTracker(),
            {full_name(inputs["shard_names"][0]): "1000"},
        )
        snapshot = await InventoryReport.iter_snapshot(fs, report_info())
        with pytest.raises(InventoryReportIntegrityError, match="manifest"):
            await drain(snapshot)
        assert snapshot.completed is False
        assert fs.dircache == {}

    @pytest.mark.asyncio
    async def test_report_datetime_selects_the_matching_batch(self):
        older = make_snapshot_inputs(
            [[["old", "1", "2024-01-01T00:00:00Z", "x"]]],
            DEFAULT_FIELDS,
            target=date(2024, 1, 1),
        )
        newer = make_snapshot_inputs(
            [[["new", "2", "2024-01-02T00:00:00Z", "y"]]],
            DEFAULT_FIELDS,
            target=date(2024, 1, 2),
        )
        listing = older["listing"] + newer["listing"]
        store = {**older["store"], **newer["store"]}
        fs = FakeFileSystem(newer["config"], listing, store)

        # Default: newest manifest wins.
        newest = await InventoryReport.iter_snapshot(fs, report_info())
        assert newest.target_datetime == date(2024, 1, 2).isoformat()

        selected = await InventoryReport.iter_snapshot(
            fs, report_info(), report_datetime="2024-01-01"
        )
        assert selected.target_datetime == date(2024, 1, 1).isoformat()
        with pytest.raises(InventoryReportIntegrityError, match="target date"):
            await InventoryReport.iter_snapshot(
                fs, report_info(), report_datetime="2025-12-31"
            )

    @pytest.mark.asyncio
    async def test_field_count_mismatch_carries_position(self, monkeypatch):
        # Header declares four fields but the data row only has three.
        payload = b"name,size,updated,customField\n" b"only,three,fields\n"
        shard_name = shard_base_name(0)
        manifest = manifest_doc([shard_name], DEFAULT_FIELDS, records_processed=1)
        manifest_payload = json.dumps(manifest).encode()
        listing = build_listing([payload], [shard_name], manifest_payload)
        store = {
            (DEST, full_name(shard_name)): payload,
            (DEST, full_name(manifest_base_name())): manifest_payload,
        }
        fs = FakeFileSystem(raw_report_config(DEFAULT_FIELDS), listing, store)
        install_seam(
            monkeypatch,
            fs,
            StreamTracker(),
            {full_name(shard_name): "1000"},
        )
        snapshot = await InventoryReport.iter_snapshot(fs, report_info())
        with pytest.raises(InventoryReportParseError) as exc_info:
            await drain(snapshot)
        assert exc_info.value.shard == full_name(shard_name)
        assert exc_info.value.record_index == 0


class TestHeadFallback:
    @pytest.mark.asyncio
    async def test_head_only_for_records_missing_values(self, monkeypatch):
        fields = ["name", "size", "updated"]
        rows = [
            ["keep-1", "10", "2024-01-01T00:00:00Z"],
            ["missing-size", "", "2024-01-01T00:00:00Z"],
            ["keep-2", "30", "2024-01-01T00:00:00Z"],
        ]
        inputs = make_snapshot_inputs([rows], fields)
        fs = FakeFileSystem(
            inputs["config"],
            inputs["listing"],
            inputs["store"],
            head_store={"missing-size": {"size": 777}},
            head_delay=0.01,
        )
        install_seam(
            monkeypatch,
            fs,
            StreamTracker(),
            {full_name(inputs["shard_names"][0]): "1000"},
        )
        snapshot = await InventoryReport.iter_snapshot(
            fs,
            report_info(),
            head_missing_metadata=True,
            head_concurrency=2,
        )
        objects = await drain(snapshot)
        assert fs.head_calls == [f"{SRC}/missing-size"]
        by_name = {obj["name"]: obj for obj in objects}
        assert by_name[f"{SRC}/missing-size"]["size"] == 777
        assert by_name[f"{SRC}/keep-1"]["size"] == 10
        assert snapshot._filler.max_in_flight <= 2

    @pytest.mark.asyncio
    async def test_missing_size_without_head_policy_fails_with_location(
        self, monkeypatch
    ):
        fields = ["name", "size", "updated"]
        rows = [["oops", "", "2024-01-01T00:00:00Z"]]
        inputs = make_snapshot_inputs([rows], fields)
        fs = FakeFileSystem(inputs["config"], inputs["listing"], inputs["store"])
        install_seam(
            monkeypatch,
            fs,
            StreamTracker(),
            {full_name(inputs["shard_names"][0]): "1000"},
        )
        snapshot = await InventoryReport.iter_snapshot(fs, report_info())
        with pytest.raises(InventoryReportParseError) as exc_info:
            await drain(snapshot)
        assert exc_info.value.record_index == 0
        assert fs.head_calls == []

    @pytest.mark.asyncio
    async def test_schema_without_size_uses_bounded_head_fallback(self, monkeypatch):
        fields = ["name", "updated"]
        rows = [[f"obj-{i}", "2024-01-01T00:00:00Z"] for i in range(8)]
        inputs = make_snapshot_inputs([rows], fields)
        head_store = {f"obj-{i}": {"size": 1000 + i} for i in range(8)}
        fs = FakeFileSystem(
            inputs["config"],
            inputs["listing"],
            inputs["store"],
            head_store=head_store,
            head_delay=0.01,
        )
        install_seam(
            monkeypatch,
            fs,
            StreamTracker(),
            {full_name(inputs["shard_names"][0]): "1000"},
        )
        snapshot = await InventoryReport.iter_snapshot(
            fs,
            report_info(),
            head_missing_metadata=True,
            head_concurrency=2,
        )
        assert snapshot._head_fields == ("size",)
        objects = await drain(snapshot)
        assert len(fs.head_calls) == 8
        assert snapshot._filler.max_in_flight <= 2
        assert {obj["size"] for obj in objects} == {1000 + i for i in range(8)}

    @pytest.mark.asyncio
    async def test_schema_without_size_and_head_disabled_fails_at_setup(self):
        fields = ["name", "updated"]
        inputs = make_snapshot_inputs([[["o", "2024-01-01T00:00:00Z"]]], fields)
        fs = FakeFileSystem(inputs["config"], inputs["listing"], inputs["store"])
        with pytest.raises(InventoryReportIntegrityError, match="size"):
            await InventoryReport.iter_snapshot(fs, report_info())


class TestCancellationAndResources:
    @pytest.mark.asyncio
    async def test_early_exit_closes_stream_and_cancels_producer(self, monkeypatch):
        rows = [[f"key-{i}", str(i), "2024-01-01T00:00:00Z", "x"] for i in range(100)]
        inputs = make_snapshot_inputs([rows[:40], rows[40:]], DEFAULT_FIELDS)
        fs = FakeFileSystem(inputs["config"], inputs["listing"], inputs["store"])
        tracker = StreamTracker()
        generations = {
            full_name(name): str(1000 + i)
            for i, name in enumerate(inputs["shard_names"])
        }
        install_seam(monkeypatch, fs, tracker, generations)

        snapshot = await InventoryReport.iter_snapshot(
            fs,
            report_info(),
            chunk_size=4,
            record_queue_size=5,
        )
        seen = 0
        async with snapshot:
            async for _ in snapshot:
                seen += 1
                if seen == 3:
                    break
        assert snapshot.completed is False
        assert snapshot._closed is True
        assert tracker.closed_shards == tracker.open_shards
        # Only one shard response is ever open at a time.
        assert tracker.max_concurrent == 1
        assert snapshot._producer.done()
        assert snapshot._queue.empty()

    @pytest.mark.asyncio
    async def test_consumer_task_cancellation_unwinds_stream(self, monkeypatch):
        rows = [[f"key-{i}", str(i), "2024-01-01T00:00:00Z", "x"] for i in range(100)]
        inputs = make_snapshot_inputs([rows], DEFAULT_FIELDS)
        fs = FakeFileSystem(inputs["config"], inputs["listing"], inputs["store"])
        tracker = StreamTracker()
        install_seam(
            monkeypatch,
            fs,
            tracker,
            {full_name(inputs["shard_names"][0]): "1000"},
        )
        snapshot = await InventoryReport.iter_snapshot(
            fs, report_info(), record_queue_size=2
        )

        async def consume():
            async for _ in snapshot:
                await asyncio.sleep(0)

        task = asyncio.ensure_future(consume())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await snapshot.aclose()
        assert tracker.closed_shards == tracker.open_shards
        assert snapshot.completed is False

    @pytest.mark.asyncio
    async def test_abandoned_snapshot_releases_everything_on_aclose(self, monkeypatch):
        rows = [[f"key-{i}", str(i), "2024-01-01T00:00:00Z", "x"] for i in range(100)]
        inputs = make_snapshot_inputs([rows], DEFAULT_FIELDS)
        fs = FakeFileSystem(inputs["config"], inputs["listing"], inputs["store"])
        tracker = StreamTracker()
        install_seam(
            monkeypatch,
            fs,
            tracker,
            {full_name(inputs["shard_names"][0]): "1000"},
        )
        # A consumer that exits without ``async with`` still has a deterministic
        # recovery path via ``aclose``.
        snapshot = await InventoryReport.iter_snapshot(
            fs, report_info(), record_queue_size=1
        )
        count = 0
        async for _ in snapshot:
            count += 1
            if count == 2:
                break
        assert snapshot.completed is False
        await snapshot.aclose()
        assert snapshot._closed is True
        assert snapshot._producer.done()
        assert tracker.closed_shards == tracker.open_shards
        assert snapshot._queue.empty()

    @pytest.mark.asyncio
    async def test_record_queue_never_exceeds_configured_size(self, monkeypatch):
        rows = [[f"key-{i}", str(i), "2024-01-01T00:00:00Z", "x"] for i in range(60)]
        inputs = make_snapshot_inputs([rows], DEFAULT_FIELDS)
        fs = FakeFileSystem(inputs["config"], inputs["listing"], inputs["store"])
        install_seam(
            monkeypatch,
            fs,
            StreamTracker(),
            {full_name(inputs["shard_names"][0]): "1000"},
        )
        snapshot = await InventoryReport.iter_snapshot(
            fs, report_info(), record_queue_size=4
        )
        max_qsize = 0
        original_put = snapshot._queue.put

        async def tracking_put(item):
            result = await original_put(item)
            nonlocal max_qsize
            max_qsize = max(max_qsize, snapshot._queue.qsize())
            return result

        snapshot._queue.put = tracking_put
        count = 0
        async with snapshot:
            async for _ in snapshot:
                count += 1
                await asyncio.sleep(0)  # let the producer run ahead
        assert count == 60
        assert max_qsize <= 4


class TestDirectoryCache:
    @pytest.mark.asyncio
    async def test_dircache_untouched_by_default(self, monkeypatch):
        inputs = make_snapshot_inputs(
            [[["a", "1", "2024-01-01T00:00:00Z", "x"]]], DEFAULT_FIELDS
        )
        fs = FakeFileSystem(inputs["config"], inputs["listing"], inputs["store"])
        install_seam(
            monkeypatch,
            fs,
            StreamTracker(),
            {full_name(inputs["shard_names"][0]): "1000"},
        )
        snapshot = await InventoryReport.iter_snapshot(fs, report_info())
        await drain(snapshot)
        assert fs.dircache == {}

    @pytest.mark.asyncio
    async def test_publish_requires_full_bucket_snapshot(self):
        inputs = make_snapshot_inputs(
            [[["a", "1", "2024-01-01T00:00:00Z", "x"]]], DEFAULT_FIELDS
        )
        fs = FakeFileSystem(inputs["config"], inputs["listing"], inputs["store"])
        with pytest.raises(ValueError, match="directory cache"):
            await InventoryReport.iter_snapshot(
                fs, report_info(), prefix="a", publish_dircache=True
            )
        with pytest.raises(ValueError, match="directory cache"):
            await InventoryReport.iter_snapshot(
                fs,
                report_info(use_snapshot_listing=False),
                publish_dircache=True,
            )

    @pytest.mark.asyncio
    async def test_partial_iteration_never_publishes(self, monkeypatch):
        rows = [[f"key-{i}", str(i), "2024-01-01T00:00:00Z", "x"] for i in range(20)]
        inputs = make_snapshot_inputs([rows], DEFAULT_FIELDS)
        fs = FakeFileSystem(inputs["config"], inputs["listing"], inputs["store"])
        install_seam(
            monkeypatch,
            fs,
            StreamTracker(),
            {full_name(inputs["shard_names"][0]): "1000"},
        )
        snapshot = await InventoryReport.iter_snapshot(
            fs, report_info(), publish_dircache=True, record_queue_size=2
        )
        async with snapshot:
            async for _ in snapshot:
                break
        assert SRC not in fs.dircache

    @pytest.mark.asyncio
    async def test_full_snapshot_is_published_atomically(self, monkeypatch):
        rows = [[f"key-{i}", str(i), "2024-01-01T00:00:00Z", "x"] for i in range(10)]
        inputs = make_snapshot_inputs([rows], DEFAULT_FIELDS)
        fs = FakeFileSystem(inputs["config"], inputs["listing"], inputs["store"])
        install_seam(
            monkeypatch,
            fs,
            StreamTracker(),
            {full_name(inputs["shard_names"][0]): "1000"},
        )
        snapshot = await InventoryReport.iter_snapshot(
            fs, report_info(), publish_dircache=True
        )
        objects = await drain(snapshot)
        assert fs.dircache[SRC] == objects
        assert snapshot.completed is True


class TestNoLiveListing:
    @pytest.mark.asyncio
    async def test_pinned_content_survives_bucket_changes_and_no_live_calls(
        self, monkeypatch
    ):
        rows = [[f"key-{i}", str(i), "2024-01-01T00:00:00Z", "x"] for i in range(12)]
        inputs = make_snapshot_inputs([rows[:6], rows[6:]], DEFAULT_FIELDS)
        fs = FakeFileSystem(inputs["config"], inputs["listing"], inputs["store"])
        tracker = StreamTracker()
        generations = {
            full_name(name): str(1000 + i)
            for i, name in enumerate(inputs["shard_names"])
        }
        install_seam(monkeypatch, fs, tracker, generations)

        snapshot = await InventoryReport.iter_snapshot(fs, report_info())

        # Simulate the live bucket changing after the batch was pinned.
        for key in list(fs._store):
            fs._store[key] = b"completely different live content"
        fs._listing.append({"name": "brand-new-object", "generation": "1", "size": "1"})

        objects = await drain(snapshot)
        baseline = baseline_objects(inputs["raw_payloads"], DEFAULT_FIELDS, DEFAULT_CSV)
        assert [obj["name"] for obj in objects] == [
            f"{SRC}/{row['name']}" for row in baseline
        ]
        # Every media read used the pinned generations.
        assert set(tracker.requested_generations) == set(generations.values())

        # Only report-config, destination-bucket listing and media calls.
        list_calls = [call for call in fs.calls if call[1] == "b/{}/o"]
        media_calls = [call for call in fs.calls if call[1] == "b/{}/o/{}"]
        config_calls = [call for call in fs.calls if "reportConfigs" in call[1]]
        assert len(config_calls) == 1
        assert list_calls and media_calls
        assert all(call[2][0] == DEST for call in list_calls)
        assert all(call[3].get("prefix") for call in list_calls)
        assert all(call[2][0] == DEST for call in media_calls)
        assert all(call[3].get("alt") == "media" for call in media_calls)
        # No source-bucket object listing and no unconditional HEAD traffic.
        assert all(call[2][0] != SRC for call in fs.calls if call[2])
        assert fs.head_calls == []


class TestMediaStream:
    class _FakeContent:
        def __init__(self, data):
            self._data = data

        def iter_chunked(self, size):
            data = self._data

            async def generator():
                # Real aiohttp chunk reads always suspend on network I/O.
                for offset in range(0, len(data), size):
                    await asyncio.sleep(0)
                    yield data[offset : offset + size]

            return generator()

    class _FakeResponse:
        def __init__(self, data):
            self.status = 200
            self.content = TestMediaStream._FakeContent(data)
            self.closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            self.closed = True
            return False

    class _FakeSession:
        def __init__(self, data):
            self._data = data
            self.requests = []
            self.responses = []

        def request(self, method, url, params, headers, timeout):
            self.requests.append(
                {
                    "method": method,
                    "url": url,
                    "params": params,
                    "headers": headers,
                    "timeout": timeout,
                }
            )
            response = TestMediaStream._FakeResponse(self._data)
            self.responses.append(response)
            return response

    class _SessionFS(FakeFileSystem):
        def __init__(self, data):
            super().__init__({}, [], {})
            self.session = TestMediaStream._FakeSession(data)

    @pytest.mark.asyncio
    async def test_media_stream_pins_generation_and_closes_response(self):
        data = b"0123456789abcdef"
        fs = self._SessionFS(data)
        shard = PinnedShard(
            name=full_name("shard.csv"),
            generation="1337",
            size=len(data),
            compressed=False,
            shard_index=0,
        )
        chunks = []
        async for chunk in InventoryReport._iter_media_chunks(fs, DEST, shard, 5):
            chunks.append(chunk)
        assert b"".join(chunks) == data
        request = fs.session.requests[0]
        assert request["params"]["alt"] == "media"
        assert request["params"]["generation"] == "1337"
        assert shard.name in request["url"]
        assert fs.session.responses[0].closed is True

    @pytest.mark.asyncio
    async def test_media_stream_response_closed_on_cancellation(self):
        data = b"x" * 1000
        fs = self._SessionFS(data)
        shard = PinnedShard(
            name=full_name("shard.csv"),
            generation="1337",
            size=len(data),
            compressed=False,
            shard_index=0,
        )

        async def consume():
            # Explicit aclose in finally mirrors InventorySnapshot's cleanup.
            generator = InventoryReport._iter_media_chunks(fs, DEST, shard, 7)
            try:
                async for _ in generator:
                    await asyncio.sleep(0)
            finally:
                await generator.aclose()

        task = asyncio.ensure_future(consume())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert fs.session.responses[0].closed is True


class TestMemoryProfile:
    @pytest.mark.asyncio
    async def test_streaming_peak_is_bounded_vs_full_materialization(self, monkeypatch):
        rows = [
            [f"object-key-{i:06d}", str(i), "2024-01-01T00:00:00Z", f"tag-{i}"]
            for i in range(80_000)
        ]
        inputs = make_snapshot_inputs(
            [rows[:30_000], rows[30_000:60_000], rows[60_000:]],
            DEFAULT_FIELDS,
        )
        payload_bytes = sum(len(p) for p in inputs["raw_payloads"])
        assert payload_bytes > 4 * 1024 * 1024

        fs = FakeFileSystem(inputs["config"], inputs["listing"], inputs["store"])
        generations = {
            full_name(name): str(1000 + i)
            for i, name in enumerate(inputs["shard_names"])
        }
        install_seam(monkeypatch, fs, StreamTracker(), generations)

        # Streaming path: tiny chunks, tiny queue, nothing retained.
        tracemalloc.start()
        streaming_snapshot = await InventoryReport.iter_snapshot(
            fs,
            report_info(),
            chunk_size=256,
            record_queue_size=8,
        )
        streamed = 0
        async with streaming_snapshot:
            async for obj in streaming_snapshot:
                streamed += 1
                del obj
        _, streaming_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert streamed == 80_000

        # Naive old-style path: download everything, decode, split, list.
        tracemalloc.start()
        objects = []
        for payload in inputs["raw_payloads"]:
            text = payload.decode()
            for line in text.split("\n"):
                if not line:
                    continue
                objects.append(line.split(","))
        _, naive_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert len(objects) > 80_000

        # Streaming must use substantially less peak memory than holding the
        # full decoded report plus a per-line list.
        assert streaming_peak < naive_peak * 0.6, (
            streaming_peak,
            naive_peak,
        )
        assert streaming_snapshot._queue.maxsize == 8


def test_default_queue_size_is_bounded():
    assert 0 < DEFAULT_RECORD_QUEUE_SIZE <= 2000
