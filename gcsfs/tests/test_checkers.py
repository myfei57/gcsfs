import base64
import os
import random
from base64 import b64decode, b64encode
from hashlib import md5

import pytest

from gcsfs.checkers import Crc32cChecker, MD5Checker, SizeChecker, crcmod
from gcsfs.retry import ChecksumError


def google_response_from_data(expected_data: bytes, actual_data=None):
    actual_data = actual_data or expected_data
    checksum = md5(actual_data)
    checksum_b64 = base64.b64encode(checksum.digest()).decode("UTF-8")
    if crcmod is not None:
        checksum = crcmod.Crc(0x11EDC6F41, initCrc=0, xorOut=0xFFFFFFFF)
        checksum.update(actual_data)
        crc = base64.b64encode(checksum.digest()).decode()

    class response:
        content_length = len(actual_data)
        headers = {"X-Goog-Hash": f"md5={checksum_b64}"}
        if crcmod is not None:
            headers["X-Goog-Hash"] += f", crc32c={crc}"

    return response


def google_response_from_data_with_reverse_header_order(
    expected_data: bytes, actual_data=None
):
    actual_data = actual_data or expected_data
    checksum = md5(actual_data)
    checksum_b64 = base64.b64encode(checksum.digest()).decode("UTF-8")
    if crcmod is not None:
        checksum = crcmod.Crc(0x11EDC6F41, initCrc=0, xorOut=0xFFFFFFFF)
        checksum.update(actual_data)
        crc = base64.b64encode(checksum.digest()).decode()

    class response:
        content_length = len(actual_data)
        headers = {}
        if crcmod is not None:
            headers["X-Goog-Hash"] = f"crc32c={crc}, md5={checksum_b64}"
        else:
            headers["X-Goog-Hash"] = f"md5={checksum_b64}"

    return response


def google_json_response_from_data(expected_data: bytes, actual_data=None):
    actual_data = actual_data or expected_data
    checksum = md5(actual_data)
    checksum_b64 = base64.b64encode(checksum.digest()).decode("UTF-8")

    response = {"md5Hash": checksum_b64, "size": len(actual_data)}

    # some manual checksums verified using gsutil ls -L
    # also can add using https://crccalc.com/
    # be careful about newlines
    crc32c_points = {
        b"hello world\n": "8P9ykg==",
        b"different checksum": "DoesntMatter==",
    }

    try:
        response["crc32c"] = crc32c_points[actual_data]
    except KeyError:
        pass

    return response


params = [
    (MD5Checker(), b"hello world", b"different checksum", (ChecksumError,)),
    (MD5Checker(), b"hello world", b"hello world", ()),
]

if crcmod is not None:
    params.append(
        (Crc32cChecker(), b"hello world", b"different checksum", (ChecksumError,))
    )
    params.append((Crc32cChecker(), b"hello world", b"hello world", ()))


@pytest.mark.parametrize("checker, data, actual_data, raises", params)
def test_validate_headers(checker, data, actual_data, raises):
    response = google_response_from_data(actual_data)
    checker.update(data)

    if raises:
        with pytest.raises(raises):
            checker.validate_headers(response.headers)
    else:
        checker.validate_headers(response.headers)


params = [
    (MD5Checker(), b"hello world", b"different checksum", (ChecksumError,)),
    (MD5Checker(), b"hello world", b"hello world", ()),
]

if crcmod is not None:
    params.append(
        (Crc32cChecker(), b"hello world", b"different checksum", (ChecksumError,))
    )
    params.append((Crc32cChecker(), b"hello world", b"hello world", ()))


@pytest.mark.parametrize("checker, data, actual_data, raises", params)
def test_validate_headers_with_reverse_order(checker, data, actual_data, raises):
    response = google_response_from_data_with_reverse_header_order(actual_data)
    checker.update(data)

    if raises:
        with pytest.raises(raises):
            checker.validate_headers(response.headers)
    else:
        checker.validate_headers(response.headers)


params = [
    (MD5Checker(), b"hello world", b"different checksum", (ChecksumError,)),
    (MD5Checker(), b"hello world", b"hello world", ()),
    (SizeChecker(), b"hello world", b"hello world", ()),
    (SizeChecker(), b"hello world", b"different size", (ChecksumError,)),
]

if crcmod is not None:
    params.append((Crc32cChecker(), b"hello world", b"hello world", ()))
    params.append(
        (Crc32cChecker(), b"hello world", b"different size", (ChecksumError,))
    )


@pytest.mark.parametrize("checker, data, actual_data, raises", params)
def test_checker_validate_http_response(checker, data, actual_data, raises):
    response = google_response_from_data(data, actual_data=actual_data)
    checker.update(data)
    if raises:
        with pytest.raises(raises):
            checker.validate_http_response(response)
    else:
        checker.validate_http_response(response)


params = [
    (MD5Checker(), b"hello world", b"different checksum", (ChecksumError,)),
    (MD5Checker(), b"hello world", b"hello world", ()),
    (SizeChecker(), b"hello world", b"hello world", ()),
    (SizeChecker(), b"hello world", b"different size", (ChecksumError,)),
]
if crcmod is not None:
    params.extend(
        [
            (Crc32cChecker(), b"hello world", b"different checksum", (ChecksumError,)),
            (Crc32cChecker(), b"hello world\n", b"hello world\n", ()),
        ]
    )


@pytest.mark.parametrize("checker, data, actual_data, raises", params)
def test_checker_validate_json_response(checker, data, actual_data, raises):
    response = google_json_response_from_data(data, actual_data=actual_data)
    checker.update(data)
    if raises:
        with pytest.raises(raises):
            checker.validate_json_response(response)
    else:
        checker.validate_json_response(response)


def test_get_consistency_checker():
    from gcsfs.checkers import ConsistencyChecker, get_consistency_checker

    assert isinstance(get_consistency_checker("size"), SizeChecker)
    assert isinstance(get_consistency_checker("md5"), MD5Checker)
    assert isinstance(get_consistency_checker(None), ConsistencyChecker)
    assert isinstance(get_consistency_checker("unknown"), ConsistencyChecker)

    if crcmod is not None:
        assert isinstance(get_consistency_checker("crc32c"), Crc32cChecker)


def test_consistency_checker_base():
    from gcsfs.checkers import ConsistencyChecker

    base = ConsistencyChecker()
    base.update(b"abc")
    base.validate_json_response({})
    base.validate_headers({})
    base.validate_http_response(None)


def test_checker_missing_headers():
    md5_checker = MD5Checker()
    md5_checker.update(b"test")
    # Headers present but without md5
    with pytest.raises(NotImplementedError, match="No md5 checksum"):
        md5_checker.validate_headers({"X-Goog-Hash": "crc32c=12345"})

    if crcmod is not None:
        crc_checker = Crc32cChecker()
        crc_checker.update(b"test")
        with pytest.raises(NotImplementedError, match="No crc32c checksum"):
            crc_checker.validate_headers({"X-Goog-Hash": "md5=12345"})


# ---------------------------------------------------------------------------
# CRC32C polynomial combination (parallel chunk mode)
# ---------------------------------------------------------------------------

from gcsfs.checkers import (
    CRC32C_XOR_OUT,
    Crc32cChunkHasher,
    Crc32cCombiner,
    combine_crc32c,
    crc32c_apply_zeros,
    crc32c_b64_to_value,
    crc32c_raw_to_b64,
    crc32c_raw_to_value,
)


def _raw_crc32c(data):
    hasher = Crc32cChunkHasher()
    hasher.update(data)
    return hasher.raw_value


def _serial_crc32c_b64(data):
    """Independent serial baseline using the existing GCS-convention checker."""
    checker = Crc32cChecker()
    checker.update(data)
    return b64encode(checker.crc32c.digest()).decode()


@pytest.mark.skipif(crcmod is None, reason="crcmod is not installed")
@pytest.mark.parametrize("seed", range(20))
def test_crc32c_combine_random_splits_vs_serial(seed):
    rng = random.Random(seed)
    data = os.urandom(rng.randint(0, 4000))
    n_chunks = rng.randint(1, 8)
    cuts = sorted(rng.sample(range(1, len(data)), min(n_chunks - 1, len(data) - 1)))
    bounds = [0, *cuts, len(data)]
    pieces = [data[bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)]
    ranges = [(bounds[i], bounds[i + 1] - bounds[i]) for i in range(len(bounds) - 1)]

    serial_raw = _raw_crc32c(data)

    # Merge adjacent segments in a random order: completion order must not
    # affect the polynomial combination.
    segments = [
        (bounds[i], bounds[i + 1], _raw_crc32c(pieces[i])) for i in range(len(pieces))
    ]
    while len(segments) > 1:
        i = rng.randrange(len(segments) - 1)
        a = segments[i]
        b = segments.pop(i + 1)
        segments[i] = (
            a[0],
            b[1],
            combine_crc32c(a[2], b[2], b[1] - b[0]),
        )
    assert segments[0][2] == serial_raw

    # The combiner accepts chunks out of order and produces the same result.
    combiner = Crc32cCombiner(ranges, len(data))
    order = list(range(len(ranges)))
    rng.shuffle(order)
    for index in order:
        start, length = ranges[index]
        combiner.add_chunk(index, start, start + length, _raw_crc32c(pieces[index]))
    assert combiner.raw_crc() == serial_raw
    assert combiner.b64_digest() == _serial_crc32c_b64(data)


@pytest.mark.skipif(crcmod is None, reason="crcmod is not installed")
def test_crc32c_combine_zero_length():
    raw_a = _raw_crc32c(b"abc")
    raw_empty = _raw_crc32c(b"")
    assert raw_empty == 0
    # Zero-length second interval is the identity combination.
    assert combine_crc32c(raw_a, raw_empty, 0) == raw_a
    # Zero-length first interval leaves the second CRC untouched.
    assert combine_crc32c(raw_empty, raw_a, 3) == raw_a
    # Zero-length object maps to the standard empty-input GCS checksum.
    assert crc32c_raw_to_b64(0, 0) == "AAAAAA=="
    assert crc32c_b64_to_value("AAAAAA==") == 0


@pytest.mark.skipif(crcmod is None, reason="crcmod is not installed")
def test_crc32c_combine_non_power_of_two_lengths():
    data = os.urandom(100003)
    cut = 997  # deliberately awkward split
    merged = combine_crc32c(
        _raw_crc32c(data[:cut]), _raw_crc32c(data[cut:]), len(data) - cut
    )
    assert merged == _raw_crc32c(data)


@pytest.mark.skipif(crcmod is None, reason="crcmod is not installed")
def test_crc32c_apply_zeros_length_over_32_bits():
    huge = (1 << 33) + 1234567
    left, right = huge // 3, huge - huge // 3
    # Operators are powers of one base operator, so T_(left+right) must equal
    # T_right o T_left even for lengths that do not fit in 32 bits.
    for value in (0, 1, 0xDEADBEEF, CRC32C_XOR_OUT):
        composed = crc32c_apply_zeros(crc32c_apply_zeros(value, left), right)
        direct = crc32c_apply_zeros(value, left + right)
        assert composed == direct

    # A virtual >32-bit tail length still combines consistently.
    raw_a, raw_b = _raw_crc32c(b"A-part"), _raw_crc32c(b"B-part")
    assert combine_crc32c(raw_a, raw_b, huge) == (
        crc32c_apply_zeros(raw_a, huge) ^ raw_b
    )


@pytest.mark.skipif(crcmod is None, reason="crcmod is not installed")
def test_crc32c_combine_is_not_plain_xor():
    first, second = b"foo", b"barbaz"
    xor_guess = _raw_crc32c(first) ^ _raw_crc32c(second)
    assert xor_guess != _raw_crc32c(first + second)
    assert combine_crc32c(
        _raw_crc32c(first), _raw_crc32c(second), len(second)
    ) == _raw_crc32c(first + second)


@pytest.mark.skipif(crcmod is None, reason="crcmod is not installed")
def test_crc32c_gcs_b64_convention():
    # Known vector, big-endian base64, matching the serial Crc32cChecker and
    # published GCS representation.
    sample = b"hello world\n"
    expected_b64 = "8P9ykg=="
    assert _serial_crc32c_b64(sample) == expected_b64
    assert crc32c_raw_to_b64(_raw_crc32c(sample), len(sample)) == expected_b64
    value = crc32c_b64_to_value(expected_b64)
    assert value == crc32c_raw_to_value(_raw_crc32c(sample), len(sample))
    assert value.to_bytes(4, "big") == b64decode(expected_b64)


@pytest.mark.skipif(crcmod is None, reason="crcmod is not installed")
def test_crc32c_combiner_rejects_overlaps_holes_and_duplicates():
    ranges = [(0, 4), (4, 4)]

    # Bad plans: hole, overlap and wrong total size.
    with pytest.raises(ValueError):
        Crc32cCombiner([(0, 4), (5, 3)], 8)
    with pytest.raises(ValueError):
        Crc32cCombiner([(0, 4), (3, 4)], 7)
    with pytest.raises(ValueError):
        Crc32cCombiner(ranges, 9)

    combiner = Crc32cCombiner(ranges, 8)
    combiner.add_chunk(0, 0, 4, _raw_crc32c(b"abcd"))
    with pytest.raises(ChecksumError):
        combiner.add_chunk(0, 0, 4, _raw_crc32c(b"abcd"))
    with pytest.raises(ChecksumError):
        combiner.add_chunk(1, 4, 7, _raw_crc32c(b"efg"))  # wrong interval
    with pytest.raises(ChecksumError):
        combiner.raw_crc()  # hole: index 1 missing

    combiner.add_chunk(1, 4, 8, _raw_crc32c(b"efgh"))
    assert combiner.is_complete()
    assert combiner.b64_digest() == _serial_crc32c_b64(b"abcdefgh")

    with pytest.raises(ChecksumError):
        combiner.validate_json_response({"crc32c": "DoesntMatter=="})
    with pytest.raises(NotImplementedError):
        combiner.validate_json_response({})


@pytest.mark.skipif(crcmod is None, reason="crcmod is not installed")
def test_crc32c_chunk_hasher_requires_crcmod(monkeypatch):
    import gcsfs.checkers as checker_module

    monkeypatch.setattr(checker_module, "crcmod", None)
    with pytest.raises(ImportError, match="crcmod"):
        Crc32cChunkHasher()
