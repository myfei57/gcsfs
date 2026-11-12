import base64
from base64 import b64encode
from hashlib import md5

from .retry import ChecksumError

try:
    import crcmod
except ImportError:
    crcmod = None


class ConsistencyChecker:
    def __init__(self):
        pass

    def update(self, data: bytes):
        pass

    def validate_json_response(self, gcs_object):
        pass

    def validate_headers(self, headers):
        pass

    def validate_http_response(self, r):
        pass


class MD5Checker(ConsistencyChecker):
    def __init__(self):
        self.md = md5()

    def update(self, data):
        self.md.update(data)

    def validate_json_response(self, gcs_object):
        mdback = gcs_object["md5Hash"]
        if b64encode(self.md.digest()) != mdback.encode():
            raise ChecksumError("MD5 checksum failed")

    def validate_headers(self, headers):
        if headers is not None and "X-Goog-Hash" in headers:

            dig = [
                bit.split("=")[1]
                for bit in headers["X-Goog-Hash"].split(",")
                if bit and bit.strip().startswith("md5=")
            ]
            if dig:
                if b64encode(self.md.digest()).decode().rstrip("=") != dig[0]:
                    raise ChecksumError("Checksum failure")
            else:
                raise NotImplementedError(
                    "No md5 checksum available to do consistency check. GCS does "
                    "not provide md5 sums for composite objects."
                )

    def validate_http_response(self, r):
        return self.validate_headers(r.headers)


class SizeChecker(ConsistencyChecker):
    def __init__(self):
        self.size = 0

    def update(self, data: bytes):
        self.size += len(data)

    def validate_json_response(self, gcs_object):
        if int(gcs_object["size"]) != self.size:
            raise ChecksumError("Size mismatch")

    def validate_http_response(self, r):
        if r.content_length != self.size:
            raise ChecksumError("Size mismatch")


class Crc32cChecker(ConsistencyChecker):
    def __init__(self):
        self.crc32c = crcmod.Crc(0x11EDC6F41, initCrc=0, xorOut=0xFFFFFFFF)

    def update(self, data: bytes):
        self.crc32c.update(data)

    def validate_json_response(self, gcs_object):
        # docs for gcs_object: https://cloud.google.com/storage/docs/json_api/v1/objects
        digest = self.crc32c.digest()
        digest_b64 = base64.b64encode(digest).decode()
        expected = gcs_object["crc32c"]

        if digest_b64 != expected:
            raise ChecksumError(f'Expected "{expected}". Got "{digest_b64}"')

    def validate_headers(self, headers):
        if headers is not None:
            hasher = headers.get("X-Goog-Hash", "")
            crc = [h.split("=", 1)[1] for h in hasher.split(",") if "crc32c" in h]
            if not crc:
                raise NotImplementedError("No crc32c checksum was provided by google!")
            if crc[0] != b64encode(self.crc32c.digest()).decode():
                raise ChecksumError()

    def validate_http_response(self, r):
        return self.validate_headers(r.headers)


def get_consistency_checker(consistency: str | None) -> ConsistencyChecker:
    if consistency == "size":
        return SizeChecker()
    elif consistency == "md5":
        return MD5Checker()
    elif consistency == "crc32c":
        if crcmod is None:
            raise ImportError(
                "The python package `crcmod` is required for `consistency='crc32c'`. "
                "This can be installed with `pip install gcsfs[crc]`"
            )
        else:
            return Crc32cChecker()
    else:
        return ConsistencyChecker()


# ---------------------------------------------------------------------------
# CRC32C (Castagnoli) polynomial combination for parallel, out-of-order
# range downloads.
#
# GCS publishes the object CRC32C with the standard convention
# (register init 0xFFFFFFFF, final XOR 0xFFFFFFFF, big-endian base64).
# crcmod's ``Crc(CRC32C_POLY, initCrc=0, xorOut=0xFFFFFFFF)`` implements that
# convention; ``xorOut=0`` exposes the raw shift register (init 0, no final
# XOR) used by the combination algebra below.
#
# CRC is a linear function over GF(2). If ``R(M)`` is the raw register after
# message ``M`` and ``T_n`` denotes processing ``n`` zero bytes through the
# polynomial, then for adjacent byte intervals A, B::
#
#     R(A || B) = T_len(B)(R(A)) XOR R(B)
#
# The same identity holds for the GCS (init/final XOR) values because the
# affine corrections cancel. This is *not* a plain XOR of the two CRCs: the
# first state must be shifted through the polynomial by len(B) zero bytes.
# ``T_n`` is built by GF(2) matrix squaring and binary exponentiation over
# the bits of ``n``, so arbitrary lengths work, including zero, non-power-of
# -two and lengths larger than 32 bits.
# ---------------------------------------------------------------------------

CRC32C_POLY = 0x11EDC6F41
# Reflected (bit-reversed) Castagnoli polynomial, used by the zero-bit operator.
CRC32C_REFLECTED_POLY = 0x82F63B78
CRC32C_XOR_OUT = 0xFFFFFFFF
_GF2_DIM = 32


def _gf2_matrix_times(matrix, vector):
    """Multiply a GF(2) square matrix (row vectors) by a column vector."""
    total = 0
    index = 0
    while vector:
        if vector & 1:
            total ^= matrix[index]
        vector >>= 1
        index += 1
    return total


def _gf2_matrix_square(square, matrix):
    """Square a GF(2) matrix in place: ``square = matrix * matrix``."""
    for n in range(_GF2_DIM):
        square[n] = _gf2_matrix_times(matrix, matrix[n])


def crc32c_apply_zeros(crc, length):
    """Return the register state after ``length`` zero bytes are processed.

    Works for any non-negative ``length``: the operator is assembled by
    squaring per bit of ``length``, so lengths above 32 bits are handled
    without overflow.
    """
    if length < 0:
        raise ValueError("CRC32C zero-operator length must be non-negative")
    if length == 0:
        return crc

    # Operator for a single zero bit in the reflected representation.
    odd = [0] * _GF2_DIM
    even = [0] * _GF2_DIM
    odd[0] = CRC32C_REFLECTED_POLY
    for n in range(1, _GF2_DIM):
        odd[n] = 1 << (n - 1)

    # Two zero bits, then four zero bits.
    _gf2_matrix_square(even, odd)
    _gf2_matrix_square(odd, even)

    remaining = length
    while True:
        # Each squaring doubles the number of zero bits the operator spans;
        # the first iteration here yields the one-zero-byte (8 zero bits) op.
        _gf2_matrix_square(even, odd)
        if remaining & 1:
            crc = _gf2_matrix_times(even, crc)
        remaining >>= 1
        if remaining == 0:
            break

        _gf2_matrix_square(odd, even)
        if remaining & 1:
            crc = _gf2_matrix_times(odd, crc)
        remaining >>= 1
        if remaining == 0:
            break

    return crc


def combine_crc32c(crc_a, crc_b, len_b):
    """Polynomial combination of two adjacent raw CRC32C register states.

    Parameters
    ----------
    crc_a: int
        Raw CRC32C of the preceding interval A.
    crc_b: int
        Raw CRC32C of the following interval B.
    len_b: int
        Byte length of interval B (may be zero, non-power-of-two or > 32 bit).
    """
    return crc32c_apply_zeros(crc_a, len_b) ^ crc_b


def crc32c_raw_to_value(raw_crc, size):
    """Convert a raw register (init 0, no XOR-out) to the GCS CRC32C value."""
    return raw_crc ^ crc32c_apply_zeros(CRC32C_XOR_OUT, size) ^ CRC32C_XOR_OUT


def crc32c_value_to_b64(value):
    """Encode a CRC32C integer using the GCS big-endian base64 representation."""
    return b64encode(value.to_bytes(4, byteorder="big")).decode()


def crc32c_b64_to_value(b64_value):
    """Decode the GCS big-endian base64 CRC32C representation to an integer."""
    return int.from_bytes(base64.b64decode(b64_value), byteorder="big")


def crc32c_raw_to_b64(raw_crc, size):
    """Raw register to the published GCS big-endian base64 CRC32C string."""
    return crc32c_value_to_b64(crc32c_raw_to_value(raw_crc, size))


class Crc32cChunkHasher:
    """Raw CRC32C (init 0, no XOR-out) computed on the bytes actually received.

    Per-chunk hashing deliberately ignores any whole-object hash advertised in
    a range response's headers: GCS returns the *whole object* checksum there,
    which is never the checksum of the requested byte range.
    """

    def __init__(self):
        if crcmod is None:
            raise ImportError(
                "The python package `crcmod` is required for `consistency='crc32c'`. "
                "This can be installed with `pip install gcsfs[crc]`"
            )
        self._crc = crcmod.Crc(CRC32C_POLY, initCrc=0, xorOut=0)

    def update(self, data: bytes):
        self._crc.update(data)

    @property
    def raw_value(self) -> int:
        return self._crc.crcValue


class Crc32cCombiner:
    """Combines per-chunk raw CRC32C values into the whole-object checksum.

    Chunks may be registered in arbitrary completion order; each must match
    its planned, non-overlapping byte interval exactly. Overlaps, holes,
    duplicate completions or intervals outside the plan are rejected instead
    of being treated as valid blocks. No data is buffered or re-read: only the
    fixed-size integer CRC states and their interval lengths are combined.
    """

    def __init__(self, ranges, total_size):
        self.ranges = list(ranges)
        self.total_size = int(total_size)
        self._validate_plan()
        self._chunks = {}

    def _validate_plan(self):
        if self.total_size < 0:
            raise ValueError("total_size must be non-negative")
        cursor = 0
        for index, (start, length) in enumerate(self.ranges):
            if start != cursor or length <= 0:
                raise ValueError(
                    f"Invalid CRC32C chunk plan at index {index}: ranges must be "
                    "contiguous, non-empty and start at zero"
                )
            cursor = start + length
        if cursor != self.total_size:
            raise ValueError(
                "CRC32C chunk plan does not cover the whole object: "
                f"{cursor} planned of {self.total_size} bytes"
            )

    def add_chunk(self, index, start, end, raw_crc):
        """Register a completed chunk's raw CRC against its planned interval."""
        if index in self._chunks:
            raise ChecksumError(f"Duplicate completion of CRC32C chunk index {index}")
        if not 0 <= index < len(self.ranges):
            raise ChecksumError(f"CRC32C chunk index {index} is out of range")
        planned_start, planned_length = self.ranges[index]
        if (start, end) != (planned_start, planned_start + planned_length):
            raise ChecksumError(
                f"CRC32C chunk {index} reported interval [{start}, {end}) but "
                f"[{planned_start}, {planned_start + planned_length}) was planned"
            )
        self._chunks[index] = raw_crc

    @property
    def pending_indices(self):
        return [i for i in range(len(self.ranges)) if i not in self._chunks]

    def is_complete(self):
        return not self.pending_indices

    def raw_crc(self):
        """Fold all chunk CRCs in byte order via the polynomial combination."""
        if not self.is_complete():
            raise ChecksumError(
                "Cannot combine CRC32C with missing chunk(s): "
                f"pending indices {self.pending_indices}"
            )
        merged = 0
        for index, (start, length) in enumerate(self.ranges):
            merged = combine_crc32c(merged, self._chunks[index], length)
        return merged

    def b64_digest(self):
        """Whole-object GCS big-endian base64 CRC32C of the covered bytes."""
        return crc32c_raw_to_b64(self.raw_crc(), self.total_size)

    def validate_json_response(self, gcs_object):
        expected = gcs_object.get("crc32c")
        if not expected:
            raise NotImplementedError(
                "No crc32c checksum is available for this object; refusing to "
                "validate a parallel CRC32C download without a whole-object checksum"
            )
        actual = self.b64_digest()
        if actual != expected:
            raise ChecksumError(f'Expected "{expected}". Got "{actual}"')
