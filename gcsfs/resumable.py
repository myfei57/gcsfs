"""
Checkpoint-backed resumable upload sessions for Google Cloud Storage.

This module implements a resumable upload API that survives process restarts.
The GCS resumable session URL, the source file identity, the frozen target
parameters and the *server-confirmed* offset are persisted in a versioned,
checksummed, atomically updated checkpoint file that is private to the caller.

Safety rules implemented here:

* The byte offset is only advanced on a ``308`` ``Range`` acknowledgement (or
  on the final ``200``/``201``). When the server only received part of a chunk,
  or the previous response was lost, upload resumes from the position the
  server actually confirms -- never from a local "bytes sent" counter.
* A ``308`` without a ``Range`` header means the server holds no bytes, so the
  upload is rewound to offset zero. ``404``/``410`` mean the session is gone;
  a new session is never created implicitly and the target is never
  overwritten without an explicit caller action.
* Non-final chunks are multiples of 256 KiB; the final chunk (including the
  empty-file case) may have any length.
* The source file size/mtime and the digest of every confirmed prefix are
  verified before any byte is sent on resume. Target parameters
  (``ifGenerationMatch``, metadata, content type, KMS key, requester pays) are
  frozen in the checkpoint and re-validated against the caller arguments.
* The session URL is only ever written to the caller-named private checkpoint
  file. It never appears in log records or in exception messages.
* At most one HTTP request is in flight per session, memory is bounded by the
  chunk size, and retries are bounded. Only re-openable local file paths are
  accepted; non-replayable streams are rejected up front.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import mimetypes
import os
import re
import tempfile
from datetime import datetime, timezone

import aiohttp

from .core import GCS_MAX_BLOCK_SIZE, GCS_MIN_BLOCK_SIZE

logger = logging.getLogger("gcsfs.resumable")

CHECKPOINT_VERSION = 1
_DIGEST_ALG = "sha256"
# Bounded read size when re-hashing a confirmed prefix after a restart.
_PREFIX_VERIFY_CHUNK = 8 * 2**20
_DEFAULT_CHUNKSIZE = 50 * 2**20
_DEFAULT_MAX_ATTEMPTS = 6
# Cap exponential backoff so retry behaviour stays bounded.
_MAX_BACKOFF = 30.0
# HTTP statuses worth retrying before giving up on a single request.
_TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

STATE_UPLOADING = "uploading"
STATE_FINAL_UNCERTAIN = "final_uncertain"
STATE_COMPLETE = "complete"
_STATES = frozenset({STATE_UPLOADING, STATE_FINAL_UNCERTAIN, STATE_COMPLETE})

_RANGE_RE = re.compile(r"bytes=(\d+)-(\d+)$")
_CONTENT_RANGE_RE = re.compile(r"bytes (\d+)-(\d+)/(\*|\d+)$")


class ResumableUploadError(Exception):
    """Base class for resumable upload failures.

    Messages reference the target object or local file, never the resumable
    session URL, which must stay confined to the private checkpoint.
    """


class ResumableUploadCheckpointError(ResumableUploadError):
    """The checkpoint is missing, corrupt, tampered with or unsupported."""


class ResumableUploadSessionExpired(ResumableUploadError):
    """The session responded 404/410. A new session requires explicit action."""


class ResumableUploadSourceChanged(ResumableUploadError):
    """The source file no longer matches the identity recorded at upload."""


class ResumableUploadTargetMismatch(ResumableUploadError):
    """Resume arguments differ from the target parameters frozen initially."""


class ResumableUploadFinalUncertain(ResumableUploadError):
    """The final request may or may not have completed; the object may exist."""


class ResumableUploadStateError(ResumableUploadError):
    """The requested operation is not valid for the current checkpoint state."""


# ---------------------------------------------------------------------------
# Checkpoint persistence
# ---------------------------------------------------------------------------


def _canonical(payload: dict) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _build_envelope(payload: dict) -> dict:
    return {
        "version": CHECKPOINT_VERSION,
        "digest_alg": _DIGEST_ALG,
        "checksum": hashlib.sha256(_canonical(payload)).hexdigest(),
        "payload": payload,
    }


def save_checkpoint(checkpoint_path: str, payload: dict) -> None:
    """Atomically write a versioned, checksummed checkpoint with private perms."""
    checkpoint_path = os.path.abspath(os.fspath(checkpoint_path))
    directory = os.path.dirname(checkpoint_path)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    envelope = _build_envelope(payload)
    serialized = json.dumps(envelope, sort_keys=True).encode("utf-8")

    fd, tmp_path = tempfile.mkstemp(
        dir=directory, prefix=".gcsfs-resumable-", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, checkpoint_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    # Best-effort directory fsync so the rename survives a crash. This is not
    # available on every platform (e.g. Windows); the atomic replace already
    # guarantees readers never observe a half-written file.
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load and fully validate a checkpoint.

    Any parse, version, checksum or structural failure raises
    ``ResumableUploadCheckpointError``; callers must not touch the network in
    that case, because the session URL and confirmed offset cannot be trusted.
    """
    checkpoint_path = os.path.abspath(os.fspath(checkpoint_path))
    try:
        with open(checkpoint_path, "rb") as handle:
            raw = handle.read()
        envelope = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ResumableUploadCheckpointError(
            f"resumable upload checkpoint at {checkpoint_path} cannot be read"
        ) from exc

    if not isinstance(envelope, dict) or envelope.get("version") != CHECKPOINT_VERSION:
        raise ResumableUploadCheckpointError(
            f"unsupported resumable upload checkpoint version: {checkpoint_path}"
        )
    if envelope.get("digest_alg") != _DIGEST_ALG:
        raise ResumableUploadCheckpointError(
            f"unsupported checkpoint digest algorithm: {checkpoint_path}"
        )
    payload = envelope.get("payload")
    expected_checksum = envelope.get("checksum")
    if not isinstance(payload, dict) or not isinstance(expected_checksum, str):
        raise ResumableUploadCheckpointError(
            f"malformed resumable upload checkpoint: {checkpoint_path}"
        )
    actual_checksum = hashlib.sha256(_canonical(payload)).hexdigest()
    if not hmac.compare_digest(actual_checksum, expected_checksum):
        raise ResumableUploadCheckpointError(
            f"checksum verification failed for resumable upload checkpoint "
            f"{checkpoint_path}; refusing to contact the upload session"
        )
    try:
        _require_fields(payload)
    except (KeyError, TypeError) as exc:
        raise ResumableUploadCheckpointError(
            f"malformed resumable upload checkpoint: {checkpoint_path}"
        ) from exc
    if payload["state"] not in _STATES:
        raise ResumableUploadCheckpointError(
            f"unknown checkpoint state {payload['state']!r}: {checkpoint_path}"
        )
    return payload


def _require_fields(payload: dict) -> None:
    """Touch every required field so a structurally short checkpoint fails."""
    _ = payload["session_url"]
    _ = payload["state"]
    _ = payload["total_size"]
    _ = payload["confirmed_offset"]
    _ = payload["generation"]
    target = payload["target"]
    for field in (
        "bucket",
        "key",
        "rpath",
        "content_type",
        "metadata",
        "fixed_key_metadata",
        "kms_key_name",
        "if_generation_match",
        "mode",
        "requester_pays",
    ):
        _ = target[field]
    source = payload["source"]
    for field in ("path", "size", "mtime_ns", "prefix_digest"):
        _ = source[field]


# ---------------------------------------------------------------------------
# Identity handling
# ---------------------------------------------------------------------------


def _source_identity(lpath: str) -> dict:
    if not os.path.isfile(lpath):
        raise ResumableUploadSourceChanged(f"source file {lpath} is not a regular file")
    stat = os.stat(lpath)
    return {
        "path": os.path.abspath(lpath),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _hash_prefix(lpath: str, length: int) -> "hashlib._Hash":
    """Hash the first ``length`` bytes of the source in bounded reads."""
    hasher = hashlib.sha256()
    remaining = length
    with open(lpath, "rb") as handle:
        while remaining > 0:
            bit = handle.read(min(remaining, _PREFIX_VERIFY_CHUNK))
            if not bit:
                raise ResumableUploadSourceChanged(
                    f"source file {lpath} shrank below its recorded size"
                )
            hasher.update(bit)
            remaining -= len(bit)
    return hasher


def _verify_source(source: dict, lpath: str, confirmed_offset: int) -> "hashlib._Hash":
    """Validate size/mtime and the confirmed-prefix digest, returning a hasher
    positioned at the confirmed offset."""
    identity = _source_identity(lpath)
    for field in ("path", "size", "mtime_ns"):
        if identity[field] != source[field]:
            raise ResumableUploadSourceChanged(
                f"source file {lpath} identity field {field!r} changed since the "
                "checkpoint was written; refusing to resume"
            )
    hasher = _hash_prefix(lpath, confirmed_offset)
    if not hmac.compare_digest(hasher.hexdigest(), source["prefix_digest"]):
        raise ResumableUploadSourceChanged(
            f"content digest of the confirmed prefix of {lpath} no longer matches "
            "the checkpoint; refusing to resume"
        )
    return hasher


def _freeze_target(
    *,
    bucket: str,
    key: str,
    rpath: str,
    content_type: str,
    metadata: dict | None,
    fixed_key_metadata: dict | None,
    kms_key_name: str | None,
    if_generation_match: int | None,
    mode: str,
    requester_pays,
) -> dict:
    return {
        "bucket": bucket,
        "key": key,
        "rpath": rpath,
        "content_type": content_type,
        "metadata": dict(metadata or {}),
        "fixed_key_metadata": dict(fixed_key_metadata or {}),
        "kms_key_name": kms_key_name,
        "if_generation_match": if_generation_match,
        "mode": mode,
        "requester_pays": requester_pays,
    }


def _validate_target(frozen: dict, current: dict) -> None:
    mismatched = sorted(
        field for field in frozen if frozen[field] != current.get(field)
    )
    if mismatched:
        raise ResumableUploadTargetMismatch(
            "resumable upload target parameters changed "
            f"({', '.join(mismatched)}); refusing to resume the existing session"
        )


def _normalize_chunksize(chunksize: int) -> int:
    size = (int(chunksize) // GCS_MIN_BLOCK_SIZE) * GCS_MIN_BLOCK_SIZE
    size = max(size, GCS_MIN_BLOCK_SIZE)
    return min(size, GCS_MAX_BLOCK_SIZE)


# ---------------------------------------------------------------------------
# HTTP transport against a resumable session
# ---------------------------------------------------------------------------


def _header(headers: dict | None, name: str):
    if not headers:
        return None
    lowered = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered:
            return value
    return None


async def _session_request(
    fs,
    target: str,
    method: str,
    url: str,
    *,
    headers: dict | None = None,
    data=None,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
):
    """Perform one bounded-retry HTTP exchange against a session URL.

    The session URL is never logged and never embedded in a raised exception.
    """
    await fs._set_session()
    for attempt in range(max_attempts):
        try:
            async with fs.session.request(
                method=method,
                url=url,
                params=fs._get_params({}),
                headers=fs._get_headers(headers or {}),
                data=data,
                timeout=fs.requests_timeout,
            ) as response:
                status = response.status
                response_headers = dict(response.headers)
                body = await response.read()
            if status in _TRANSIENT_STATUSES and attempt + 1 < max_attempts:
                logger.warning(
                    "resumable upload for %s received transient status %s; retrying",
                    target,
                    status,
                )
                await asyncio.sleep(min(2.0**attempt, _MAX_BACKOFF))
                continue
            return status, response_headers, body
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if attempt + 1 >= max_attempts:
                raise ResumableUploadError(
                    f"lost connection to the resumable upload session for {target} "
                    f"after {max_attempts} attempts"
                ) from exc
            logger.warning(
                "resumable upload for %s hit a transport error; retrying", target
            )
            await asyncio.sleep(min(2.0**attempt, _MAX_BACKOFF))
    # Unreachable: the loop either returns or raises.
    raise ResumableUploadError(
        f"resumable upload for {target} failed without a server response"
    )


def _confirmed_from_range(headers: dict) -> int | None:
    """Return the confirmed byte offset from a 308 ``Range`` header."""
    raw = _header(headers, "Range")
    if not raw:
        return None
    match = _RANGE_RE.match(raw.strip())
    if not match:
        raise ResumableUploadError(
            "malformed Range header in resumable upload response"
        )
    return int(match.group(2)) + 1


def _parse_object(body) -> dict | None:
    if not body:
        return None
    try:
        parsed = json.loads(body)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


async def _query_status(fs, target: str, url: str, total_size: int, max_attempts: int):
    """Ask GCS how many bytes the session currently holds.

    Returns ``(state, confirmed_offset, object_json_or_None)``.
    """
    headers = {
        "Content-Range": f"bytes */{total_size}",
        "Content-Length": "0",
    }
    status, response_headers, body = await _session_request(
        fs,
        target,
        "POST",
        url,
        headers=headers,
        data=None,
        max_attempts=max_attempts,
    )
    if status in (200, 201):
        return STATE_COMPLETE, total_size, _parse_object(body)
    if status == 308:
        confirmed = _confirmed_from_range(response_headers)
        # A 308 without Range means the server has not stored any bytes.
        return STATE_UPLOADING, 0 if confirmed is None else confirmed, None
    if status in (404, 410):
        raise ResumableUploadSessionExpired(
            f"resumable upload session for {target} expired ({status}); start a "
            "new upload explicitly instead of overwriting the target implicitly"
        )
    if status == 412:
        raise FileExistsError(target)
    raise ResumableUploadError(
        f"unexpected status {status} while querying the resumable session for {target}"
    )


# ---------------------------------------------------------------------------
# Upload orchestration
# ---------------------------------------------------------------------------


class _ResumableUpload:
    def __init__(
        self,
        fs,
        lpath: str,
        rpath: str,
        checkpoint_path: str,
        *,
        metadata,
        content_type,
        fixed_key_metadata,
        mode,
        if_generation_match,
        kms_key_name,
        chunksize,
        max_attempts,
    ):
        from . import core

        self.fs = fs
        self.core = core
        self.lpath = lpath
        self.rpath = rpath
        self.checkpoint_path = os.path.abspath(os.fspath(checkpoint_path))
        self.metadata = metadata
        self.fixed_key_metadata = fixed_key_metadata
        self.mode = mode
        self.kms_key_name = kms_key_name
        self.chunksize = _normalize_chunksize(chunksize)
        self.max_attempts = max_attempts

        bucket, key, generation = fs.split_path(rpath)
        if generation:
            raise ValueError("Cannot write to specific object generation")
        self.bucket, self.key = bucket, key
        self.if_generation_match = core._resolve_generation_match(
            mode, if_generation_match
        )
        if content_type is None:
            guessed, _ = mimetypes.guess_type(lpath)
            content_type = guessed or "application/octet-stream"
        self.content_type = content_type

        identity = _source_identity(lpath)
        self.source_identity = identity
        self.total_size = identity["size"]

        self.target = _freeze_target(
            bucket=bucket,
            key=key,
            rpath=rpath,
            content_type=content_type,
            metadata=metadata,
            fixed_key_metadata=fixed_key_metadata,
            kms_key_name=kms_key_name,
            if_generation_match=self.if_generation_match,
            mode=mode,
            requester_pays=fs.requester_pays,
        )

        self.session_url: str | None = None
        self.confirmed = 0
        self.hasher = hashlib.sha256()
        self.state = STATE_UPLOADING
        self.generation = None
        self.created_at = datetime.now(timezone.utc).isoformat()

    async def run(self) -> dict:
        if os.path.exists(self.checkpoint_path):
            await self._resume()
        else:
            await self._start_new()
        if self.state != STATE_COMPLETE:
            await self._upload_loop()
        return self.result()

    async def _start_new(self) -> None:
        logger.info("starting new resumable upload %s -> %s", self.lpath, self.rpath)
        self.session_url = await self.core._http_initiate_upload(
            self.fs,
            self.bucket,
            self.key,
            self.content_type,
            self.metadata,
            self.fixed_key_metadata,
            self.mode,
            self.kms_key_name,
            self.if_generation_match,
        )
        self.confirmed = 0
        self.hasher = hashlib.sha256()
        self.state = STATE_UPLOADING
        self._save()

    async def _resume(self) -> None:
        payload = load_checkpoint(self.checkpoint_path)
        _validate_target(payload["target"], self.target)
        self.session_url = payload["session_url"]
        self.generation = payload.get("generation")
        self.created_at = payload.get("created_at", self.created_at)

        if payload["state"] == STATE_COMPLETE:
            logger.info("resumable upload for %s is already complete", self.rpath)
            self.state = STATE_COMPLETE
            self.confirmed = payload["total_size"]
            return

        # Source identity and confirmed-prefix digest are verified before any
        # network contact.
        self.hasher = _verify_source(
            payload["source"], self.lpath, payload["confirmed_offset"]
        )
        self.confirmed = payload["confirmed_offset"]
        self.state = payload["state"]

        # Always ask the server first; a local "sent" length is not trusted.
        server_state, server_confirmed, obj = await _query_status(
            self.fs,
            self.rpath,
            self.session_url,
            self.total_size,
            self.max_attempts,
        )
        if server_state == STATE_COMPLETE:
            logger.info("resumable session for %s was already finalized", self.rpath)
            self.confirmed = self.total_size
            self.hasher = _hash_prefix(self.lpath, self.total_size)
            self.state = STATE_COMPLETE
            self.generation = (obj or {}).get("generation")
            self._save()
            return

        if server_confirmed < self.confirmed:
            # The server is authoritative for what it stored.
            logger.warning(
                "server for %s confirms %d bytes, behind local checkpoint %d; "
                "rewinding",
                self.rpath,
                server_confirmed,
                self.confirmed,
            )
            self.confirmed = server_confirmed
            self.hasher = _hash_prefix(self.lpath, server_confirmed)
        elif server_confirmed > self.confirmed:
            # A previous response was lost: extend the verified prefix.
            logger.info(
                "server for %s is ahead at %d bytes (local checkpoint %d)",
                self.rpath,
                server_confirmed,
                self.confirmed,
            )
            extra = server_confirmed - self.confirmed
            with open(self.lpath, "rb") as handle:
                handle.seek(self.confirmed)
                while extra > 0:
                    bit = handle.read(min(extra, _PREFIX_VERIFY_CHUNK))
                    if not bit:
                        raise ResumableUploadSourceChanged(
                            f"source file {self.lpath} shrank during resume"
                        )
                    self.hasher.update(bit)
                    extra -= len(bit)
            self.confirmed = server_confirmed
        # Equal offsets (including repeated 308 acks) change nothing.
        self.state = STATE_UPLOADING
        self._save()

    async def _upload_loop(self) -> None:
        stalled_attempts = 0
        with open(self.lpath, "rb") as handle:
            handle.seek(self.confirmed)
            while self.state != STATE_COMPLETE:
                remaining = self.total_size - self.confirmed
                read_size = min(self.chunksize, remaining)
                data = handle.read(read_size)
                if len(data) != read_size:
                    raise ResumableUploadSourceChanged(
                        f"source file {self.lpath} shrank during upload"
                    )
                is_final = self.confirmed + len(data) == self.total_size
                if data and not is_final and len(data) % GCS_MIN_BLOCK_SIZE:
                    raise ResumableUploadError(
                        "non-final resumable upload chunks must be multiples of "
                        f"{GCS_MIN_BLOCK_SIZE} bytes"
                    )
                content_range = self._content_range(data, is_final)
                headers = {
                    "Content-Range": content_range,
                    "Content-Length": str(len(data)),
                    "Content-Type": self.content_type,
                }
                try:
                    status, response_headers, body = await _session_request(
                        self.fs,
                        self.rpath,
                        "POST",
                        self.session_url,
                        headers=headers,
                        data=data,
                        max_attempts=self.max_attempts,
                    )
                except ResumableUploadError:
                    if is_final:
                        # The object may already exist; never delete it.
                        self.state = STATE_FINAL_UNCERTAIN
                        self._save()
                        raise ResumableUploadFinalUncertain(
                            f"final status of upload to {self.rpath} is unknown; "
                            "resume to reconcile, the object may already exist"
                        )
                    raise

                if status in (200, 201):
                    self.hasher.update(data)
                    self.confirmed += len(data)
                    self.state = STATE_COMPLETE
                    obj = _parse_object(body)
                    self.generation = (obj or {}).get("generation")
                    self._save()
                    logger.info("resumable upload to %s completed", self.rpath)
                    return

                if status == 308:
                    new_confirmed = _confirmed_from_range(response_headers)
                    if new_confirmed is None:
                        # No Range header: the server holds zero bytes.
                        new_confirmed = 0
                    previous = self.confirmed
                    self._reconcile(new_confirmed, handle, data)
                    self._save()
                    if new_confirmed > previous:
                        stalled_attempts = 0
                    else:
                        # Duplicate acknowledgement, a no-Range rewind or any
                        # response that fails to advance the confirmed offset.
                        stalled_attempts += 1
                    if stalled_attempts >= self.max_attempts:
                        raise ResumableUploadError(
                            f"resumable upload to {self.rpath} made no confirmed "
                            f"progress after {self.max_attempts} responses"
                        )
                    continue

                if status in (404, 410):
                    if is_final:
                        # The final response may simply have been lost after
                        # GCS finalized the object: reconcile via status query
                        # on resume instead of declaring the session expired.
                        self.state = STATE_FINAL_UNCERTAIN
                        self._save()
                        raise ResumableUploadFinalUncertain(
                            f"final status of upload to {self.rpath} is unknown "
                            f"({status}); resume to reconcile, the object may "
                            "already exist"
                        )
                    raise ResumableUploadSessionExpired(
                        f"resumable upload session for {self.rpath} expired "
                        f"({status}); start a new upload explicitly"
                    )
                if status == 412:
                    raise FileExistsError(self.rpath)
                raise ResumableUploadError(
                    f"unexpected status {status} while uploading to {self.rpath}"
                )

    def _reconcile(self, new_confirmed: int, handle, data: bytes) -> None:
        previous = self.confirmed
        if new_confirmed < previous:
            self.confirmed = new_confirmed
            self.hasher = _hash_prefix(self.lpath, new_confirmed)
        elif new_confirmed > previous:
            extra = new_confirmed - previous
            if extra > len(data):
                raise ResumableUploadError(
                    "server confirmed more bytes than were sent in the last chunk"
                )
            self.hasher.update(data[:extra])
            self.confirmed = new_confirmed
        # Equal offsets: duplicate acknowledgement, idempotent no-op.
        handle.seek(self.confirmed)

    def _content_range(self, data: bytes, is_final: bool) -> str:
        if not data:
            # Empty final chunk (notably empty files).
            return f"bytes */{self.total_size}"
        end = self.confirmed + len(data) - 1
        if is_final:
            return f"bytes {self.confirmed}-{end}/{self.total_size}"
        return f"bytes {self.confirmed}-{end}/*"

    def _save(self) -> None:
        payload = {
            "session_url": self.session_url,
            "state": self.state,
            "target": self.target,
            "source": {
                "path": self.source_identity["path"],
                "size": self.source_identity["size"],
                "mtime_ns": self.source_identity["mtime_ns"],
                "prefix_digest": self.hasher.hexdigest(),
            },
            "total_size": self.total_size,
            "confirmed_offset": self.confirmed,
            "generation": self.generation,
            "created_at": self.created_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        save_checkpoint(self.checkpoint_path, payload)

    def result(self) -> dict:
        return {
            "state": self.state,
            "confirmed_offset": self.confirmed,
            "total_size": self.total_size,
            "generation": self.generation,
            "checkpoint_path": self.checkpoint_path,
        }


# ---------------------------------------------------------------------------
# Public async entry points
# ---------------------------------------------------------------------------


async def resumable_upload_file(
    fs,
    lpath,
    rpath,
    checkpoint_path,
    *,
    metadata=None,
    content_type=None,
    fixed_key_metadata=None,
    mode="overwrite",
    if_generation_match=None,
    kms_key_name=None,
    chunksize=_DEFAULT_CHUNKSIZE,
    max_attempts=_DEFAULT_MAX_ATTEMPTS,
) -> dict:
    """Upload ``lpath`` to ``rpath`` using a checkpointed resumable session.

    A resumable session is created only when no checkpoint exists. Re-invoking
    the call after a crash (including in a new process) validates the checkpoint
    and source identity, queries the session status and continues from the
    server-confirmed offset. See the module docstring for the safety rules.

    Parameters
    ----------
    fs: GCSFileSystem
    lpath: str or path-like
        Local source file. Streams are rejected because they cannot be
        re-read after a restart.
    rpath: str
        Target GCS object.
    checkpoint_path: str or path-like
        Caller-chosen private location for the checkpoint. The file is created
        with owner-only permissions and is the only place the session URL is
        stored.
    metadata, fixed_key_metadata, content_type, mode, if_generation_match,
    kms_key_name
        Target parameters, frozen for the lifetime of the session and
        re-validated on every resume.
    chunksize: int
        Read/request size. Non-final requests are rounded to a multiple of
        256 KiB and capped at the GCS maximum chunk size.
    max_attempts: int
        Bound on per-request transport retries and on unconfirmed responses.
    """
    if not isinstance(lpath, (str, os.PathLike)):
        raise TypeError(
            "resumable_upload_file requires a re-openable local file path; "
            "streams cannot be replayed after a process restart"
        )
    lpath = os.path.abspath(os.fspath(lpath))
    if not os.path.isfile(lpath):
        raise FileNotFoundError(lpath)

    upload = _ResumableUpload(
        fs,
        lpath,
        rpath,
        checkpoint_path,
        metadata=metadata,
        content_type=content_type,
        fixed_key_metadata=fixed_key_metadata,
        mode=mode,
        if_generation_match=if_generation_match,
        kms_key_name=kms_key_name,
        chunksize=chunksize,
        max_attempts=max_attempts,
    )
    return await upload.run()


async def resumable_upload_status(fs, checkpoint_path) -> dict:
    """Query the server-side status of a checkpointed upload and refresh it.

    The checkpoint is the source of truth for target parameters; the current
    filesystem must still match its requester-pays setting.
    """
    payload = load_checkpoint(checkpoint_path)
    if fs.requester_pays != payload["target"]["requester_pays"]:
        raise ResumableUploadTargetMismatch(
            "requester_pays setting no longer matches the checkpoint"
        )
    if payload["state"] == STATE_COMPLETE:
        return _status_result(payload, checkpoint_path)

    state, confirmed, obj = await _query_status(
        fs,
        payload["target"]["rpath"],
        payload["session_url"],
        payload["total_size"],
        _DEFAULT_MAX_ATTEMPTS,
    )
    if state == STATE_COMPLETE:
        payload["state"] = STATE_COMPLETE
        payload["confirmed_offset"] = payload["total_size"]
        payload["generation"] = (obj or {}).get("generation")
        save_checkpoint(checkpoint_path, payload)
        return _status_result(payload, checkpoint_path)

    if confirmed != payload["confirmed_offset"]:
        # Record the server truth with a digest recomputed from the source.
        source_path = payload["source"]["path"]
        identity = _source_identity(source_path)
        if (
            identity["size"] != payload["source"]["size"]
            or identity["mtime_ns"] != payload["source"]["mtime_ns"]
        ):
            raise ResumableUploadSourceChanged(
                f"source file {source_path} changed; refusing to refresh status"
            )
        hasher = _hash_prefix(source_path, confirmed)
        payload["confirmed_offset"] = confirmed
        payload["source"]["prefix_digest"] = hasher.hexdigest()
        save_checkpoint(checkpoint_path, payload)
    return _status_result(payload, checkpoint_path)


def _status_result(payload: dict, checkpoint_path) -> dict:
    return {
        "state": payload["state"],
        "confirmed_offset": payload["confirmed_offset"],
        "total_size": payload["total_size"],
        "generation": payload.get("generation"),
        "checkpoint_path": os.path.abspath(os.fspath(checkpoint_path)),
    }


async def cancel_resumable_upload(fs, checkpoint_path, *, abandon: bool) -> dict:
    """Finish a checkpointed upload by the explicit retain/abandon protocol.

    * ``abandon=False`` retains both the GCS session and the checkpoint so the
      upload can be resumed later. Nothing is deleted.
    * ``abandon=True`` cancels only this one resumable session and removes the
      checkpoint. No target object and no other session is touched. Abandoning
      is refused while the final status is uncertain, because the object may
      already have been created.
    """
    checkpoint_path = os.path.abspath(os.fspath(checkpoint_path))
    payload = load_checkpoint(checkpoint_path)
    state = payload["state"]

    if state == STATE_FINAL_UNCERTAIN:
        raise ResumableUploadStateError(
            f"upload to {payload['target']['rpath']} has an uncertain final "
            "status; resume it instead of cancelling, so an existing object is "
            "not lost"
        )

    if not abandon:
        logger.info(
            "retaining resumable session for %s at caller request",
            payload["target"]["rpath"],
        )
        return {
            "state": state,
            "abandoned": False,
            "checkpoint_path": checkpoint_path,
        }

    if state != STATE_COMPLETE:
        from . import core

        # Only this session is cancelled; the precondition is stripped because
        # cancellation must not itself evaluate ifGenerationMatch.
        url = core._strip_generation_match(payload["session_url"])
        status, _, _ = await _session_request(
            fs,
            payload["target"]["rpath"],
            "DELETE",
            url,
            max_attempts=_DEFAULT_MAX_ATTEMPTS,
        )
        if status not in (200, 204, 404):
            raise ResumableUploadError(
                f"unexpected status {status} while cancelling the session for "
                f"{payload['target']['rpath']}"
            )
        logger.info("abandoned resumable session for %s", payload["target"]["rpath"])
    os.remove(checkpoint_path)
    return {
        "state": state,
        "abandoned": True,
        "checkpoint_path": checkpoint_path,
    }
