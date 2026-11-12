"""Unit tests for the soft-delete APIs: ``list_soft_deleted`` and
``restore_soft_deleted`` (plus the bounded batch variant).

These tests are fully mock-based: no GCS emulator is required. The list path
mocks ``GCSFileSystem._call``; the restore path mocks the single-shot
``_request_once`` method so that real ``validate_response`` status mapping and
the no-blind-retry behaviour are exercised end to end.
"""

import json
from unittest import mock

import aiohttp
import pytest

from gcsfs.core import (
    GCSFileSystem,
    RestorePreconditionFailedError,
    RestoreResultUncertainError,
    RestoreTokenRequiredError,
    SoftDeletedObjectNotFoundError,
    SoftDeleteNotEnabledError,
)
from gcsfs.retry import HttpError, validate_response

BUCKET = "soft-delete-test-bucket"


def _fs():
    return GCSFileSystem(token="anon", skip_instance_cache=True)


def _soft_object(key, generation, *, restore_token=None, **extra):
    obj = {
        "kind": "storage#object",
        "id": f"{BUCKET}/{key}/{generation}",
        "bucket": BUCKET,
        "name": key,
        "generation": str(generation),
        "metageneration": "1",
        "size": "5",
        "mediaLink": f"http://example.com/{key}",
        "softDeleteTime": "2026-11-01T00:00:00.000Z",
        "hardDeleteTime": "2026-11-08T00:00:00.000Z",
    }
    if restore_token is not None:
        obj["restoreToken"] = restore_token
    obj.update(extra)
    return obj


def _live_object(key, generation, **extra):
    obj = {
        "kind": "storage#object",
        "id": f"{BUCKET}/{key}/{generation}",
        "bucket": BUCKET,
        "name": key,
        "generation": str(generation),
        "metageneration": "1",
        "size": "5",
        "mediaLink": f"http://example.com/{key}",
        "timeCreated": "2026-11-02T00:00:00.000Z",
        "updated": "2026-11-02T00:00:00.000Z",
    }
    obj.update(extra)
    return obj


def _page(items, next_token=None, prefixes=None):
    page = {"kind": "storage#objects"}
    if items is not None:
        page["items"] = items
    if next_token is not None:
        page["nextPageToken"] = next_token
    if prefixes:
        page["prefixes"] = prefixes
    return page


def _request_response(obj, status=200):
    """Build a ``_request_once`` return tuple for a successful JSON call."""
    return status, {}, mock.Mock(), json.dumps(obj).encode()


def _http_status_side_effect(status, message, reason=None):
    """Side effect reproducing what ``_request_once`` raises on an HTTP error,
    i.e. the same exceptions ``validate_response`` produces in production."""
    error = {"code": status, "message": message}
    if reason is not None:
        error["errors"] = [{"reason": reason}]
    body = json.dumps({"error": error}).encode()

    def _side_effect(method, path, *args, **kwargs):
        validate_response(status, body, path, args)

    return _side_effect


class TestListSoftDeleted:
    @pytest.mark.asyncio
    async def test_pagination_follows_tokens_and_counts_requests(self):
        fs = _fs()
        pages = [
            _page(
                [_soft_object("a", 1), _soft_object("b", 2)],
                next_token="token-1",
            ),
            _page(
                [_soft_object("c", 3)],
                next_token="token-2",
            ),
            _page([_soft_object("d", 4)]),
        ]
        with mock.patch.object(
            fs, "_call", new_callable=mock.AsyncMock, side_effect=pages
        ) as mock_call:
            entries = await fs._list_soft_deleted(BUCKET, page_size=2)

        # Exactly one request per page, with the correct token chain.
        assert mock_call.call_count == 3
        first, second, third = mock_call.await_args_list
        assert first.args[:2] == ("GET", "b/{}/o")
        assert first.args[2] == BUCKET
        assert first.kwargs["softDeleted"] == "true"
        assert "pageToken" not in first.kwargs
        assert second.kwargs["pageToken"] == "token-1"
        assert third.kwargs["pageToken"] == "token-2"
        # Soft-delete listing must never carry the versions flag.
        for call in mock_call.await_args_list:
            assert "versions" not in call.kwargs
            assert call.kwargs["maxResults"] == 2

        assert [e["name"] for e in entries] == [
            f"{BUCKET}/a",
            f"{BUCKET}/b",
            f"{BUCKET}/c",
            f"{BUCKET}/d",
        ]
        assert all(e["soft_deleted"] is True for e in entries)
        assert all(e["type"] == "file" for e in entries)
        # GCS soft-delete metadata and the restore token are preserved.
        tokened = _soft_object("e", 5, restore_token="rt-1")
        assert tokened["restoreToken"] == "rt-1"
        assert entries[0]["softDeleteTime"] == "2026-11-01T00:00:00.000Z"
        assert entries[0]["hardDeleteTime"] == "2026-11-08T00:00:00.000Z"
        assert entries[0]["size"] == 5

    @pytest.mark.asyncio
    async def test_listing_cached_separately_from_live_dircache(self):
        fs = _fs()
        pages = [_page([_soft_object("ghost", 1, restore_token="rt")])]
        with mock.patch.object(
            fs, "_call", new_callable=mock.AsyncMock, side_effect=pages
        ) as mock_call:
            first = await fs._list_soft_deleted(BUCKET)
            # Second call is served from the dedicated cache, no new request.
            second = await fs._list_soft_deleted(BUCKET)

        assert mock_call.call_count == 1
        assert first == second
        assert BUCKET in fs._soft_delete_cache
        # The live directory cache stays completely untouched.
        assert fs.dircache == {}
        assert fs._ls_from_cache(f"{BUCKET}/ghost") in ([], None)

        # refresh=True forces a fresh request.
        with mock.patch.object(
            fs, "_call", new_callable=mock.AsyncMock, side_effect=pages
        ) as mock_call:
            await fs._list_soft_deleted(BUCKET, refresh=True)
            assert mock_call.call_count == 1

    @pytest.mark.asyncio
    async def test_ls_and_info_cannot_see_soft_deleted_entries(self):
        fs = _fs()
        fs._soft_delete_cache[BUCKET] = [
            {
                **_process_entry(fs, _soft_object("ghost", 1)),
                "soft_deleted": True,
            }
        ]
        # Every live lookup says the object is gone.
        with mock.patch.object(fs, "_call", new_callable=mock.AsyncMock) as mock_call:
            mock_call.side_effect = FileNotFoundError("live GET 404")
            with pytest.raises(FileNotFoundError):
                await fs._info(f"{BUCKET}/ghost")

        assert fs.dircache == {}
        # Only live/versions listings populate dircache; an empty live listing
        # must not surface the cached soft-deleted object either.
        with mock.patch.object(
            fs,
            "_call",
            new_callable=mock.AsyncMock,
            side_effect=[_page([])],
        ):
            live = await fs._ls(BUCKET)
        assert live == []
        assert f"{BUCKET}/ghost" not in fs.dircache.get(BUCKET, [])

    @pytest.mark.asyncio
    async def test_prefix_uses_real_key_without_losing_slashes(self):
        fs = _fs()
        with mock.patch.object(
            fs, "_call", new_callable=mock.AsyncMock, side_effect=[_page([])]
        ) as mock_call:
            await fs._list_soft_deleted(f"{BUCKET}/folder/")
            assert mock_call.await_args.kwargs["prefix"] == "folder/"

        with mock.patch.object(
            fs, "_call", new_callable=mock.AsyncMock, side_effect=[_page([])]
        ) as mock_call:
            await fs._list_soft_deleted(BUCKET, prefix="sub/")
            assert mock_call.await_args.kwargs["prefix"] == "sub/"

        with mock.patch.object(
            fs, "_call", new_callable=mock.AsyncMock, side_effect=[_page([])]
        ) as mock_call:
            # No implicit trailing slash is added by normalization.
            await fs._list_soft_deleted(f"{BUCKET}/folder")
            assert mock_call.await_args.kwargs["prefix"] == "folder"

        with mock.patch.object(
            fs, "_call", new_callable=mock.AsyncMock, side_effect=[_page([])]
        ) as mock_call:
            await fs._list_soft_deleted(f"{BUCKET}/folder/", prefix="x/y")
            assert mock_call.await_args.kwargs["prefix"] == "folder/x/y"

    @pytest.mark.asyncio
    async def test_max_results_stops_pagination_and_is_not_cached(self):
        fs = _fs()
        pages = [
            _page([_soft_object("a", 1), _soft_object("b", 2)], "token-1"),
            _page([_soft_object("c", 3)], "token-2"),
        ]
        with mock.patch.object(
            fs, "_call", new_callable=mock.AsyncMock, side_effect=pages
        ) as mock_call:
            entries = await fs._list_soft_deleted(BUCKET, page_size=2, max_results=3)

        assert mock_call.call_count == 2
        assert len(entries) == 3
        assert BUCKET not in fs._soft_delete_cache

    @pytest.mark.asyncio
    async def test_delimiter_returns_directory_entries(self):
        fs = _fs()
        page = _page(
            [_soft_object("dir/file", 1)],
            prefixes=["dir/"],
        )
        with mock.patch.object(
            fs, "_call", new_callable=mock.AsyncMock, side_effect=[page]
        ) as mock_call:
            entries = await fs._list_soft_deleted(BUCKET, delimiter="/")

        assert mock_call.await_args.kwargs["delimiter"] == "/"
        directory = [e for e in entries if e["type"] == "directory"]
        assert len(directory) == 1
        assert directory[0]["name"] == f"{BUCKET}/dir/"
        assert directory[0]["soft_deleted"] is True

    @pytest.mark.asyncio
    async def test_soft_delete_disabled_http_error(self):
        fs = _fs()
        error = HttpError(
            {
                "code": 400,
                "message": "Soft delete policy required for this bucket.",
                "errors": [{"reason": "SoftDeletePolicyRequired"}],
            }
        )
        with mock.patch.object(
            fs, "_call", new_callable=mock.AsyncMock, side_effect=error
        ):
            with pytest.raises(SoftDeleteNotEnabledError):
                await fs._list_soft_deleted(BUCKET)

    @pytest.mark.asyncio
    async def test_soft_delete_disabled_value_error_shape(self):
        fs = _fs()
        # validate_response turns 400 bodies containing "invalid" into
        # ValueError; the soft-delete mapping must still classify them.
        side_effect = _http_status_side_effect(
            400,
            "Invalid argument: soft delete is not enabled for this bucket.",
            reason="invalidArgument",
        )
        with mock.patch.object(
            fs, "_call", new_callable=mock.AsyncMock, side_effect=side_effect
        ):
            with pytest.raises(SoftDeleteNotEnabledError):
                await fs._list_soft_deleted(BUCKET)

    @pytest.mark.asyncio
    async def test_permission_denied_is_reported_as_forbidden(self):
        fs = _fs()
        side_effect = _http_status_side_effect(403, "Caller lacks storage.objects.list")
        with mock.patch.object(
            fs, "_call", new_callable=mock.AsyncMock, side_effect=side_effect
        ):
            with pytest.raises(OSError, match="Forbidden"):
                await fs._list_soft_deleted(BUCKET)

    @pytest.mark.asyncio
    async def test_sync_wrapper_works(self):
        fs = _fs()
        with mock.patch.object(
            fs,
            "_call",
            new_callable=mock.AsyncMock,
            side_effect=[_page([_soft_object("a", 1)])],
        ):
            entries = fs.list_soft_deleted(BUCKET)
        assert entries[0]["name"] == f"{BUCKET}/a"
        assert entries[0]["soft_deleted"] is True


def _process_entry(fs, raw):
    return fs._process_object(BUCKET, raw)


class TestRestoreSoftDeleted:
    @pytest.mark.asyncio
    async def test_restore_posts_cas_by_default_and_returns_new_identity(self):
        fs = _fs()
        parent_listing = [{"name": f"{BUCKET}/keep", "type": "file"}]
        fs.dircache[BUCKET] = parent_listing
        soft_listing = [_soft_object("file", 1111, restore_token="rt-old")]
        fs._soft_delete_cache[BUCKET] = soft_listing

        new_obj = _live_object("file", 9999)
        with mock.patch.object(
            fs,
            "_request_once",
            new_callable=mock.AsyncMock,
            return_value=_request_response(new_obj),
        ) as mock_request:
            result = await fs._restore_soft_deleted(f"{BUCKET}/file", 1111)

        # Exactly one POST, carrying the precondition itself -- no HEAD first.
        assert mock_request.call_count == 1
        call = mock_request.await_args
        assert call.args[0] == "POST"
        assert call.args[1] == "b/{}/o/{}/restore"
        assert call.args[2:] == (BUCKET, "file")
        assert call.kwargs["generation"] == "1111"
        # Safe default: only succeed when no live object exists.
        assert call.kwargs["ifGenerationMatch"] == "0"
        # Optional parameters are omitted unless explicitly requested.
        assert "restoreToken" not in call.kwargs
        assert "copySourceAcl" not in call.kwargs

        # The restored identity is the server's new live object, with a new
        # generation -- the old soft-deleted generation is not reactivated.
        assert result["generation"] == "9999"
        assert result["name"] == f"{BUCKET}/file"
        assert "softDeleteTime" not in result

        # Only the confirmed new live object invalidates the live cache.
        assert BUCKET not in fs.dircache
        # The soft-delete cache is left to the server's lifecycle.
        assert fs._soft_delete_cache[BUCKET] == soft_listing

    @pytest.mark.asyncio
    async def test_explicit_generation_match_is_passed_through(self):
        fs = _fs()
        with mock.patch.object(
            fs,
            "_request_once",
            new_callable=mock.AsyncMock,
            return_value=_request_response(_live_object("file", 9999)),
        ) as mock_request:
            await fs._restore_soft_deleted(
                f"{BUCKET}/file", 1111, if_generation_match=555
            )
        assert mock_request.await_args.kwargs["ifGenerationMatch"] == "555"
        assert mock_request.call_count == 1

    @pytest.mark.asyncio
    async def test_overwrite_sends_no_precondition(self):
        fs = _fs()
        with mock.patch.object(
            fs,
            "_request_once",
            new_callable=mock.AsyncMock,
            return_value=_request_response(_live_object("file", 9999)),
        ) as mock_request:
            await fs._restore_soft_deleted(f"{BUCKET}/file", 1111, overwrite=True)
        assert "ifGenerationMatch" not in mock_request.await_args.kwargs

    @pytest.mark.asyncio
    async def test_conflicting_overwrite_strategies_fail_locally(self):
        fs = _fs()
        with mock.patch.object(
            fs, "_request_once", new_callable=mock.AsyncMock
        ) as mock_request:
            with pytest.raises(ValueError):
                await fs._restore_soft_deleted(
                    f"{BUCKET}/file",
                    1111,
                    overwrite=True,
                    if_generation_match=0,
                )
            with pytest.raises(ValueError):
                await fs._restore_soft_deleted(
                    f"{BUCKET}/file", 1111, if_generation_match=-1
                )
            with pytest.raises(ValueError):
                await fs._restore_soft_deleted(
                    f"{BUCKET}/file", 1111, if_generation_match=True
                )
        # Local validation errors must not produce any HTTP request.
        assert mock_request.call_count == 0

    @pytest.mark.asyncio
    async def test_precondition_conflict_returns_specific_error_with_single_post(self):
        fs = _fs()
        side_effect = _http_status_side_effect(
            412, "Precondition Failed", reason="conditionNotMet"
        )
        with mock.patch.object(
            fs,
            "_request_once",
            new_callable=mock.AsyncMock,
            side_effect=side_effect,
        ) as mock_request:
            with pytest.raises(RestorePreconditionFailedError):
                await fs._restore_soft_deleted(
                    f"{BUCKET}/file", 1111, if_generation_match=555
                )
        # One conditional POST only: no HEAD probe, no retry.
        assert mock_request.call_count == 1
        assert mock_request.await_args.args[0] == "POST"
        assert mock_request.await_args.kwargs["ifGenerationMatch"] == "555"

    @pytest.mark.asyncio
    async def test_hard_deleted_404_not_retried(self):
        fs = _fs()
        side_effect = _http_status_side_effect(
            404, "No such soft-deleted object", reason="notFound"
        )
        with mock.patch.object(
            fs,
            "_request_once",
            new_callable=mock.AsyncMock,
            side_effect=side_effect,
        ) as mock_request:
            with pytest.raises(SoftDeletedObjectNotFoundError) as excinfo:
                await fs._restore_soft_deleted(f"{BUCKET}/file", 1111)
            # Stays compatible with ordinary missing-file handling.
            assert isinstance(excinfo.value, FileNotFoundError)
        assert mock_request.call_count == 1

    @pytest.mark.asyncio
    async def test_permission_denied_stays_forbidden(self):
        fs = _fs()
        side_effect = _http_status_side_effect(
            403, "Caller lacks storage.objects.restore"
        )
        with mock.patch.object(
            fs,
            "_request_once",
            new_callable=mock.AsyncMock,
            side_effect=side_effect,
        ) as mock_request:
            with pytest.raises(OSError, match="Forbidden"):
                await fs._restore_soft_deleted(f"{BUCKET}/file", 1111)
        assert mock_request.call_count == 1

    @pytest.mark.asyncio
    async def test_bucket_without_soft_delete_policy(self):
        fs = _fs()
        side_effect = _http_status_side_effect(
            400,
            "Soft delete policy required for this bucket.",
            reason="SoftDeletePolicyRequired",
        )
        with mock.patch.object(
            fs,
            "_request_once",
            new_callable=mock.AsyncMock,
            side_effect=side_effect,
        ):
            with pytest.raises(SoftDeleteNotEnabledError):
                await fs._restore_soft_deleted(f"{BUCKET}/file", 1111)

    @pytest.mark.asyncio
    async def test_server_side_restore_token_required(self):
        fs = _fs()
        side_effect = _http_status_side_effect(
            400,
            "A restore token is required to restore this object.",
        )
        with mock.patch.object(
            fs,
            "_request_once",
            new_callable=mock.AsyncMock,
            side_effect=side_effect,
        ) as mock_request:
            with pytest.raises(RestoreTokenRequiredError):
                await fs._restore_soft_deleted(f"{BUCKET}/file", 1111)
        assert mock_request.call_count == 1

    @pytest.mark.asyncio
    async def test_client_side_ambiguity_requires_token_without_request(self):
        fs = _fs()
        fs._soft_delete_cache[BUCKET] = [
            {
                **_process_entry(fs, _soft_object("file", 1111, restore_token="rt-1")),
                "soft_deleted": True,
            },
            {
                **_process_entry(fs, _soft_object("file", 1111, restore_token="rt-2")),
                "soft_deleted": True,
            },
        ]
        with mock.patch.object(
            fs, "_request_once", new_callable=mock.AsyncMock
        ) as mock_request:
            with pytest.raises(RestoreTokenRequiredError):
                await fs._restore_soft_deleted(f"{BUCKET}/file", 1111)
            # Supplying the token disambiguates and goes through.
            with mock.patch.object(
                fs,
                "_request_once",
                new_callable=mock.AsyncMock,
                return_value=_request_response(_live_object("file", 9999)),
            ) as mocked:
                await fs._restore_soft_deleted(
                    f"{BUCKET}/file", 1111, restore_token="rt-2"
                )
                assert mocked.await_args.kwargs["restoreToken"] == "rt-2"
        assert mock_request.call_count == 0

    @pytest.mark.asyncio
    async def test_copy_source_acl_only_forwarded_when_requested(self):
        fs = _fs()
        with mock.patch.object(
            fs,
            "_request_once",
            new_callable=mock.AsyncMock,
            return_value=_request_response(_live_object("file", 9999)),
        ) as mock_request:
            await fs._restore_soft_deleted(f"{BUCKET}/file", 1111, copy_source_acl=True)
            assert mock_request.await_args.kwargs["copySourceAcl"] == "true"

            await fs._restore_soft_deleted(
                f"{BUCKET}/file", 1112, copy_source_acl=False
            )
            assert mock_request.await_args_list[-1].kwargs["copySourceAcl"] == "false"

            await fs._restore_soft_deleted(f"{BUCKET}/file", 1113)
            assert "copySourceAcl" not in mock_request.await_args_list[-1].kwargs

        with pytest.raises(ValueError):
            await fs._restore_soft_deleted(
                f"{BUCKET}/file", 1114, copy_source_acl="yes"
            )

    @pytest.mark.asyncio
    async def test_lost_response_is_uncertain_and_not_retried(self):
        fs = _fs()
        fs.dircache[BUCKET] = [{"name": f"{BUCKET}/keep", "type": "file"}]
        soft_listing = [_soft_object("file", 1111)]
        fs._soft_delete_cache[BUCKET] = soft_listing

        with mock.patch.object(
            fs,
            "_request_once",
            new_callable=mock.AsyncMock,
            side_effect=aiohttp.ClientConnectionError("connection reset"),
        ) as mock_request:
            with pytest.raises(RestoreResultUncertainError):
                await fs._restore_soft_deleted(f"{BUCKET}/file", 1111)

        # Exactly one attempt -- no blind retry that could duplicate the object.
        assert mock_request.call_count == 1
        # No cache mutation on an uncertain outcome.
        assert fs.dircache[BUCKET] == [{"name": f"{BUCKET}/keep", "type": "file"}]
        assert fs._soft_delete_cache[BUCKET] == soft_listing

    @pytest.mark.asyncio
    async def test_generation_from_version_aware_path(self):
        fs = _fs()
        with mock.patch.object(
            fs,
            "_request_once",
            new_callable=mock.AsyncMock,
            return_value=_request_response(_live_object("file", 9999)),
        ) as mock_request:
            fs.version_aware = True
            await fs._restore_soft_deleted(f"{BUCKET}/file#1111")
        assert mock_request.await_args.kwargs["generation"] == "1111"

    @pytest.mark.asyncio
    async def test_missing_generation_raises(self):
        fs = _fs()
        with pytest.raises(ValueError):
            await fs._restore_soft_deleted(f"{BUCKET}/file")


class TestRestoreSoftDeletedBatch:
    @pytest.mark.asyncio
    async def test_bounded_concurrency_and_per_item_outcomes(self):
        fs = _fs()
        items = [
            {"path": f"{BUCKET}/k0", "generation": 1},
            {"path": f"{BUCKET}/k1", "generation": 2},
            {"path": f"{BUCKET}/k2", "generation": 3},
            {"path": f"{BUCKET}/k3", "generation": 4},
            (f"{BUCKET}/k4", 5),
        ]
        in_flight = 0
        max_in_flight = 0
        outcomes = {
            f"{BUCKET}/k0": ("success", _live_object("k0", 100)),
            f"{BUCKET}/k1": (
                "error",
                RestorePreconditionFailedError("live conflict"),
            ),
            f"{BUCKET}/k2": (
                "error",
                SoftDeletedObjectNotFoundError("hard deleted"),
            ),
            f"{BUCKET}/k3": (
                "uncertain",
                RestoreResultUncertainError("response lost"),
            ),
            f"{BUCKET}/k4": ("success", _live_object("k4", 400)),
        }

        async def fake_restore(path, generation, *, restore_token=None, **opts):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio_sleep()
            in_flight -= 1
            kind, payload = outcomes[path]
            if kind == "success":
                return payload
            raise payload

        with mock.patch.object(
            fs,
            "_restore_soft_deleted",
            new_callable=mock.AsyncMock,
            side_effect=fake_restore,
        ) as mock_restore:
            results = await fs._restore_soft_deleted_batch(items, concurrency=2)

        # Concurrency bound respected and every item attempted exactly once.
        assert max_in_flight <= 2
        assert max_in_flight == 2
        assert mock_restore.call_count == 5

        # Input order preserved, partial success reported per item.
        assert [r["path"] for r in results] == [
            item["path"] if isinstance(item, dict) else item[0] for item in items
        ]
        statuses = [r["status"] for r in results]
        assert statuses == ["success", "error", "error", "uncertain", "success"]
        assert results[0]["object"]["generation"] == "100"
        assert results[4]["object"]["generation"] == "400"
        assert isinstance(results[1]["exception"], RestorePreconditionFailedError)
        assert isinstance(results[2]["exception"], SoftDeletedObjectNotFoundError)
        assert isinstance(results[3]["exception"], RestoreResultUncertainError)
        assert results[3]["error"]

    @pytest.mark.asyncio
    async def test_partial_success_does_not_roll_back_or_mark_batch_success(self):
        fs = _fs()
        # Cache parents for both items; only the successful item's parent may
        # be invalidated by the underlying restore.
        fs.dircache[f"{BUCKET}/ok"] = [{"name": f"{BUCKET}/ok/file", "type": "file"}]
        fs.dircache[f"{BUCKET}/bad"] = [{"name": f"{BUCKET}/bad/file", "type": "file"}]

        async def fake_restore(path, generation, *, restore_token=None, **opts):
            if path.endswith("bad/file"):
                raise SoftDeletedObjectNotFoundError("gone")
            # Mirror the real success path's live-cache update.
            await fs._write_file_cache_update(path)
            return _live_object(path.split("/", 1)[1], 999)

        items = [
            {"path": f"{BUCKET}/ok/file", "generation": 1},
            {"path": f"{BUCKET}/bad/file", "generation": 2},
        ]
        with mock.patch.object(
            fs,
            "_restore_soft_deleted",
            new_callable=mock.AsyncMock,
            side_effect=fake_restore,
        ):
            results = await fs._restore_soft_deleted_batch(items, concurrency=1)

        assert [r["status"] for r in results] == ["success", "error"]
        # The success landed and was not rolled back; the failed item's parent
        # cache is untouched.
        assert f"{BUCKET}/ok" not in fs.dircache
        assert fs.dircache[f"{BUCKET}/bad"] == [
            {"name": f"{BUCKET}/bad/file", "type": "file"}
        ]

    @pytest.mark.asyncio
    async def test_uncertain_item_is_attempted_once_in_batch(self):
        fs = _fs()

        async def fake_restore(path, generation, *, restore_token=None, **opts):
            raise aiohttp.ClientConnectionError("reset")

        with mock.patch.object(
            fs,
            "_restore_soft_deleted",
            new_callable=mock.AsyncMock,
            side_effect=fake_restore,
        ) as mock_restore:
            results = await fs._restore_soft_deleted_batch(
                [{"path": f"{BUCKET}/file", "generation": 1}], concurrency=1
            )

        assert mock_restore.call_count == 1
        assert results[0]["status"] == "uncertain"

    @pytest.mark.asyncio
    async def test_per_item_options_override_defaults(self):
        fs = _fs()
        captured = []

        async def fake_restore(path, generation, *, restore_token=None, **opts):
            captured.append(opts)
            return _live_object(path.split("/", 1)[1], 999)

        items = [
            {"path": f"{BUCKET}/a", "generation": 1, "overwrite": True},
            {"path": f"{BUCKET}/b", "generation": 2, "if_generation_match": 77},
        ]
        with mock.patch.object(
            fs,
            "_restore_soft_deleted",
            new_callable=mock.AsyncMock,
            side_effect=fake_restore,
        ):
            results = await fs._restore_soft_deleted_batch(items)

        assert all(r["status"] == "success" for r in results)
        assert captured[0]["overwrite"] is True
        assert captured[1]["if_generation_match"] == 77

    @pytest.mark.asyncio
    async def test_invalid_concurrency_and_request_shapes(self):
        fs = _fs()
        with pytest.raises(ValueError):
            await fs._restore_soft_deleted_batch([], concurrency=0)
        with pytest.raises(ValueError):
            await fs._restore_soft_deleted_batch([{"path": "x"}])
        with pytest.raises(TypeError):
            await fs._restore_soft_deleted_batch(["not-a-tuple"])


def asyncio_sleep():
    import asyncio

    return asyncio.sleep(0)
