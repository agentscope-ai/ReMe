"""Explicit catalog confirmation, version checks, and TTL precedence."""

import os
from datetime import datetime, timezone

import pytest

from reme.components.application_context import ApplicationContext
from reme.components.file_catalog import LocalFileCatalog
from reme.components.job.base_job import BaseJob
from reme.config.config_parser import resolve_app_config
from reme.schema import CatalogChange, FileNode


@pytest.fixture(name="clock")
def _clock(monkeypatch):
    """Control the catalog's calendar without sleeping."""

    class Clock(datetime):
        """A mutable test instant with real timezone conversion."""

        instant = datetime(2026, 11, 5, 0, 30, tzinfo=timezone.utc)

        @classmethod
        def now(cls, tz=None):
            return cls.instant.astimezone(tz)

    monkeypatch.setattr("reme.components.file_catalog.base_file_catalog.datetime", Clock)
    return Clock


@pytest.fixture(name="catalog")
def _catalog(tmp_path, clock):  # pylint: disable=unused-argument
    """One scoped catalog, matching proactive defaults."""
    app = ApplicationContext(workspace_dir=str(tmp_path))
    result = LocalFileCatalog(name="resource", app_context=app, watch_dirs=["resource_dir"], ttl_days=7)
    app.components["file_catalog"] = {"resource": result}
    return result


def _write(workspace, path, mtime=1000):
    target = workspace / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(mtime), encoding="utf-8")
    os.utime(target, (mtime, mtime))
    return target


@pytest.mark.asyncio
async def test_diff_is_read_only_and_commit_confirms_only_selected_changes(catalog, tmp_path):
    """Changes repeat until confirmed; commit can acknowledge a successful subset."""
    first = _write(tmp_path, "resource/2026-11-05/a.txt")
    second = _write(tmp_path, "resource/2026-11-05/b.unknown")
    batch = await catalog.diff()
    assert batch["counts"] == {"added": 2, "modified": 0, "deleted": 0}
    assert await catalog.diff() == batch
    assert await catalog.get_nodes() == []
    assert not (tmp_path / "metadata/file_catalog/resource.jsonl.zst").exists()
    chosen = batch["changes"][:1]
    result = await catalog.commit(chosen)
    assert result == {"committed": ["resource/2026-11-05/a.txt"], "skipped": []}
    assert await catalog.commit(chosen) == result
    assert [item["path"] for item in (await catalog.diff())["changes"]] == [second.relative_to(tmp_path).as_posix()]
    reloaded = LocalFileCatalog(name="resource", app_context=catalog.app_context)
    await reloaded.load()
    assert [node.path for node in await reloaded.get_nodes()] == [first.relative_to(tmp_path).as_posix()]


@pytest.mark.asyncio
async def test_modified_and_deleted_versions_round_trip(catalog, tmp_path):
    """Both modification and deletion require the exact baseline returned by diff."""
    path = "resource/2026-11-05/note.txt"
    target = _write(tmp_path, path)
    await catalog.commit((await catalog.diff())["changes"])
    _write(tmp_path, path, 2000)
    changes = (await catalog.diff())["changes"]
    assert changes == [{"path": path, "change": "modified", "st_mtime": 2000, "previous_mtime": 1000}]
    assert (await catalog.commit(changes))["skipped"] == []
    target.unlink()
    changes = (await catalog.diff())["changes"]
    assert changes == [{"path": path, "change": "deleted", "st_mtime": None, "previous_mtime": 2000}]
    assert (await catalog.commit(changes))["skipped"] == []
    assert (await catalog.diff())["changes"] == []


@pytest.mark.asyncio
async def test_commit_skips_changed_source_and_conflicting_baseline(catalog, tmp_path):
    """A stale acknowledgement cannot consume a newer source or overwrite another commit."""
    first = "resource/2026-11-05/a.txt"
    second = "resource/2026-11-05/b.txt"
    _write(tmp_path, first)
    _write(tmp_path, second)
    changes = (await catalog.diff())["changes"]
    _write(tmp_path, first, 2000)
    result = await catalog.commit(changes)
    assert result == {"committed": [second], "skipped": [{"path": first, "reason": "source_changed"}]}
    changes = (await catalog.diff())["changes"]
    await catalog.upsert([FileNode(path=first, st_mtime=3000)])
    result = await catalog.commit(changes)
    assert result["skipped"] == [{"path": first, "reason": "catalog_changed"}]
    assert (await catalog.get_nodes([first]))[0].st_mtime == 3000


@pytest.mark.asyncio
async def test_delete_commit_cannot_confirm_a_recreated_file(catalog, tmp_path):
    """An old deletion acknowledgement cannot remove a newly recreated file's state."""
    path = "resource/2026-11-05/a.txt"
    target = _write(tmp_path, path)
    await catalog.commit((await catalog.diff())["changes"])
    target.unlink()
    deleted = (await catalog.diff())["changes"]
    _write(tmp_path, path, 2000)
    assert (await catalog.commit(deleted))["skipped"] == [{"path": path, "reason": "source_changed"}]
    assert len(await catalog.get_nodes()) == 1


@pytest.mark.asyncio
async def test_ttl_precedes_diff_commit_and_cleanup(catalog, tmp_path, clock):
    """An in-flight consumer cannot revive an expired catalog entry after cleanup."""
    path = "resource/2026-10-30/a.txt"
    target = _write(tmp_path, path)
    await catalog.commit((await catalog.diff())["changes"])
    _write(tmp_path, path, 2000)
    pending = (await catalog.diff())["changes"]
    clock.instant = datetime(2026, 11, 6, 0, 30, tzinfo=timezone.utc)
    assert target.exists()
    assert (await catalog.diff())["changes"] == []
    assert (await catalog.commit(pending))["skipped"] == [{"path": path, "reason": "expired"}]
    await catalog.cleanup_expired()
    assert not target.exists()
    assert await catalog.get_nodes() == []
    assert (await catalog.commit(pending))["skipped"] == [{"path": path, "reason": "expired"}]
    assert await catalog.get_nodes() == []


@pytest.mark.asyncio
async def test_scope_filters_both_snapshots_and_refuses_escaping_links(catalog, tmp_path):
    """Narrowing scope must not manufacture deletions for unrelated catalog entries."""
    catalog.watch_suffixes = ["txt"]
    _write(tmp_path, "resource/2026-11-05/a.txt")
    _write(tmp_path, "resource/2026-11-05/b.pdf")
    outside = _write(tmp_path, "archive/permanent.txt")
    (tmp_path / "resource/link.txt").symlink_to(outside)
    await catalog.upsert([FileNode(path="archive/old.txt", st_mtime=1), FileNode(path="resource/old.pdf", st_mtime=1)])
    changes = (await catalog.diff())["changes"]
    assert [item["path"] for item in changes] == ["resource/2026-11-05/a.txt"]
    for path in ["../archive/permanent.txt", str(outside), "archive/permanent.txt", "resource/link.txt"]:
        result = await catalog.commit([dict(changes[0], path=path)])
        assert result["skipped"] == [{"path": path, "reason": "out_of_scope"}]


@pytest.mark.asyncio
async def test_bad_batches_are_validated_before_any_commit(catalog, tmp_path):
    """Malformed or duplicate records fail before publishing any checkpoint."""
    _write(tmp_path, "resource/a.txt")
    changes = (await catalog.diff())["changes"]
    with pytest.raises(ValueError, match="previous_mtime"):
        await catalog.commit([*changes, {"path": "resource/b.txt", "change": "added", "st_mtime": 1}])
    with pytest.raises(ValueError, match="duplicate"):
        await catalog.commit(changes * 2)
    assert await catalog.get_nodes() == []


@pytest.mark.asyncio
async def test_failed_commit_write_keeps_baseline_retryable(catalog, tmp_path, monkeypatch):
    """Disk errors cannot publish a success-only in-memory checkpoint."""
    _write(tmp_path, "resource/a.txt")
    batch = await catalog.diff()
    import reme.components.file_catalog.local_file_catalog as module

    def fail(*_args):
        raise OSError("disk full")

    with monkeypatch.context() as patcher:
        patcher.setattr(module, "write_jsonl_zst", fail)
        with pytest.raises(OSError, match="disk full"):
            await catalog.commit(batch["changes"])
    assert await catalog.get_nodes() == []
    assert await catalog.diff() == batch
    assert (await catalog.commit(batch["changes"]))["skipped"] == []


@pytest.mark.asyncio
async def test_jobs_resolve_call_time_catalog_and_publish_changes_in_answer(
    tmp_path,
    clock,
):  # pylint: disable=unused-argument
    """The real Job chain supports a configured default and a per-call catalog selector."""
    config = resolve_app_config(config="proactive", log_config=False, workspace_dir=str(tmp_path))
    app = ApplicationContext(**config)
    resource = LocalFileCatalog(name="resource", app_context=app, watch_dirs=["resource_dir"], ttl_days=7)
    other = LocalFileCatalog(name="other", app_context=app, watch_dirs=["inbox"])
    app.components["file_catalog"] = {"resource": resource, "other": other}
    _write(tmp_path, "resource/a.txt")
    _write(tmp_path, "inbox/b.txt")
    diff = BaseJob(name="catalog_diff", app_context=app, **app.app_config.jobs["catalog_diff"].model_dump())
    commit = BaseJob(name="catalog_commit", app_context=app, **app.app_config.jobs["catalog_commit"].model_dump())
    await diff.start()
    await commit.start()
    try:
        default = await diff()
        assert default.answer["changes"][0]["path"] == "resource/a.txt"
        selected = await diff(file_catalog="other")
        assert selected.answer["changes"][0]["path"] == "inbox/b.txt"
        response = await commit(file_catalog="other", changes=selected.answer["changes"])
        assert response.success is True
        assert response.answer["committed"] == ["inbox/b.txt"]
        assert await resource.get_nodes() == []
        assert len(await other.get_nodes()) == 1
    finally:
        await commit.close()
        await diff.close()


@pytest.mark.parametrize("field", ["st_mtime", "previous_mtime"])
def test_catalog_versions_reject_nonfinite_values(field):
    """Nonfinite versions cannot bypass snapshot equality checks."""
    payload = {"path": "resource/a.txt", "change": "modified", "st_mtime": 1, "previous_mtime": 0}
    with pytest.raises(ValueError):
        CatalogChange.model_validate({**payload, field: float("nan")})
