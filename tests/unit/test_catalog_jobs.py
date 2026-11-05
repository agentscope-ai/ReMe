"""Pending resource paths, atomic acknowledgement, and TTL precedence."""

import os
from datetime import datetime, timezone

import pytest

from reme.components.application_context import ApplicationContext
from reme.components.file_catalog import LocalFileCatalog
from reme.components.job.base_job import BaseJob
from reme.config.config_parser import resolve_app_config
from reme.schema import FileNode


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
async def test_pending_is_read_only_and_ack_confirms_only_selected_paths(catalog, tmp_path):
    """Paths repeat until acknowledged; selected checkpoints survive a reload."""
    first = _write(tmp_path, "resource/2026-11-05/a.txt")
    second = _write(tmp_path, "resource/2026-11-05/b.unknown")
    batch = [first.relative_to(tmp_path).as_posix(), second.relative_to(tmp_path).as_posix()]
    assert await catalog.pending() == batch
    assert await catalog.pending() == batch
    assert await catalog.get_nodes() == []
    assert not (tmp_path / "metadata/file_catalog/resource.jsonl.zst").exists()
    assert await catalog.ack(batch[:1]) is True
    assert await catalog.ack(batch[:1] * 2) is True
    assert await catalog.pending() == batch[1:]
    reloaded = LocalFileCatalog(name="resource", app_context=catalog.app_context)
    await reloaded.load()
    assert [node.path for node in await reloaded.get_nodes()] == batch[:1]
    assert first.read_text() == second.read_text() == "1000"


@pytest.mark.asyncio
async def test_pending_limits_sorted_batches_until_acknowledged(catalog, tmp_path):
    """Default and custom caps leave remaining resources available after acknowledgement."""
    paths = [f"resource/2026-11-05/{index:02}.txt" for index in range(13)]
    for path in reversed(paths):
        _write(tmp_path, path)
    assert await catalog.pending() == paths[:10]
    assert await catalog.pending() == paths[:10]
    assert await catalog.pending(limit=3) == paths[:3]
    assert await catalog.ack(paths[:10]) is True
    assert await catalog.pending() == paths[10:]
    assert await catalog.ack(paths[10:]) is True
    assert await catalog.pending() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1, True, 1.5, None, "10"])
async def test_pending_requires_a_positive_integer_limit(catalog, limit):
    """Malformed limits cannot disable the response cap."""
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        await catalog.pending(limit=limit)


@pytest.mark.asyncio
async def test_modified_paths_become_pending_and_deleted_paths_are_omitted(catalog, tmp_path):
    """Acknowledgement records current mtime; missing files are never pending work."""
    path = "resource/2026-11-05/note.txt"
    target = _write(tmp_path, path)
    assert await catalog.ack([path]) is True
    _write(tmp_path, path, 2000)
    assert await catalog.pending() == [path]
    assert await catalog.ack([path]) is True
    assert (await catalog.get_nodes([path]))[0].st_mtime == 2000
    assert await catalog.pending() == []
    target.unlink()
    assert await catalog.pending() == []
    assert await catalog.ack([path]) is False


@pytest.mark.asyncio
async def test_invalid_path_prevents_the_entire_ack_batch(catalog, tmp_path):
    """A valid prefix cannot publish a partial checkpoint when a later path is missing."""
    first = "resource/2026-11-05/a.txt"
    second = "resource/2026-11-05/b.txt"
    _write(tmp_path, first)
    _write(tmp_path, second)
    assert await catalog.ack([first]) is True
    persisted = (tmp_path / "metadata/file_catalog/resource.jsonl.zst").read_bytes()
    _write(tmp_path, first, 2000)
    assert await catalog.ack([first, second, "resource/missing.txt"]) is False
    assert await catalog.get_nodes() == [FileNode(path=first, st_mtime=1000)]
    assert (tmp_path / "metadata/file_catalog/resource.jsonl.zst").read_bytes() == persisted
    assert await catalog.pending() == [first, second]


@pytest.mark.asyncio
async def test_ack_accepts_empty_batches_and_rejects_directories(catalog, tmp_path):
    """Empty acknowledgement is a no-op; directories cannot become checkpoints."""
    assert await catalog.ack([]) is True
    assert not (tmp_path / "metadata/file_catalog/resource.jsonl.zst").exists()
    (tmp_path / "resource/a").mkdir(parents=True)
    assert await catalog.ack(["resource/a"]) is False
    assert await catalog.get_nodes() == []


@pytest.mark.asyncio
async def test_ttl_precedes_pending_ack_and_cleanup(catalog, tmp_path, clock):
    """An in-flight consumer cannot revive an expired catalog entry after cleanup."""
    path = "resource/2026-10-30/a.txt"
    target = _write(tmp_path, path)
    assert await catalog.ack([path]) is True
    _write(tmp_path, path, 2000)
    pending = await catalog.pending()
    clock.instant = datetime(2026, 11, 6, 0, 30, tzinfo=timezone.utc)
    assert target.exists()
    assert await catalog.pending() == []
    assert await catalog.ack(pending) is False
    await catalog.cleanup_expired()
    assert not target.exists()
    assert await catalog.get_nodes() == []
    assert await catalog.ack(pending) is False
    assert await catalog.get_nodes() == []


@pytest.mark.asyncio
async def test_scope_filters_pending_paths_and_refuses_escaping_links(catalog, tmp_path):
    """Scope restrictions apply equally to discovery and acknowledgement."""
    catalog.watch_suffixes = ["txt"]
    _write(tmp_path, "resource/2026-11-05/a.txt")
    _write(tmp_path, "resource/2026-11-05/b.pdf")
    outside = _write(tmp_path, "archive/permanent.txt")
    (tmp_path / "resource/link.txt").symlink_to(outside)
    await catalog.upsert([FileNode(path="archive/old.txt", st_mtime=1), FileNode(path="resource/old.pdf", st_mtime=1)])
    pending = await catalog.pending()
    assert pending == ["resource/2026-11-05/a.txt"]
    for path in [
        "../archive/permanent.txt",
        str(outside),
        "archive/permanent.txt",
        "resource/link.txt",
        "resource/../archive/permanent.txt",
        "resource//2026-11-05/a.txt",
        "~/.secret",
        "C:/a.txt",
        "",
    ]:
        assert await catalog.ack([*pending, path]) is False
    assert {node.path for node in await catalog.get_nodes()} == {"archive/old.txt", "resource/old.pdf"}


@pytest.mark.asyncio
@pytest.mark.parametrize("paths", [None, "resource/a.txt", {}, [None], [1], [["resource/a.txt"]]])
async def test_ack_requires_a_list_of_strings(catalog, paths):
    """Malformed inputs fail before inspecting source files or publishing state."""
    with pytest.raises(ValueError, match="paths must be a list of strings"):
        await catalog.ack(paths)
    assert await catalog.get_nodes() == []


@pytest.mark.asyncio
async def test_failed_ack_write_keeps_baseline_retryable(catalog, tmp_path, monkeypatch):
    """Disk errors cannot publish a success-only in-memory checkpoint."""
    _write(tmp_path, "resource/a.txt")
    batch = await catalog.pending()
    import reme.components.file_catalog.local_file_catalog as module

    def fail(*_args):
        raise OSError("disk full")

    with monkeypatch.context() as patcher:
        patcher.setattr(module, "write_jsonl_zst", fail)
        assert await catalog.ack(batch) is False
    assert await catalog.get_nodes() == []
    assert not (tmp_path / "metadata/file_catalog/resource.jsonl.zst").exists()
    assert await catalog.pending() == batch
    assert await catalog.ack(batch) is True


@pytest.mark.asyncio
async def test_jobs_bind_the_resource_catalog_and_return_paths_or_boolean(
    tmp_path,
    clock,
):  # pylint: disable=unused-argument
    """Public Jobs use the configured resource scope without a caller-selected catalog."""
    config = resolve_app_config(config="proactive", log_config=False, workspace_dir=str(tmp_path))
    app = ApplicationContext(**config)
    resource = LocalFileCatalog(
        name="resource",
        app_context=app,
        **app.app_config.components["file_catalog"]["resource"].model_dump(),
    )
    other = LocalFileCatalog(name="other", app_context=app, watch_dirs=["inbox"])
    app.components["file_catalog"] = {"resource": resource, "other": other}
    path = "resource/2026-11-05/a/metadata.json"
    second_path = "resource/2026-11-05/b/metadata.json"
    _write(tmp_path, path)
    _write(tmp_path, second_path)
    _write(tmp_path, "resource/2026-11-05/a/image.png")
    _write(tmp_path, "inbox/b.txt")
    pending = BaseJob(name="resource_pending", app_context=app, **app.app_config.jobs["resource_pending"].model_dump())
    ack = BaseJob(name="resource_ack", app_context=app, **app.app_config.jobs["resource_ack"].model_dump())
    assert pending.parameters["properties"]["limit"]["default"] == 10
    assert pending.parameters["properties"]["limit"]["minimum"] == 1
    assert set(ack.parameters["properties"]) == {"paths"}
    assert ack.parameters["required"] == ["paths"]
    await pending.start()
    await ack.start()
    try:
        response = await pending()
        assert response.answer == [path, second_path]
        assert response.success is True
        assert (await pending(file_catalog="other", limit=1)).answer == [path]
        assert (await pending(limit=0)).success is False
        response = await ack(file_catalog="other", paths=[path, second_path])
        assert response.success is True
        assert response.answer is True
        assert (await pending()).answer == []
        assert len(await resource.get_nodes()) == 2
        assert await other.get_nodes() == []
        response = await ack(paths=["inbox/b.txt"])
        assert response.success is False
        assert response.answer is False
        assert (await ack()).success is False
    finally:
        await ack.close()
        await pending.close()
