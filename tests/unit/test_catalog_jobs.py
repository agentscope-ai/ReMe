"""Resource changes, atomic processing confirmation, and TTL precedence."""

import os
from datetime import datetime, timezone

import pytest

from reme.components.application_context import ApplicationContext
from reme.components.file_catalog import LocalFileCatalog
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
async def test_changes_is_read_only_and_processed_confirms_only_selected_paths(catalog, tmp_path):
    """Paths repeat until acknowledged; selected checkpoints survive a reload."""
    first = _write(tmp_path, "resource/2026-11-05/a.txt")
    second = _write(tmp_path, "resource/2026-11-05/b.unknown")
    batch = [first.relative_to(tmp_path).as_posix(), second.relative_to(tmp_path).as_posix()]
    assert await catalog.changes() == batch
    assert await catalog.changes() == batch
    assert await catalog.get_nodes() == []
    assert not (tmp_path / "metadata/file_catalog/resource.jsonl.zst").exists()
    assert await catalog.processed(batch[:1]) is True
    assert await catalog.processed(batch[:1] * 2) is True
    assert await catalog.changes() == batch[1:]
    reloaded = LocalFileCatalog(name="resource", app_context=catalog.app_context)
    await reloaded.load()
    assert [node.path for node in await reloaded.get_nodes()] == batch[:1]
    assert first.read_text() == second.read_text() == "1000"


@pytest.mark.asyncio
async def test_changes_limits_sorted_batches_until_acknowledged(catalog, tmp_path):
    """Default and custom caps leave remaining resources available after acknowledgement."""
    paths = [f"resource/2026-11-05/{index:02}.txt" for index in range(23)]
    for path in reversed(paths):
        _write(tmp_path, path)
    assert await catalog.changes() == paths[:20]
    assert await catalog.changes() == paths[:20]
    assert await catalog.changes(limit=3) == paths[:3]
    assert await catalog.processed(paths[:20]) is True
    assert await catalog.changes() == paths[20:]
    assert await catalog.processed(paths[20:]) is True
    assert await catalog.changes() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1, True, 1.5, None, "10"])
async def test_changes_requires_a_positive_integer_limit(catalog, limit):
    """Malformed limits cannot disable the response cap."""
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        await catalog.changes(limit=limit)


@pytest.mark.asyncio
async def test_modified_paths_reappear_in_changes_and_deleted_paths_are_omitted(catalog, tmp_path):
    """Processing confirmation records current mtime; missing files are omitted from changes."""
    path = "resource/2026-11-05/note.txt"
    target = _write(tmp_path, path)
    assert await catalog.processed([path]) is True
    _write(tmp_path, path, 2000)
    assert await catalog.changes() == [path]
    assert await catalog.processed([path]) is True
    assert (await catalog.get_nodes([path]))[0].st_mtime == 2000
    assert await catalog.changes() == []
    target.unlink()
    assert await catalog.changes() == []
    assert await catalog.processed([path]) is False


@pytest.mark.asyncio
async def test_invalid_path_prevents_the_entire_processed_batch(catalog, tmp_path):
    """A valid prefix cannot publish a partial checkpoint when a later path is missing."""
    first = "resource/2026-11-05/a.txt"
    second = "resource/2026-11-05/b.txt"
    _write(tmp_path, first)
    _write(tmp_path, second)
    assert await catalog.processed([first]) is True
    persisted = (tmp_path / "metadata/file_catalog/resource.jsonl.zst").read_bytes()
    _write(tmp_path, first, 2000)
    assert await catalog.processed([first, second, "resource/missing.txt"]) is False
    assert await catalog.get_nodes() == [FileNode(path=first, st_mtime=1000)]
    assert (tmp_path / "metadata/file_catalog/resource.jsonl.zst").read_bytes() == persisted
    assert await catalog.changes() == [first, second]


@pytest.mark.asyncio
async def test_processed_accepts_empty_batches_and_rejects_directories(catalog, tmp_path):
    """Empty acknowledgement is a no-op; directories cannot become checkpoints."""
    assert await catalog.processed([]) is True
    assert not (tmp_path / "metadata/file_catalog/resource.jsonl.zst").exists()
    (tmp_path / "resource/a").mkdir(parents=True)
    assert await catalog.processed(["resource/a"]) is False
    assert await catalog.get_nodes() == []


@pytest.mark.asyncio
async def test_ttl_precedes_changes_processed_and_cleanup(catalog, tmp_path, clock):
    """An in-flight consumer cannot revive an expired catalog entry after cleanup."""
    path = "resource/2026-10-30/a.txt"
    target = _write(tmp_path, path)
    assert await catalog.processed([path]) is True
    _write(tmp_path, path, 2000)
    changes = await catalog.changes()
    clock.instant = datetime(2026, 11, 6, 0, 30, tzinfo=timezone.utc)
    assert target.exists()
    assert await catalog.changes() == []
    assert await catalog.processed(changes) is False
    await catalog.cleanup_expired()
    assert not target.exists()
    assert await catalog.get_nodes() == []
    assert await catalog.processed(changes) is False
    assert await catalog.get_nodes() == []


@pytest.mark.asyncio
async def test_scope_filters_changes_paths_and_refuses_escaping_links(catalog, tmp_path):
    """Scope restrictions apply equally to discovery and acknowledgement."""
    catalog.watch_suffixes = ["txt"]
    _write(tmp_path, "resource/2026-11-05/a.txt")
    _write(tmp_path, "resource/2026-11-05/b.pdf")
    outside = _write(tmp_path, "archive/permanent.txt")
    (tmp_path / "resource/link.txt").symlink_to(outside)
    await catalog.upsert([FileNode(path="archive/old.txt", st_mtime=1), FileNode(path="resource/old.pdf", st_mtime=1)])
    changes = await catalog.changes()
    assert changes == ["resource/2026-11-05/a.txt"]
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
        assert await catalog.processed([*changes, path]) is False
    assert {node.path for node in await catalog.get_nodes()} == {"archive/old.txt", "resource/old.pdf"}


@pytest.mark.asyncio
@pytest.mark.parametrize("paths", [None, "resource/a.txt", {}, [None], [1], [["resource/a.txt"]]])
async def test_processed_requires_a_list_of_strings(catalog, paths):
    """Malformed inputs fail before inspecting source files or publishing state."""
    with pytest.raises(ValueError, match="paths must be a list of strings"):
        await catalog.processed(paths)
    assert await catalog.get_nodes() == []


@pytest.mark.asyncio
async def test_failed_processed_write_keeps_baseline_retryable(catalog, tmp_path, monkeypatch):
    """Disk errors cannot publish a success-only in-memory checkpoint."""
    _write(tmp_path, "resource/a.txt")
    batch = await catalog.changes()
    import reme.components.file_catalog.local_file_catalog as module

    def fail(*_args):
        raise OSError("disk full")

    with monkeypatch.context() as patcher:
        patcher.setattr(module, "write_jsonl_zst", fail)
        assert await catalog.processed(batch) is False
    assert await catalog.get_nodes() == []
    assert not (tmp_path / "metadata/file_catalog/resource.jsonl.zst").exists()
    assert await catalog.changes() == batch
    assert await catalog.processed(batch) is True
