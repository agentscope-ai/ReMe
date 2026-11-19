"""Dated resource retention, catalog persistence, and proactive configuration."""

import asyncio
import shutil
import threading
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import pytest
from reme.components.application_context import ApplicationContext
from reme.components.file_catalog import LocalFileCatalog
from reme.components.runtime_context import RuntimeContext
from reme.schema import FileNode
from reme.steps.index.cleanup_catalog import CleanupCatalogStep


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    """The same instant falls on different calendar days in Shanghai and Los Angeles."""

    class FixedDatetime(datetime):
        """Return a fixed instant while honoring the requested timezone."""

        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 5, 0, 30, tzinfo=timezone.utc).astimezone(tz)

    monkeypatch.setattr("reme.components.file_catalog.base_file_catalog.datetime", FixedDatetime)


def _catalog(workspace, **kwargs):
    app = ApplicationContext(workspace_dir=str(workspace), **kwargs)
    catalog = LocalFileCatalog(name="resource", app_context=app)
    app.components["file_catalog"] = {"resource": catalog}
    return catalog


def _write(workspace, path):
    target = workspace / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"resource")
    return FileNode(path=path, st_mtime=target.stat().st_mtime)


@pytest.mark.asyncio
async def test_retention_uses_directory_dates_and_persists_only_successful_deletions(tmp_path):
    """At day seven remove the entire dated tree, including uncataloged files and empty folders."""
    catalog = _catalog(tmp_path)
    expired = "resource/2026-10-29/sub/report.pdf"
    retained = [
        "resource/2026-10-30/recent.txt",
        "resource/2026-11-06/future.png",
        "resource/2026-02-30/invalid.txt",
        "resource/20261001/noncanonical.txt",
        "resource/archive/2026-10-01/permanent.txt",
        "resource/loose.txt",
        "archive/2026-10-01/permanent.txt",
        "resource-extra/2026-10-01/other.txt",
    ]
    await catalog.upsert([_write(tmp_path, path) for path in [expired, *retained]])
    _write(tmp_path, "resource/2026-10-29/untracked.bin")
    (tmp_path / "resource/2026-10-01").mkdir()
    _write(tmp_path, "resource/2026-10-02")  # A date-named file is not a directory.

    result = await catalog.cleanup_expired("resource", 7)

    assert result == {
        "deleted_dirs": ["resource/2026-10-01", "resource/2026-10-29"],
        "deleted_nodes": 1,
        "errors": {},
    }
    assert not (tmp_path / "resource/2026-10-29").exists()
    assert all((tmp_path / path).exists() for path in retained)
    assert (tmp_path / "resource/2026-10-02").is_file()
    reloaded = _catalog(tmp_path)
    await reloaded.load()
    assert {node.path for node in await reloaded.get_nodes()} == set(retained)
    assert (await catalog.cleanup_expired("resource", 7))["deleted_dirs"] == []


@pytest.mark.asyncio
async def test_cleanup_uses_configured_timezone_and_resource_directory(tmp_path):
    """The configured resource root and local date determine eligibility."""
    catalog = _catalog(tmp_path, timezone="America/Los_Angeles", resource_dir="assets/tmp")
    catalog.watch_dirs = ["resource_dir"]
    catalog.ttl_days = 7
    node = _write(tmp_path, "assets/tmp/2026-10-29/source.bin")
    await catalog.upsert([node])
    step = CleanupCatalogStep(app_context=catalog.app_context, file_catalog="resource")
    response = await step(RuntimeContext(ttl_days=7))
    assert response.success is True
    assert response.metadata["deleted_dirs"] == []
    catalog.ttl_days = 6
    response = await step(RuntimeContext())
    assert response.metadata["deleted_dirs"] == ["assets/tmp/2026-10-29"]
    assert not (tmp_path / node.path).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("ttl_days", [0, -1, True, 1.5, "7", None])
async def test_cleanup_rejects_invalid_ttl_before_deleting(tmp_path, ttl_days):
    """Invalid retention settings cannot authorize deletions."""
    catalog = _catalog(tmp_path)
    node = _write(tmp_path, "resource/2026-10-01/source.txt")
    with pytest.raises(ValueError, match="ttl_days"):
        await catalog.cleanup_expired("resource", ttl_days)
    assert (tmp_path / node.path).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("directory", ["", ".", "..", "resource/../archive", "~/resource", "/tmp"])
async def test_cleanup_rejects_unsafe_roots(tmp_path, directory):
    """Cleanup accepts only a non-root workspace-relative directory."""
    catalog = _catalog(tmp_path)
    with pytest.raises(ValueError, match="inside the workspace"):
        await catalog.cleanup_expired(directory, 7)


@pytest.mark.asyncio
async def test_cleanup_never_follows_directory_symlinks(tmp_path):
    """Dated symlinks are skipped, inner links are unlinked, and linked roots are refused."""
    catalog = _catalog(tmp_path)
    permanent = _write(tmp_path, "archive/2026-10-01/keep.txt")
    root = tmp_path / "resource"
    root.mkdir()
    (root / "2026-10-01").symlink_to(tmp_path / "archive/2026-10-01", target_is_directory=True)
    (root / "2026-10-02").mkdir()
    (root / "2026-10-02/link").symlink_to(tmp_path / "archive", target_is_directory=True)
    (tmp_path / "linked").symlink_to(root, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinks"):
        await catalog.cleanup_expired("linked", 7)
    result = await catalog.cleanup_expired("resource", 7)
    assert result["deleted_dirs"] == ["resource/2026-10-02"]
    assert (root / "2026-10-01").is_symlink()
    assert (tmp_path / permanent.path).is_file()


@pytest.mark.asyncio
async def test_cleanup_reports_failures_without_pruning_failed_catalog_entries(tmp_path, monkeypatch):
    """A failed directory remains retryable while successful deletes are persisted."""
    catalog = _catalog(tmp_path)
    nodes = [_write(tmp_path, f"resource/{day}/source.txt") for day in ["2026-10-01", "2026-10-02"]]
    await catalog.upsert(nodes)
    original = shutil.rmtree

    @wraps(original)
    def remove(path, **kwargs):
        if Path(path).name == "2026-10-01":
            raise PermissionError("blocked")
        original(path, **kwargs)

    monkeypatch.setattr("reme.components.file_catalog.local_file_catalog.shutil.rmtree", remove)
    catalog.watch_dirs = ["resource_dir"]
    catalog.ttl_days = 7
    step = CleanupCatalogStep(app_context=catalog.app_context, file_catalog="resource")
    response = await step(RuntimeContext())
    assert response.success is False
    assert response.metadata["errors"] == {"resource/2026-10-01": "blocked"}
    assert response.metadata["deleted_dirs"] == ["resource/2026-10-02"]
    reloaded = _catalog(tmp_path)
    await reloaded.load()
    assert [node.path for node in await reloaded.get_nodes()] == [nodes[0].path]


@pytest.mark.asyncio
async def test_cancellation_finishes_deletion_and_catalog_persistence(tmp_path, monkeypatch):
    """Cancellation cannot leave a deletion worker publishing outside the catalog lock."""
    catalog = _catalog(tmp_path)
    await catalog.upsert([_write(tmp_path, "resource/2026-10-01/source.txt")])
    entered, release = threading.Event(), threading.Event()
    original = shutil.rmtree

    @wraps(original)
    def remove(path, **kwargs):
        entered.set()
        assert release.wait(5)
        original(path, **kwargs)

    monkeypatch.setattr("reme.components.file_catalog.local_file_catalog.shutil.rmtree", remove)
    task = asyncio.create_task(catalog.cleanup_expired("resource", 7))
    try:
        async with asyncio.timeout(5):
            while not entered.is_set():
                await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    reloaded = _catalog(tmp_path)
    await reloaded.load()
    assert await reloaded.get_nodes() == []
    assert not (tmp_path / "resource/2026-10-01").exists()


@pytest.mark.asyncio
async def test_cleanup_cannot_follow_a_root_swapped_after_validation(tmp_path, monkeypatch):
    """Replacing resource/ with a link during deletion cannot delete permanent material."""
    catalog = _catalog(tmp_path)
    await catalog.upsert([_write(tmp_path, "resource/2026-10-01/temporary.txt")])
    permanent = _write(tmp_path, "archive/2026-10-01/permanent.txt")
    root = tmp_path / "resource"
    original = shutil.rmtree

    @wraps(original)
    def remove(path, **kwargs):
        root.rename(tmp_path / "moved-resource")
        root.symlink_to(tmp_path / "archive", target_is_directory=True)
        original(path, **kwargs)

    monkeypatch.setattr("reme.components.file_catalog.local_file_catalog.shutil.rmtree", remove)
    await catalog.cleanup_expired("resource", 7)
    assert (tmp_path / permanent.path).is_file()


@pytest.mark.asyncio
async def test_cleanup_retries_catalog_persistence_after_a_write_failure(tmp_path, monkeypatch):
    """Retrying after a failed dump persists removals even though source directories are already gone."""
    catalog = _catalog(tmp_path)
    await catalog.upsert([_write(tmp_path, "resource/2026-10-01/source.txt")])
    await catalog.dump()
    import reme.components.file_catalog.local_file_catalog as module

    original = module.write_jsonl_zst

    def fail(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(module, "write_jsonl_zst", fail)
    with pytest.raises(OSError, match="disk full"):
        await catalog.cleanup_expired("resource", 7)
    monkeypatch.setattr(module, "write_jsonl_zst", original)
    restarted = _catalog(tmp_path)
    await restarted.load()
    await restarted.cleanup_expired("resource", 7)
    reloaded = _catalog(tmp_path)
    await reloaded.load()
    assert await reloaded.get_nodes() == []


@pytest.mark.asyncio
async def test_cleanup_refuses_platforms_without_safe_directory_deletion(tmp_path, monkeypatch):
    """Unsupported platforms fail before deleting source files."""
    catalog = _catalog(tmp_path)
    node = _write(tmp_path, "resource/2026-10-01/source.txt")
    monkeypatch.setattr(shutil.rmtree, "avoids_symlink_attacks", False)
    with pytest.raises(RuntimeError, match="symlink-resistant"):
        await catalog.cleanup_expired("resource", 7)
    assert (tmp_path / node.path).is_file()
