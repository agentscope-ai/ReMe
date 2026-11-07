"""Mtime retention without catalogs, using isolated files and symlink-safe deletion."""

# pylint: disable=missing-function-docstring,redefined-outer-name

import asyncio
import os
import threading

import pytest

from reme.components.application_context import ApplicationContext
from reme.config import resolve_app_config
from reme.steps.proactive import CleanupResourcesStep

NOW = 1800000000.0


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setattr("reme.steps.proactive.cleanup_resources.time.time", lambda: NOW)


def write(workspace, name, age=8):
    path = workspace / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"content")
    os.utime(path, (NOW - age * 86400, NOW - age * 86400))
    return path


def step(workspace, **config):
    return CleanupResourcesStep(app_context=ApplicationContext(workspace_dir=str(workspace), **config))


@pytest.mark.asyncio
async def test_mtime_retention_removes_all_file_types_and_empty_directories(tmp_path):
    old = [
        write(tmp_path, "resource/screen_capture/old.png"),
        write(tmp_path, "resource/news/anything.json", age=7),
        write(tmp_path, "resource/future-directory/2030-01-01.bin"),
        write(tmp_path, "resource/root.txt"),
    ]
    fresh = write(tmp_path, "resource/2020-01-01/recent.bin", age=6.99)
    outside = write(tmp_path, "daily/personal/MEMORY.md", age=100)
    (tmp_path / "resource/empty/nested").mkdir(parents=True)
    result = await step(tmp_path)()
    assert result.success
    assert set(result.answer["deleted_files"]) == {p.relative_to(tmp_path).as_posix() for p in old}
    assert result.answer["freed_bytes"] == 7 * len(old)
    assert result.answer["errors"] == {}
    assert all(not path.exists() for path in old)
    assert fresh.is_file() and outside.is_file()
    assert not (tmp_path / "resource/empty").exists()
    assert (tmp_path / "resource").is_dir()


@pytest.mark.asyncio
async def test_configured_root_and_call_time_ttl_override(tmp_path):
    old = write(tmp_path, "assets/tmp/report.txt", age=2)
    other = write(tmp_path, "resource/keep.txt")
    cleanup = step(tmp_path, resource_dir="assets/tmp", timezone="America/Los_Angeles")
    assert (await cleanup()).answer["deleted_files"] == []
    assert (await cleanup(ttl_days=1)).answer["deleted_files"] == ["assets/tmp/report.txt"]
    assert not old.exists() and other.is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize("ttl", [0, -1, True, 1.5, "7", None])
async def test_invalid_ttl_does_not_delete(tmp_path, ttl):
    path = write(tmp_path, "resource/old.bin")
    with pytest.raises(ValueError, match="ttl_days"):
        await step(tmp_path)(ttl_days=ttl)
    assert path.is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize("root", ["", ".", "..", "resource/../archive", "~/resource", "/tmp"])
async def test_unsafe_roots_are_rejected(tmp_path, root):
    path = write(tmp_path, "resource/old.bin")
    with pytest.raises(ValueError):
        await step(tmp_path, resource_dir=root)()
    assert path.is_file()


@pytest.mark.asyncio
async def test_symlinked_roots_and_nested_links_are_never_followed(tmp_path):
    permanent = write(tmp_path, "archive/keep.txt")
    root = tmp_path / "resource"
    root.mkdir()
    (root / "linked-dir").symlink_to(permanent.parent, target_is_directory=True)
    (root / "linked-file").symlink_to(permanent)
    (tmp_path / "linked-root").symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        await step(tmp_path, resource_dir="linked-root")()
    result = await step(tmp_path)()
    assert result.success and result.answer["deleted_files"] == []
    assert permanent.is_file()
    assert (root / "linked-dir").is_symlink() and (root / "linked-file").is_symlink()


@pytest.mark.asyncio
async def test_failed_file_is_reported_while_other_deletions_finish(tmp_path, monkeypatch):
    blocked = write(tmp_path, "resource/blocked.bin")
    deleted = write(tmp_path, "resource/deleted.bin")
    unlink = os.unlink

    def remove(name, *args, **kwargs):
        if name == "blocked.bin":
            raise PermissionError("blocked")
        return unlink(name, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", remove)
    result = await step(tmp_path)()
    assert not result.success
    assert result.answer["errors"] == {"resource/blocked.bin": "blocked"}
    assert result.answer["deleted_files"] == ["resource/deleted.bin"]
    assert blocked.is_file() and not deleted.exists()


@pytest.mark.asyncio
async def test_cancellation_waits_for_deletion_worker(tmp_path, monkeypatch):
    path = write(tmp_path, "resource/old.bin")
    entered, release = threading.Event(), threading.Event()
    unlink = os.unlink

    def remove(name, *args, **kwargs):
        if name == "old.bin":
            entered.set()
            assert release.wait(5)
        return unlink(name, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", remove)
    task = asyncio.create_task(step(tmp_path)())
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
    assert not path.exists()


@pytest.mark.asyncio
async def test_root_replacement_cannot_redirect_deletion(tmp_path, monkeypatch):
    write(tmp_path, "resource/old.bin")
    permanent = write(tmp_path, "archive/old.bin")
    root = tmp_path / "resource"
    unlink = os.unlink

    def remove(name, *args, **kwargs):
        if name == "old.bin" and not root.is_symlink():
            root.rename(tmp_path / "moved-resource")
            root.symlink_to(permanent.parent, target_is_directory=True)
        return unlink(name, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", remove)
    result = await step(tmp_path)()
    assert result.success
    assert permanent.read_bytes() == b"content"
    assert not (tmp_path / "moved-resource/old.bin").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [[], ["daily"], ["../outside"]])
async def test_allowed_paths_fail_closed(tmp_path, scope):
    path = write(tmp_path, "resource/old.bin")
    with pytest.raises(PermissionError):
        await step(tmp_path)(_allowed_paths=scope)
    assert path.is_file()


@pytest.mark.asyncio
async def test_missing_root_is_a_successful_noop(tmp_path):
    result = await step(tmp_path)()
    assert result.success
    assert result.answer == {"deleted_files": [], "deleted_dirs": [], "freed_bytes": 0, "errors": {}}
    assert not (tmp_path / "resource").exists()


@pytest.mark.asyncio
async def test_unsupported_platform_fails_before_deletion(tmp_path, monkeypatch):
    path = write(tmp_path, "resource/old.bin")
    monkeypatch.delattr(os, "fwalk")
    with pytest.raises(RuntimeError, match="symlink-safe"):
        await step(tmp_path)()
    assert path.is_file()


def test_cleanup_is_daily_and_does_not_depend_on_a_catalog():
    config = resolve_app_config(config="proactive", log_config=False)
    job = config["jobs"]["resource_cleanup_cron"]
    assert job["backend"] == "cron" and job["cron"] == "0 3 * * *"
    assert job["steps"] == [{"backend": "cleanup_resources_step", "ttl_days": 7}]
    assert "file_catalog" not in config["components"]
