"""Real watcher isolation and deterministic shutdown at the Step boundary."""

import asyncio
import threading
from contextlib import aclosing
from unittest.mock import AsyncMock

import anyio
import pytest
from watchfiles import Change

from reme.components.application_context import ApplicationContext
from reme.components.file_catalog import LocalFileCatalog
from reme.components.runtime_context import RuntimeContext
from reme.schema import FileNode
from reme.steps.base_step import BaseStep
from reme.steps.index import _threaded_watch
from reme.steps.index._threaded_watch import watch_in_thread
from reme.steps.index.watch_changes import WatchChangesStep


async def _until(predicate, timeout=5):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


def _workers():
    return {t for t in threading.enumerate() if t.name.startswith("reme-watch")}


def _step(workspace, stop, mode="watch"):
    (workspace / "daily").mkdir(parents=True)
    context = ApplicationContext(workspace_dir=str(workspace), daily_dir="daily")
    step = WatchChangesStep(
        app_context=context,
        watch_mode=mode,
        force_polling=False,
        step=10,
        debounce=50,
    )
    step.context = RuntimeContext(stop_event=stop, watch_dirs=["daily_dir"], watch_suffixes=["md"])
    step.dispatch_steps = AsyncMock()
    return step


def test_watch_mode_defaults_and_validation():
    """Existing callers retain awatch; unknown modes fail explicitly."""
    assert WatchChangesStep().watch_mode == "awatch"
    for value in ("invalid", "", None, True):
        with pytest.raises(ValueError, match="watch_mode"):
            WatchChangesStep(watch_mode=value)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["watch", "awatch"])
async def test_watcher_reconciles_startup_gap_on_first_idle_batch(tmp_path, monkeypatch, mode):
    """Both watchers reconcile once even when no subsequent change arrives."""

    async def idle_batches(*_paths, **kwargs):
        assert kwargs["yield_on_timeout"] is True
        yield set()
        yield set()

    monkeypatch.setattr("reme.steps.index.watch_changes.awatch", idle_batches)
    monkeypatch.setattr("reme.steps.index.watch_changes.watch_in_thread", idle_batches)
    step = _step(tmp_path, asyncio.Event(), mode=mode)
    step.monitor_type = "file_store"
    step.dispatch_step_specs = ["update_index_step"]

    await step.execute()

    step.dispatch_steps.assert_awaited_once_with(
        [
            {
                "backend": "init_changes_step",
                "monitor_type": "file_store",
                "monitor_name": "default",
                "recursive": True,
            },
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["watch", "awatch"])
async def test_watcher_merges_startup_scan_and_first_change_batch(tmp_path, monkeypatch, mode):
    """Overlapping events and missed additions/deletions dispatch once together."""
    step = _step(tmp_path, asyncio.Event(), mode=mode)
    catalog = LocalFileCatalog(app_context=step.app_context)
    step.app_context.components["file_catalog"] = {"default": catalog}
    await catalog.upsert([FileNode(path="daily/deleted.md", st_mtime=1)])
    step.monitor_type = "file_catalog"
    step.dispatch_step_specs = ["update_catalog_step"]
    changed = tmp_path / "daily/changed.md"
    missed = tmp_path / "daily/missed.md"

    async def first_batch(*_paths, **_kwargs):
        changed.write_text("new", encoding="utf-8")
        missed.write_text("missed", encoding="utf-8")
        yield {(Change.added, str(changed))}
        yield set()

    batches = []

    async def dispatch(specs, **kwargs):
        if specs[0] == "update_catalog_step":
            batches.append(kwargs["changes"])
            return []
        return await BaseStep.dispatch_steps(step, specs, **kwargs)

    monkeypatch.setattr("reme.steps.index.watch_changes.awatch", first_batch)
    monkeypatch.setattr("reme.steps.index.watch_changes.watch_in_thread", first_batch)
    step.dispatch_steps = dispatch
    await step.execute()

    assert len(batches) == 1
    assert sorted((item["path"], item["change"]) for item in batches[0]) == sorted(
        [
            (str(changed), "added"),
            (str(missed), "added"),
            (str(tmp_path / "daily/deleted.md"), "deleted"),
        ],
    )


@pytest.mark.asyncio
async def test_default_mode_uses_awatch_and_closes_on_dispatch_failure(tmp_path, monkeypatch):
    """An exception in downstream processing closes the default generator."""
    closed = []

    async def fake_awatch(*_paths, **_kwargs):
        try:
            yield {(Change.added, str(tmp_path / "daily/note.md"))}
        finally:
            closed.append(True)

    monkeypatch.setattr("reme.steps.index.watch_changes.awatch", fake_awatch)
    step = _step(tmp_path, asyncio.Event(), mode="awatch")
    step.dispatch_steps.side_effect = RuntimeError("dispatch failed")
    with pytest.raises(RuntimeError, match="dispatch failed"):
        await step.execute()
    assert closed == [True]


@pytest.mark.asyncio
async def test_41_real_watchers_deliver_changes_without_borrowing_request_threads(tmp_path):
    """Every watcher starts and dispatches beyond AnyIO's default capacity."""
    before = _workers()
    limiter = anyio.to_thread.current_default_thread_limiter()
    borrowed_before = limiter.borrowed_tokens
    stop = asyncio.Event()
    steps = [_step(tmp_path / str(i), stop) for i in range(41)]
    tasks = [asyncio.create_task(step.execute()) for step in steps]
    try:
        await _until(lambda: len(_workers() - before) == 41)
        # Repeated writes also cover watchers whose initial scan is still starting.
        async with asyncio.timeout(5):
            while not all(step.dispatch_steps.await_count for step in steps):
                for i in range(41):
                    path = tmp_path / str(i) / "daily/note.md"
                    path.write_text(path.read_text() + "x" if path.exists() else "x", encoding="utf-8")
                await asyncio.sleep(0.05)
        assert limiter.borrowed_tokens == borrowed_before
        async with asyncio.timeout(1):
            assert await anyio.to_thread.run_sync(lambda: "request served") == "request served"
        for step in steps:
            changes = step.dispatch_steps.await_args.kwargs["changes"]
            assert changes and all(item["path"].endswith("note.md") for item in changes)
    finally:
        stop.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
    assert _workers() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_type", [asyncio.Event, threading.Event])
async def test_idle_stop_releases_private_worker(tmp_path, monkeypatch, stop_type):
    """Both supported background-job stop events terminate an idle wait."""
    before = _workers()
    entered = threading.Event()

    def idle_watch(*_paths, stop_event, **_kwargs):
        entered.set()
        stop_event.wait()
        yield from ()

    monkeypatch.setattr(_threaded_watch, "watch", idle_watch)
    stop = stop_type()
    step = _step(tmp_path, stop)
    task = asyncio.create_task(step.execute())
    await _until(entered.is_set)
    stop.set()
    await asyncio.wait_for(task, timeout=1)
    assert _workers() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("force_polling", [False, True])
async def test_real_idle_watcher_stops_with_low_power_poll_interval(tmp_path, force_polling):
    """Stopping must not wait for the five-second polling interval."""
    before = _workers()
    stop = asyncio.Event()
    step = _step(tmp_path, stop)
    step.force_polling = force_polling
    task = asyncio.create_task(step.execute())
    try:
        await _until(lambda: bool(_workers() - before))
        await asyncio.sleep(0.05)
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=2)
    assert _workers() == before


@pytest.mark.asyncio
async def test_cancellation_waits_for_initial_scan_even_when_cancelled_twice(tmp_path, monkeypatch):
    """Cancellation cannot detach a still-running native watcher."""
    before = _workers()
    entered = threading.Event()
    stopping = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    def slow_watch(*_paths, stop_event, **_kwargs):
        try:
            entered.set()
            stop_event.wait()
            stopping.set()
            assert release.wait(5)
            yield from ()
        finally:
            closed.set()

    monkeypatch.setattr(_threaded_watch, "watch", slow_watch)
    task = asyncio.create_task(_step(tmp_path, asyncio.Event()).execute())
    try:
        await _until(entered.is_set)
        task.cancel()
        await _until(stopping.is_set)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert closed.is_set()
    assert _workers() == before


@pytest.mark.asyncio
async def test_watch_failure_propagates_and_allows_restart(tmp_path, monkeypatch):
    """Supervision can retry a failed watcher without accumulating threads."""
    before = _workers()
    attempts = []

    def broken_watch(*_paths, **_kwargs):
        attempts.append(threading.get_ident())
        raise OSError("watch failed")
        yield  # pylint: disable=unreachable

    monkeypatch.setattr(_threaded_watch, "watch", broken_watch)
    step = _step(tmp_path, asyncio.Event())
    for _ in range(2):
        with pytest.raises(OSError, match="watch failed"):
            await step.execute()
        assert _workers() == before
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_pull_backpressure_and_dispatch_failure_close_on_worker(tmp_path, monkeypatch):
    """Batches are pulled on demand and the generator closes on its worker."""
    before = _workers()
    threads = []
    closed = threading.Event()

    def batches(*_paths, **_kwargs):
        threads.append(threading.get_ident())
        try:
            yield {(Change.added, str(tmp_path / "daily/note.md"))}
            raise AssertionError("must not pull a batch while dispatch is pending")
        finally:
            threads.append(threading.get_ident())
            closed.set()

    monkeypatch.setattr(_threaded_watch, "watch", batches)
    step = _step(tmp_path, asyncio.Event())

    async def fail_dispatch(*_args, **_kwargs):
        await asyncio.sleep(0.05)
        assert not closed.is_set()
        raise RuntimeError("dispatch failed")

    step.dispatch_steps.side_effect = fail_dispatch
    with pytest.raises(RuntimeError, match="dispatch failed"):
        await step.execute()
    assert closed.is_set()
    assert len(threads) == 2 and threads[0] == threads[1] != threading.get_ident()
    assert _workers() == before


@pytest.mark.asyncio
async def test_already_stopped_watcher_does_not_start_thread(tmp_path):
    """A stop requested before startup allocates no worker."""
    before = _workers()
    stop = asyncio.Event()
    stop.set()
    async with aclosing(watch_in_thread(tmp_path, stop_event=stop)) as watcher:
        assert [batch async for batch in watcher] == []
    assert _workers() == before
