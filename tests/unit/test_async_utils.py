"""Side-effecting thread calls retain ownership through cancellation."""

import asyncio
import threading
import time

import anyio
import pytest

from reme.utils.async_utils import complete_in_thread


@pytest.mark.asyncio
@pytest.mark.parametrize("cancellations", [0, 1, 3])
@pytest.mark.parametrize("fail", [False, True])
async def test_complete_in_thread_finishes_before_propagating_result(cancellations, fail):
    """Repeated cancellation waits for success or failure without detaching work."""
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def operation():
        started.set()
        try:
            assert release.wait(5)
            if fail:
                raise RuntimeError("thread failed")
            return "complete"
        finally:
            finished.set()

    task = asyncio.create_task(complete_in_thread(operation))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        for _ in range(cancellations):
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done()
        assert not finished.is_set()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)

    assert finished.is_set()
    if cancellations:
        with pytest.raises(asyncio.CancelledError):
            task.result()
    elif fail:
        with pytest.raises(RuntimeError, match="thread failed"):
            task.result()
    else:
        assert task.result() == "complete"


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_name", ["fail_after", "move_on_after"])
@pytest.mark.parametrize("fail", [False, True])
async def test_complete_in_thread_preserves_anyio_cancel_scope(scope_name, fail):
    """AnyIO must recognize its original cancellation after the thread finishes."""
    finished = threading.Event()

    def operation():
        time.sleep(0.05)
        finished.set()
        if fail:
            raise RuntimeError("thread failed after cancellation")

    async def run_scope():
        with getattr(anyio, scope_name)(0.01) as scope:
            await complete_in_thread(operation)
        assert scope.cancelled_caught

    if scope_name == "fail_after":
        with pytest.raises(TimeoutError):
            await run_scope()
    else:
        await run_scope()
    assert finished.is_set()


@pytest.mark.asyncio
async def test_complete_in_thread_retains_first_cancellation_message():
    """Repeated cancellation must retain the original caller's reason."""
    started, release = threading.Event(), threading.Event()

    def operation():
        started.set()
        assert release.wait(5)

    task = asyncio.create_task(complete_in_thread(operation))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        for message in ("original cancellation", "cancel again", "cancel once more"):
            task.cancel(message)
            await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError, match="original cancellation"):
            await task
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
