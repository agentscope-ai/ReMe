"""Pull watchfiles batches on a private thread, with invocation-owned cleanup."""

import asyncio
import threading
from collections.abc import AsyncIterator, Generator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from watchfiles import Change, watch

ChangeBatch = set[tuple[Change, str]]


async def _wait_for_stop(stop_event: asyncio.Event | threading.Event) -> None:
    if isinstance(stop_event, asyncio.Event):
        await stop_event.wait()
    else:
        # BackgroundJob also supports running on its own event-loop thread.
        while not stop_event.is_set():
            await asyncio.sleep(0.05)


def _next_batch(iterator: Generator[ChangeBatch, None, None]) -> ChangeBatch | None:
    # StopIteration cannot be transported through an asyncio Future.
    return next(iterator, None)


async def watch_in_thread(
    *paths: str | Path,
    stop_event: asyncio.Event | threading.Event,
    **kwargs: Any,
) -> AsyncIterator[ChangeBatch]:
    """Adapt ``watch()`` without borrowing either shared executor's workers.

    Only one next() is submitted at a time, so downstream processing supplies
    backpressure without an unbounded batch queue. The generator, including its
    initial scan, advances and closes on the same private worker.
    """
    if stop_event.is_set():
        return

    loop = asyncio.get_running_loop()
    thread_stop = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reme-watch")
    iterator = watch(*paths, stop_event=thread_stop, raise_interrupt=False, **kwargs)
    stopped = asyncio.create_task(_wait_for_stop(stop_event))
    pending: asyncio.Future | None = None

    async def close() -> None:
        thread_stop.set()
        stopped.cancel()
        try:
            await asyncio.gather(stopped, *([pending] if pending is not None else []), return_exceptions=True)
            await loop.run_in_executor(executor, iterator.close)
        finally:
            # No work remains, so this only joins the now-idle private worker.
            executor.shutdown(wait=True)

    try:
        while not stop_event.is_set():
            pending = loop.run_in_executor(executor, _next_batch, iterator)
            done, _ = await asyncio.wait((pending, stopped), return_when=asyncio.FIRST_COMPLETED)
            if stopped in done:
                break
            batch = pending.result()
            pending = None
            if batch is None:
                break
            yield batch
    finally:
        # Cancellation must not abandon a running next(), even when shutdown
        # cancels this invocation again while it is already cleaning up.
        cleanup = asyncio.create_task(close())
        cancelled = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                cancelled = True
        cleanup.result()
        if cancelled:
            raise asyncio.CancelledError
