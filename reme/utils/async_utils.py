"""Small asyncio helpers."""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any


async def finish_cleanup(operation: Awaitable):
    """Finish cleanup before propagating any cancellation received while waiting."""
    task = asyncio.ensure_future(operation)
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
        except Exception:
            if cancellation is None:
                raise
            break
    if cancellation is not None:
        with suppress(Exception):
            task.result()
        raise cancellation
    return task.result()


async def complete_in_thread(func: Callable[..., Any], /, *args) -> Any:
    """Finish a side-effecting thread call, including through repeated cancellation."""
    return await finish_cleanup(asyncio.to_thread(func, *args))
