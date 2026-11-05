"""Small asyncio helpers."""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any, TypeVar

T = TypeVar("T")


async def finish_cleanup(operation: Awaitable[T]) -> T:
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
        with suppress(Exception, asyncio.CancelledError):
            task.result()
        raise cancellation
    return task.result()


async def complete_in_thread(func: Callable[..., Any], /, *args, **kwargs) -> Any:
    """Finish a side-effecting thread call, including through repeated cancellation."""
    return await finish_cleanup(asyncio.to_thread(func, *args, **kwargs))
