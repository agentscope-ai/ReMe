"""Small asyncio helpers."""

import asyncio
from collections.abc import Callable
from contextlib import suppress
from typing import Any


async def complete_in_thread(func: Callable[..., Any], /, *args) -> Any:
    """Finish a side-effecting thread call, including through repeated cancellation."""
    task = asyncio.create_task(asyncio.to_thread(func, *args))
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
