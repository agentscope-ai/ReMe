"""Small asyncio helpers."""

import asyncio
from collections.abc import Callable
from contextlib import suppress
from typing import Any


async def complete_in_thread(func: Callable[..., Any], /, *args) -> Any:
    """Finish a side-effecting thread call before propagating cancellation."""
    task = asyncio.create_task(asyncio.to_thread(func, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        with suppress(Exception):
            await task
        raise
