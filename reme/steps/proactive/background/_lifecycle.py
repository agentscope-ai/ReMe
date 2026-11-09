"""Small lifecycle helpers shared by PA background steps."""

import asyncio

from ....utils.async_utils import finish_cleanup


def require_stop_event(stop) -> asyncio.Event:
    """Require the BackgroundJob's application-owned stop signal."""
    if not isinstance(stop, asyncio.Event):
        raise RuntimeError("PA background steps require an asyncio stop_event")
    return stop


async def wait_for_stop(stop: asyncio.Event, seconds: float) -> None:
    """Wait until shutdown or the next local scheduling check."""
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except TimeoutError:
        pass


async def cancel_workers(workers) -> None:
    """Await all workers after requesting cancellation."""
    for worker in workers:
        worker.cancel()
    await finish_cleanup(asyncio.gather(*workers, return_exceptions=True))
