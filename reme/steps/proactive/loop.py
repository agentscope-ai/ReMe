"""Completion-relative wakeups owned by an ordinary BackgroundJob."""

import asyncio
import math

from ..base_step import BaseStep
from ...components import R

WAKEUP_KEY = "pa_next_wakeup_minutes"


def _minutes(value) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not math.isfinite(value * 60)
        or value <= 0
    ):
        raise ValueError("minutes must be a positive finite number")
    return value


@R.register("pa_loop_step")
class PaLoopStep(BaseStep):
    """Wait the configured delay, then run one fresh, serial round at a time."""

    async def execute(self):
        stop = self.context.stop_event
        if not isinstance(stop, asyncio.Event):
            raise RuntimeError("pa_loop_step requires an asyncio stop_event")
        metadata = self.app_context.metadata
        metadata.pop(WAKEUP_KEY, None)
        default_minutes = _minutes(self.kwargs.get("default_wakeup_minutes"))
        minutes = default_minutes
        try:
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=minutes * 60)
                except asyncio.TimeoutError:
                    pass
                if stop.is_set():
                    break
                metadata.pop(WAKEUP_KEY, None)
                result = await self.run_job(
                    self.kwargs.get("job", "pa_proactive"),
                    stop_event=stop,
                    default_wakeup_minutes=default_minutes,
                )
                selected = metadata.pop(WAKEUP_KEY, default_minutes)
                minutes = selected if result.success else default_minutes
        finally:
            metadata.pop(WAKEUP_KEY, None)
        return self.context.response
