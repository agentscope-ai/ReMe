"""Completion-relative wakeups owned by an ordinary BackgroundJob."""

import asyncio
import math

from ..base_step import BaseStep
from ...components import R

WAKEUP_KEY = "pa_next_wakeup_minutes"


@R.register("set_next_wakeup_step")
class SetNextWakeupStep(BaseStep):
    """Set this round's next delay; the loop consumes it once."""

    async def execute(self):
        minutes = self.context.get("minutes")
        if (
            isinstance(minutes, bool)
            or not isinstance(minutes, (int, float))
            or not math.isfinite(minutes)
            or not math.isfinite(minutes * 60)
            or minutes <= 0
        ):
            raise ValueError("minutes must be a positive finite number")
        self.app_context.metadata[WAKEUP_KEY] = minutes
        self.context.response.answer = {"minutes": minutes}
        return self.context.response


@R.register("pa_loop_step")
class PaLoopStep(BaseStep):
    """Wait five minutes after startup, then run one fresh, serial round at a time."""

    async def execute(self):
        stop = self.context.stop_event
        if not isinstance(stop, asyncio.Event):
            raise RuntimeError("pa_loop_step requires an asyncio stop_event")
        metadata = self.app_context.metadata
        metadata.pop(WAKEUP_KEY, None)
        minutes = 5
        try:
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=minutes * 60)
                except asyncio.TimeoutError:
                    pass
                if stop.is_set():
                    break
                metadata.pop(WAKEUP_KEY, None)
                result = await self.run_job(self.kwargs.get("job", "pa_proactive"), stop_event=stop)
                selected = metadata.pop(WAKEUP_KEY, 5)
                minutes = selected if result.success else 5
        finally:
            metadata.pop(WAKEUP_KEY, None)
        return self.context.response
