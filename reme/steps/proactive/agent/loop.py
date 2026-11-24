"""BackgroundJob-owned proactive execution and inbox wakeups."""

import asyncio

from .base import BaseAgentStep
from ....core.registry import R


@R.register("proactive_loop_step")
class ProactiveLoopStep(BaseAgentStep):
    """Dispatch independent serial agents and await their cancellation on shutdown."""

    async def execute(self):
        stop = self.context.stop_event
        if not isinstance(stop, asyncio.Event):
            raise RuntimeError("proactive_loop_step requires a BackgroundJob stop_event")
        runtime = self.proactive_runtime
        try:
            while not stop.is_set():
                await runtime.tick(self.run_job)
                await runtime.wait(stop)
        finally:
            await runtime.stop_workers()
        return self.context.response
