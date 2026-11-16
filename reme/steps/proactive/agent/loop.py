"""BackgroundJob-owned proactive execution and inbox wakeups."""

import asyncio

from .base import BaseAgentStep
from ....components import R
from ....utils.async_utils import finish_cleanup


@R.register("agent_loop_step")
class AgentLoopStep(BaseAgentStep):
    """Dispatch independent serial agents and await their cancellation on shutdown."""

    async def execute(self):
        stop = self.context.stop_event
        if not isinstance(stop, asyncio.Event):
            raise RuntimeError("agent_loop_step requires a BackgroundJob stop_event")
        runtime = self.agent_runtime
        try:
            while not stop.is_set():
                await runtime.tick(self.run_job)
                await runtime.wait(stop)
        finally:
            await runtime.notifications.close()
            tasks = list(runtime.workers.values())
            for task in tasks:
                task.cancel()
            if tasks:
                await finish_cleanup(asyncio.gather(*tasks, return_exceptions=True))
        return self.context.response
