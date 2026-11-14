"""BackgroundJob-owned proactive execution and inbox wakeups."""

import asyncio

from ..agent import BaseAgentStep
from ....components import R
from ....utils.async_utils import finish_cleanup


@R.register("proactive_loop_step")
class ProactiveLoopStep(BaseAgentStep):
    """Dispatch independent serial agents and await their cancellation on shutdown."""

    async def execute(self):
        stop = self.context.stop_event
        if not isinstance(stop, asyncio.Event):
            raise RuntimeError("proactive_loop_step requires a BackgroundJob stop_event")
        runtime = self.agent_runtime
        try:
            while not stop.is_set():
                await runtime.tick(self.run_job, self.kwargs.get("job", "proactive_run"))
                await runtime.wait(stop)
        finally:
            tasks = list(runtime.workers.values())
            for task in tasks:
                task.cancel()
            if tasks:
                await finish_cleanup(asyncio.gather(*tasks, return_exceptions=True))
        return self.context.response


@R.register("inbox_loop_step")
class InboxLoopStep(BaseAgentStep):
    """Request serial Chat turns independently of proactive model execution."""

    async def execute(self):
        stop = self.context.stop_event
        if not isinstance(stop, asyncio.Event):
            raise RuntimeError("inbox_loop_step requires a BackgroundJob stop_event")
        try:
            while not stop.is_set():
                await self.agent_runtime.inbox.tick(self.run_job, self.kwargs.get("job", "inbox_chat"))
                await self.agent_runtime.wait(stop)
        finally:
            await self.agent_runtime.inbox.close()
        return self.context.response
