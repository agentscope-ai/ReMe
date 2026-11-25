"""BackgroundJob-owned proactive execution and inbox wakeups."""

import asyncio

from .base import BaseAgentStep
from ....core.registry import R


@R.register("proactive_loop_step")
class ProactiveLoopStep(BaseAgentStep):
    """Run the Proactive Manager under the BackgroundJob lifecycle."""

    async def execute(self):
        stop = self.context.stop_event
        if not isinstance(stop, asyncio.Event):
            raise RuntimeError("proactive_loop_step requires a BackgroundJob stop_event")
        await self.proactive_manager.run(stop, self.run_job)
        return self.context.response


@R.register("run_proactive_agent_step")
class RunProactiveAgentStep(BaseAgentStep):
    """Select the runtime's execution step under the shared Job."""

    def __init__(self, prompt_dict=None, **kwargs):
        super().__init__(**kwargs)
        self.prompt_overrides = prompt_dict

    async def execute(self):
        arguments = self.parameters()
        if set(arguments) != {"agent_name"}:
            raise ValueError("run_proactive_agent_step requires only agent_name")
        self.require_paths(self.workspace_path)
        definition = await self.proactive_manager.definition(arguments["agent_name"])
        (response,) = await self.dispatch_steps(
            [
                {
                    "backend": definition.step_backend,
                    "language": self.language,
                    "prompt_dict": self.prompt_overrides,
                    **self.kwargs,
                },
            ],
        )
        return response
