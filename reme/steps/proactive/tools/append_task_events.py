"""Append observations to a background task's input queue."""

from ..agent import BaseAgentStep
from ....core.registry import R
from ....schema.proactive_agent import AppendTaskEventsRequest


@R.register("append_task_events_step")
class AppendTaskEventsStep(BaseAgentStep):
    """Validate complete events and every referenced workspace path before append."""

    async def execute(self):
        request = AppendTaskEventsRequest.model_validate(self.parameters())
        runtime = self.proactive_runtime
        self.require_paths(runtime.agents.directory)
        for event in request.events:
            self.require_paths(*(runtime.files.path(path) for path in event.paths))
        self.context.response.answer = await runtime.append_events(request.agent_name, request.events)
        return self.context.response
