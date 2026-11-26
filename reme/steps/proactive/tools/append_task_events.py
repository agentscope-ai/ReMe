"""Append the same observations to one or more background task inboxes."""

from ..agent import BaseAgentStep
from ....core.registry import R
from ....schema.proactive_agent import AppendTaskEventsRequest


@R.register("append_task_events_step")
class AppendTaskEventsStep(BaseAgentStep):
    """Validate complete events and every referenced workspace path before append."""

    async def execute(self):
        request = AppendTaskEventsRequest.model_validate(self.parameters())
        manager = self.proactive_manager
        self.require_paths(manager.registry.directory)
        for event in request.events:
            self.require_paths(*(manager.files.path(path) for path in event.paths))
        result = await manager.deliver_events(request.task_names, request.events)
        self.context.response.answer = result
        self.context.response.success = not (result["rejected"] or result["errors"])
        return self.context.response
