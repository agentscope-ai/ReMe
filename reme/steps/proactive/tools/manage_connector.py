"""Explicit event delivery and durable control of configured collection Jobs."""

from ..base import BaseAgentStep
from ....core.registry import R
from ....schema.proactive_agent import ManageConnectorRequest


@R.register("manage_connector_step")
class ManageConnectorStep(BaseAgentStep):
    """Control the workflow gate without starting a collector or creating a component."""

    async def execute(self):
        request = ManageConnectorRequest.model_validate(self.parameters())
        if request.action == "send":
            manager = self.agent_manager
            self.require_paths(manager.registry.directory)
            for event in request.events:
                self.require_paths(*(manager.files.path(path) for path in event.paths))
            result = await manager.deliver_events(request.task_names, request.events)
            self.context.response.success = not (result["rejected"] or result["errors"])
        else:
            from ..connector.base_connector import set_connector_enabled

            allowed = self.kwargs.get("connector_jobs", [])
            if request.connector_job not in allowed:
                raise ValueError("Connector Job is not in the configured control list")
            job = self.get_job(request.connector_job)
            if job is None or not job.enabled or not job.is_started:
                raise ValueError("Connector Job must be configured and started")
            result = await set_connector_enabled(self, request.connector_job, request.action == "enable")
        self.context.response.answer = result
        return self.context.response
