"""Explicit event delivery and durable control of configured collection Jobs."""

from ..base import BaseAgentStep
from ..connector.base import read_connector_control, update_connector_control
from ....core.registry import R
from ....schema.proactive_agent import ManageConnectorRequest, validate_runtime_names


@R.register("manage_connector_step")
class ManageConnectorStep(BaseAgentStep):
    """Control the workflow gate without starting a collector or creating a component."""

    async def preset(self, name):
        """Describe a configured Job without constructing or running its collection Steps."""
        job = self.get_job(name)
        config = self.app_context.app_config.jobs.get(name)
        steps = job.step_configs if job is not None else [step.model_dump() for step in config.steps] if config else []
        defaults = []
        for values in steps:
            if values.get("connector_job") != name:
                continue
            names = values.get("runtime_names")
            for target in validate_runtime_names([] if names is None else names, allow_empty=True):
                if target not in defaults:
                    defaults.append(target)
        control = await read_connector_control(self, name)
        return {
            "connector_job": name,
            "description": (job.description if job else "") or (config.description if config else ""),
            "schedule": getattr(job, "cron_expr", getattr(config, "cron", None)),
            "available": bool(job and job.enabled and job.is_started),
            "enabled": control["enabled"],
            "runtime_names": control.get("runtime_names", defaults),
        }

    async def execute(self):
        request = ManageConnectorRequest.model_validate(self.parameters())
        if request.action == "send":
            manager = self.agent_manager
            self.require_paths(manager.registry.directory)
            for event in request.events:
                self.require_paths(*(manager.files.path(path) for path in event.paths))
            result = await manager.deliver_events(request.runtime_names, request.events)
            self.context.response.success = not (result["rejected"] or result["errors"])
        else:
            allowed = validate_runtime_names(self.kwargs.get("connector_jobs", []), allow_empty=True)
            if request.action == "list":
                items = [await self.preset(name) for name in allowed]
                self.context.response.answer = {"items": items, "total": len(items)}
                return self.context.response
            if request.connector_job not in allowed:
                raise ValueError("Connector Job is not in the configured control list")
            job = self.get_job(request.connector_job)
            if job is None or not job.enabled or not job.is_started:
                raise ValueError("Connector Job must be configured and started")
            if request.action == "update":
                if request.runtime_names:
                    manager = self.agent_manager
                    self.require_paths(manager.registry.directory)
                    for name in request.runtime_names:
                        if (await manager.definition(name)).agent_type != "event":
                            raise ValueError("Connector targets must be event-triggered Runtimes")
                change = {"runtime_names": request.runtime_names}
            else:
                change = {"enabled": request.action == "enable"}
            saved = await update_connector_control(self, request.connector_job, **change)
            result = {**await self.preset(request.connector_job), "changed": saved["changed"]}
        self.context.response.answer = result
        return self.context.response
