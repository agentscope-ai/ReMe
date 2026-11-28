"""Validate and authorize Connector operations, then delegate to their owners."""

from ..base import BaseAgentStep
from ...base_step import Ref
from ....components.proactive.connector import ConnectorManager
from ....core.registry import R
from ....enumeration import ComponentEnum
from ....schema.proactive_connector import ManageConnectorRequest


@R.register("manage_connector_step")
class ManageConnectorStep(BaseAgentStep):
    """Management has one identity, one partial update and one source catalog."""

    connector_manager = Ref(ConnectorManager, ComponentEnum.CONNECTOR_MANAGER)

    async def execute(self):
        request = ManageConnectorRequest.model_validate(self.parameters())
        if request.action == "send":
            self.require_paths(self.agent_manager.registry.directory)
            for event in request.events:
                self.require_paths(*(self.agent_manager.files.path(path) for path in event.paths))
            result = await self.agent_manager.deliver_events(request.runtime_names, request.events)
            self.context.response.success = not (result["rejected"] or result["errors"])
        else:
            manager = self.connector_manager
            self.require_paths(
                manager.directory,
                manager.controls.directory,
                manager.files.path(self.config_value("resource_dir")),
            )
            if request.action == "list":
                items = await manager.list_sources()
                result = {"items": items, "total": len(items)}
            elif request.action == "create":
                self.require_paths(manager.resource_root, self.agent_manager.registry.directory)
                result = await manager.create(request.model_dump(include={"name", "binding", "runtime_names"}))
            else:
                changes = request.model_dump(exclude_unset=True, exclude={"action", "source_id"})
                if "runtime_names" in changes or request.enabled is True:
                    self.require_paths(manager.resource_root, self.agent_manager.registry.directory)
                result = await manager.update(request.source_id, **changes)
        self.context.response.answer = result
        return self.context.response
