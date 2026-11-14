"""Publish a complete resource package as an observation for a named agent."""

from hashlib import sha256
from pathlib import PurePosixPath

from ..agent import BaseAgentStep
from ....components import R
from ....schema import Event, ResourceMetadata


@R.register("resource_event_step")
class ResourceEventStep(BaseAgentStep):
    """Run after a connector; stable package IDs make publication retry-safe."""

    async def execute(self):
        agent_id = self.context.get("agent_id")
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("resource_event_step requires a target agent_id")
        if (await self.agent_runtime.definition(agent_id)).trigger_type != "event":
            raise ValueError("resource observations require an event agent")
        resource = self.context.get("resource")
        if not isinstance(resource, dict) or not isinstance(resource.get("path"), str):
            raise ValueError("resource_event_step requires a published resource")
        manifest = self.agent_runtime.files.path(resource["path"])
        self.require_paths(manifest)
        metadata = ResourceMetadata.model_validate_json(self.agent_runtime.files.read(manifest))
        if metadata.source == "cron":
            raise ValueError("resource source is reserved for framework triggers")
        parent = PurePosixPath(resource["path"]).parent
        paths = [resource["path"], *(str(parent / item.path) for item in metadata.files)]
        self.require_paths(*(self.agent_runtime.files.path(path) for path in paths))
        event = Event(
            id=f"resource-{sha256(parent.as_posix().encode()).hexdigest()}",
            event_ts=metadata.created_at,
            source=metadata.source,
            description=metadata.description,
            paths=paths,
        )
        result = await self.agent_runtime.append(agent_id, event)
        self.context.response.answer = {**resource, **result}
        return self.context.response
