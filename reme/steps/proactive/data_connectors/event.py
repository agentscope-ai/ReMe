"""Publish a complete resource package as an observation for a named agent."""

from hashlib import sha256
from pathlib import PurePosixPath

from ..agent import BaseAgentStep
from ....core.registry import R
from ....schema import Event, ResourceMetadata


@R.register("resource_event_step")
class ResourceEventStep(BaseAgentStep):
    """Run after a connector; stable package IDs make publication retry-safe."""

    async def execute(self):
        if "resource" in self.context and self.context["resource"] is None:
            self.context.response.answer = {"skipped": "no_new_data"}
            return self.context.response
        agent_name = self.context.get("agent_name")
        if not isinstance(agent_name, str) or not agent_name:
            raise ValueError("resource_event_step requires a target agent_name")
        resource = self.context.get("resource")
        if not isinstance(resource, dict) or not isinstance(resource.get("path"), str):
            raise ValueError("resource_event_step requires a published resource")
        manifest = self.proactive_runtime.files.path(resource["path"])
        self.require_paths(manifest)
        metadata = ResourceMetadata.model_validate_json(self.proactive_runtime.files.read(manifest))
        parent = PurePosixPath(resource["path"]).parent
        paths = [resource["path"], *(str(parent / item.path) for item in metadata.files)]
        self.require_paths(*(self.proactive_runtime.files.path(path) for path in paths))
        event = Event(
            id=f"resource-{sha256(parent.as_posix().encode()).hexdigest()}",
            event_ts=metadata.created_at,
            source=metadata.source,
            description=metadata.description,
            paths=paths,
        )
        result = await self.proactive_runtime.append_events(agent_name, [event])
        self.context.response.answer = {**resource, **result}
        return self.context.response
