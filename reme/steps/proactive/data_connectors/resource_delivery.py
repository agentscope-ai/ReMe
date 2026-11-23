"""Recover event delivery from published resource packages."""

import asyncio
from hashlib import sha256

from ....schema import Event, ResourceMetadata
from ....schema.resource_metadata import RESOURCE_METADATA_FILENAME
from ....schema.proactive_agent import validate_task_names
from ....utils.async_utils import complete_in_thread

_DELIVERY_FILENAME = ".delivery.json"


def resource_event(workspace, directory, metadata, paths):
    """Derive a stable observation ID from its workspace-relative package directory."""
    relative = directory.relative_to(workspace).as_posix()
    return Event(
        id=f"resource-{sha256(relative.encode()).hexdigest()}",
        event_ts=metadata.created_at,
        source=metadata.source,
        description=metadata.description,
        paths=[path.relative_to(workspace).as_posix() for path in paths],
    )


class ResourceDelivery:
    """Freeze recipients before publishing and retain only unfinished deliveries."""

    def __init__(self, connector, files, root, task_names):
        self.connector = connector
        self.files = files
        self.root = root
        self.task_names = task_names

    def prepare(self, directory):
        """Stage the delivery beside the payloads before the manifest becomes visible."""
        path = directory / _DELIVERY_FILENAME
        self.connector.require_path(path)
        self.files.write_json(path, {"pending": self.task_names})

    async def flush(self):
        """Retry the original observation for each remaining recipient."""
        pattern = f"*/*/{RESOURCE_METADATA_FILENAME}"
        for manifest in await asyncio.to_thread(lambda: sorted(self.root.glob(pattern))):
            await self.deliver(manifest)

    async def deliver(self, manifest):
        """Send one published package to its unfinished recipients."""
        state_path = manifest.with_name(_DELIVERY_FILENAME)
        self.connector.require_path(state_path)
        state = await asyncio.to_thread(self.files.read_json, state_path, None)
        if state is None or not state["pending"]:
            return

        targets = validate_task_names(state["pending"])
        self.connector.require_path(manifest)
        metadata = ResourceMetadata.model_validate_json(await asyncio.to_thread(self.files.read, manifest))
        if metadata.source != self.connector.source:
            raise ValueError(f"Unexpected resource source: {metadata.source}")
        if not self.connector.delivery_ready(manifest, metadata):
            return
        paths = [manifest.parent / name for name in self.connector.event_payloads(metadata)]
        for path in paths:
            self.connector.require_path(path)
        event = resource_event(self.files.workspace, manifest.parent, metadata, paths)
        result = await self.connector.proactive_manager.deliver_events(task_names=targets, events=[event])
        rejected = result["rejected"]
        if rejected:
            self.connector.logger.warning(f"Resource event {event.id} rejected by tasks: {rejected}")
        if result["deliveries"] or rejected:
            await complete_in_thread(
                self.files.write_json,
                state_path,
                {"pending": list(result["errors"]), "rejected": {**state.get("rejected", {}), **rejected}},
            )
        if result["errors"]:
            raise RuntimeError(f"Resource event delivery failed: {result['errors']}")
