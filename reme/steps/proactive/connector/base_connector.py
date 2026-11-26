"""Collect payloads off-loop and publish one complete resource package."""

import asyncio
from contextlib import suppress
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import re
import shutil
from threading import Event as ThreadEvent
from uuid import uuid4

from ....schema.resource_metadata import RESOURCE_METADATA_FILENAME
from ....schema.proactive_agent import validate_task_names
from ....schema.proactive_runtime import validate_agent_name
from ....components.proactive.manager import BaseAgentManager
from ....enumeration import ComponentEnum
from ...base_step import BaseStep, Ref
from ...file_io import check_path_permission, resolve_path
from ....utils.files.path_lock import get_path_lock
from ....utils.files.workspace_files import WorkspaceFiles
from ....schema import ApplicationConfig, ResourceMetadata, Event
from ....utils.async_utils import finish_cleanup, complete_in_thread
from ....utils.files.file_sync import sync_directory, sync_file
from ....utils.timezone_utils import resolve_timezone


def resource_package_id(now: datetime) -> str:
    """Name a collection batch with local time and six UUID4 hex characters."""
    return f"{now:%H%M%S}_{uuid4().hex[:6]}"


# Control and delivery are files, not invocation-spanning Step fields.
def connector_control_path(step, connector_job):
    """Resolve the bounded workspace path of a configured collection gate."""
    name = validate_agent_name(connector_job)
    files = WorkspaceFiles(step.workspace_path)
    path = files.path(f"{step.config_value('metadata_dir')}/proactive/connectors/{name}.json")
    if not check_path_permission(
        step.workspace_path,
        path,
        step.context.get("_allowed_paths") if step.context else None,
    ):
        raise PermissionError("no permission to control this collection workflow")
    return files, path


def _read_connector_control(files, path, name):
    value = files.read_json(path, None)
    if value is None:
        return {"schema_version": 1, "connector_job": name, "enabled": True}
    if (
        set(value) != {"schema_version", "connector_job", "enabled", "updated_at"}
        or value["schema_version"] != 1
        or value["connector_job"] != name
        or type(value["enabled"]) is not bool
    ):
        raise ValueError("Invalid persistent Connector control state")
    stamp = datetime.fromisoformat(value["updated_at"])
    if stamp.tzinfo is None:
        raise ValueError("Connector control time must include a timezone")
    return value


async def connector_enabled(step, connector_job):
    """Read the durable collection gate; absent gates default to enabled."""
    files, path = connector_control_path(step, connector_job)
    return (await asyncio.to_thread(_read_connector_control, files, path, connector_job))["enabled"]


async def set_connector_enabled(step, connector_job, enabled):
    """Atomically update one configured workflow gate under its path lock."""
    files, path = connector_control_path(step, connector_job)
    async with await get_path_lock(path):
        previous = await asyncio.to_thread(_read_connector_control, files, path, connector_job)
        changed = previous["enabled"] != enabled
        config = step.app_context.app_config if step.app_context else ApplicationConfig()
        now = datetime.now(resolve_timezone(config.timezone))
        await complete_in_thread(
            files.write_json,
            path,
            {
                "schema_version": 1,
                "connector_job": connector_job,
                "enabled": enabled,
                "updated_at": now.isoformat(),
            },
        )
    return {"connector_job": connector_job, "enabled": enabled, "changed": changed}


def resource_event(workspace, directory, metadata, paths):
    """Create a stable standard Event for one published resource package."""
    relative = directory.relative_to(workspace).as_posix()
    return Event(
        id="resource-" + sha256(relative.encode()).hexdigest(),
        event_ts=metadata.created_at,
        source=metadata.source,
        description=metadata.description,
        paths=list(
            dict.fromkeys(
                path.relative_to(workspace).as_posix() for path in [directory / RESOURCE_METADATA_FILENAME, *paths]
            ),
        ),
    )


class BaseConnectorStep(BaseStep):
    """Subclasses write all payloads in ``collect`` and return their file entries."""

    source: str = ""
    description: str = ""
    _cancelled: ThreadEvent
    agent_manager = Ref(BaseAgentManager, ComponentEnum.AGENT_MANAGER)

    def __init__(self, connector_job, task_names=None, **kwargs):
        super().__init__(**kwargs)
        self.task_names = None if task_names is None else validate_task_names(task_names)
        self.connector_job = validate_agent_name(connector_job)

    def collect(self, directory: Path) -> list[dict]:
        """Write payloads into their final directory; raise if collection fails."""
        raise NotImplementedError

    def resource_target(self) -> tuple[Path, datetime]:
        """Validate the destination before collection or any external I/O."""
        config = self.app_context.app_config if self.app_context else ApplicationConfig()
        workspace = self.workspace_path.resolve()
        root, error = resolve_path(workspace, config.resource_dir)
        if error or root == workspace or (workspace / config.resource_dir).absolute() != root:
            raise ValueError(error or "resource_dir must stay inside the workspace without symlinks")
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", self.source):
            raise ValueError("resource source must be a lowercase directory name")
        now = datetime.now(resolve_timezone(config.timezone))
        target = root / self.source / now.date().isoformat() / resource_package_id(now)
        if not check_path_permission(workspace, target, self.context.get("_allowed_paths")):
            raise PermissionError("no permission to write this resource package")
        return target, now

    async def execute(self):
        if not await connector_enabled(self, self.connector_job):
            self.context.response.answer = {"skipped": "paused", "connector_job": self.connector_job}
            return self.context.response
        target, now = self.resource_target()
        root = target.parent.parent
        async with await get_path_lock(root):
            if self.task_names:
                self.require_paths(root, WorkspaceFiles(self.workspace_path).path(self.config_value("reme_agent_dir")))
                if not await self.flush_deliveries(root):
                    self.context.response.answer = {"skipped": "paused"}
                    return self.context.response
            if not await connector_enabled(self, self.connector_job):
                self.context.response.answer = {"skipped": "paused", "connector_job": self.connector_job}
                return self.context.response
            response = await self._collect(target, now)
            if self.task_names and self.context["resource"] is not None:
                await self.deliver_resource(target / RESOURCE_METADATA_FILENAME)
            return response

    def require_paths(self, *paths):
        """Enforce workspace and invocation permissions for resources and delivery state."""
        for path in paths:
            WorkspaceFiles(self.workspace_path).checked(path)
            if not check_path_permission(self.workspace_path, path, self.context.get("_allowed_paths")):
                raise PermissionError("no permission to access collector files")

    def event_payloads(self, metadata):
        """Choose the published payloads to include in the observation."""
        return [item.path for item in metadata.files]

    async def _collect(self, target, now):
        cancelled = self._cancelled = ThreadEvent()
        task = asyncio.create_task(asyncio.to_thread(self._collect_and_publish, target, now, cancelled))
        try:
            metadata = await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled.set()
            with suppress(Exception):
                await finish_cleanup(task)
            raise

        if metadata is None:
            self.context["resource"] = None
            self.context.response.success = True
            self.context.response.answer = {"skipped": "no_new_data"}
            return self.context.response
        result = {"path": self.to_workspace_relative(target / RESOURCE_METADATA_FILENAME), "files": metadata["files"]}
        self.context["resource"] = result
        self.context.response.success = True
        self.context.response.answer = result
        self.context.response.metadata.update(result)
        return self.context.response

    def _collect_and_publish(self, target: Path, created_at: datetime, cancelled: ThreadEvent) -> dict | None:
        """Write resources in place, then atomically publish their manifest before event delivery."""
        store = WorkspaceFiles(self.workspace_path)
        store.checked(target.parent).mkdir(parents=True, exist_ok=True)
        store.checked(target).mkdir()
        published = False
        try:
            sync_directory(target.parent)
            files = self.collect(target)
            if cancelled.is_set() or (self.context.stop_event is not None and self.context.stop_event.is_set()):
                raise RuntimeError("Resource collection stopped before publication")
            if files == []:
                return None
            metadata = ResourceMetadata(
                dataclass=type(self).__name__,
                source=self.source,
                created_at=created_at,
                modified_at=datetime.now(created_at.tzinfo),
                description=self.kwargs.get("description", self.description),
                files=files,
            )
            for item in metadata.files:
                payload = store.checked(target / item.path)
                if not payload.is_file() or payload.stat().st_size == 0:
                    raise ValueError(f"Missing, empty, or symlinked resource payload: {item.path}")
                with payload.open("rb") as handle:
                    sync_file(handle.fileno())
            if cancelled.is_set() or (self.context.stop_event is not None and self.context.stop_event.is_set()):
                raise RuntimeError("Resource collection stopped before publication")
            result = metadata.model_dump(mode="json", exclude_none=True)
            if self.task_names:
                self.prepare_delivery(target, metadata)
            store.write_json(target / RESOURCE_METADATA_FILENAME, result)
            published = True
            return result
        finally:
            if not published:
                shutil.rmtree(store.checked(target))

    def prepare_delivery(self, directory, metadata):
        """Freeze recipients and Event content before publishing a manifest."""
        files = WorkspaceFiles(self.workspace_path)
        paths = [directory / name for name in self.event_payloads(metadata)]
        event = resource_event(files.workspace, directory, metadata, paths)
        files.write_json(
            directory / ".delivery.json",
            {
                "connector_job": self.connector_job,
                "task_names": self.task_names,
                "delivered_to": [],
                "rejected": {},
                "events": [event.model_dump(mode="json")],
            },
        )

    async def deliver_batch(self, events, batch, save):
        """Retry only unresolved recipients and persist each completed handoff."""
        if not await connector_enabled(self, self.connector_job):
            return False
        targets = validate_task_names(batch["task_names"]) if batch["task_names"] else []
        rejected = batch.setdefault("rejected", {})
        delivered = batch.setdefault("delivered_to", [])
        pending = [name for name in targets if name not in delivered and name not in rejected]
        if events and pending:
            result = await self.agent_manager.deliver_events(task_names=pending, events=events)
            delivered.extend(name for name in result["deliveries"] if name not in delivered)
            rejected.update(result["rejected"])
            if result["deliveries"] or result["rejected"]:
                await save(batch)
            if result["errors"]:
                raise RuntimeError(f"Resource event delivery failed: {result['errors']}")
        return True

    async def flush_deliveries(self, root):
        """Recover published batches before requesting new source data."""
        for manifest in sorted(root.glob(f"*/*/{RESOURCE_METADATA_FILENAME}")):
            try:
                batch = await self._read_delivery(manifest)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.logger.error(f"Invalid saved resource {manifest}: {exc}")
                continue
            if not await self._deliver_saved(manifest, batch):
                return False
        return True

    async def deliver_resource(self, manifest):
        """Validate a saved package and hand off its frozen observation."""
        return await self._deliver_saved(manifest, await self._read_delivery(manifest))

    async def _read_delivery(self, manifest):
        files = WorkspaceFiles(self.workspace_path)
        state_path = manifest.with_name(".delivery.json")
        self.require_paths(manifest, state_path)
        state = await asyncio.to_thread(files.read_json, state_path, None)
        if state is None:
            return None
        if state.get("connector_job") != self.connector_job:
            raise ValueError("Resource delivery belongs to another Connector Job")
        targets = validate_task_names(state["task_names"])
        if all(name in state["delivered_to"] or name in state.get("rejected", {}) for name in targets):
            return None
        metadata = ResourceMetadata.model_validate_json(await asyncio.to_thread(files.read, manifest))
        if metadata.source != self.source:
            raise ValueError("Resource source does not match this collector")
        for item in metadata.files:
            path = files.checked(manifest.parent / item.path)
            self.require_paths(path)
            if not path.is_file():
                raise ValueError("Published resource payload is missing")
        events = [Event.model_validate(value) for value in state["events"]]
        return state, events

    async def _deliver_saved(self, manifest, batch):
        if batch is None:
            return True
        state, events = batch
        files = WorkspaceFiles(self.workspace_path)

        async def save(value):
            await complete_in_thread(files.write_json, manifest.with_name(".delivery.json"), value)

        return await self.deliver_batch(events, state, save)
