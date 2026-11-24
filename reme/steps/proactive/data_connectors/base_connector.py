"""Collect payloads off-loop and publish one complete resource package."""

import asyncio
from abc import abstractmethod
from contextlib import suppress
from datetime import datetime
from pathlib import Path
import re
import shutil
from threading import Event
from uuid import uuid4
from zoneinfo import ZoneInfo

from ....schema.resource_metadata import RESOURCE_METADATA_FILENAME
from ....schema.proactive_agent import validate_task_names
from ....components.proactive_manager import BaseProactiveManager
from ....enumeration import ComponentEnum
from ...base_step import BaseStep, Ref
from ...file_io import check_path_permission, resolve_path
from .resource_delivery import ResourceDelivery
from ....utils.files.path_lock import get_path_lock
from ....utils.files.workspace_files import WorkspaceFiles
from ....schema import ApplicationConfig, ResourceMetadata
from ....utils.async_utils import finish_cleanup
from ....utils.files.file_sync import sync_directory, sync_file


def resource_package_id(now: datetime) -> str:
    """Name a collection batch with local time and six UUID4 hex characters."""
    return f"{now:%H%M%S}_{uuid4().hex[:6]}"


class BaseConnectorStep(BaseStep):
    """Subclasses write all payloads in ``collect`` and return their file entries."""

    source: str = ""
    description: str = ""
    _cancelled: Event
    proactive_manager = Ref(BaseProactiveManager, ComponentEnum.PROACTIVE_MANAGER)

    def __init__(self, task_names=None, **kwargs):
        super().__init__(**kwargs)
        self.task_names = None if task_names is None else validate_task_names(task_names)
        self.delivery: ResourceDelivery | None = None

    @abstractmethod
    def collect(self, directory: Path) -> list[dict]:
        """Write payloads into their final directory; raise if any collection fails."""

    def resource_target(self) -> tuple[Path, datetime]:
        """Validate the destination before collection or any external I/O."""
        config = self.app_context.app_config if self.app_context else ApplicationConfig()
        workspace = self.workspace_path.resolve()
        root, error = resolve_path(workspace, config.resource_dir)
        if error or root == workspace or (workspace / config.resource_dir).absolute() != root:
            raise ValueError(error or "resource_dir must stay inside the workspace without symlinks")
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", self.source):
            raise ValueError("resource source must be a lowercase directory name")
        now = datetime.now(ZoneInfo(config.timezone)) if config.timezone else datetime.now().astimezone()
        target = root / self.source / now.date().isoformat() / resource_package_id(now)
        if not check_path_permission(workspace, target, self.context.get("_allowed_paths")):
            raise PermissionError("no permission to write this resource package")
        return target, now

    async def execute(self):
        assert self.context is not None
        target, now = self.resource_target()
        root = target.parent.parent
        async with await get_path_lock(root):
            if self.task_names is not None:
                files = WorkspaceFiles(self.workspace_path)
                self.require_path(root)
                self.require_path(files.path(self.config_value("reme_agent_dir")))
                self.delivery = ResourceDelivery(self, files, root, self.task_names)
                await self.delivery.flush()
            response = await self._collect(target, now)
            if self.delivery is not None and self.context["resource"] is not None:
                await self.delivery.deliver(target / RESOURCE_METADATA_FILENAME)
            return response

    def require_path(self, path):
        """Enforce workspace and invocation permissions for resources and delivery state."""
        WorkspaceFiles(self.workspace_path).checked(path)
        if not check_path_permission(self.workspace_path, path, self.context.get("_allowed_paths")):
            raise PermissionError("no permission to access collector files")

    def event_payloads(self, metadata):
        """Choose the published payloads to include in the observation."""
        return [item.path for item in metadata.files]

    async def _collect(self, target, now):
        cancelled = self._cancelled = Event()
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

    def _collect_and_publish(self, target: Path, created_at: datetime, cancelled: Event) -> dict | None:
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
            if self.delivery is not None:
                self.delivery.prepare(target)
            store.write_json(target / RESOURCE_METADATA_FILENAME, result)
            published = True
            return result
        finally:
            if not published:
                shutil.rmtree(store.checked(target))
