"""Collect payloads off-loop and publish one complete resource package."""

import asyncio
from abc import abstractmethod
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from uuid import uuid4
from zoneinfo import ZoneInfo

from ..base_step import BaseStep
from ..file_io._path import _check_path_permission, resolve_path
from ...schema import ApplicationConfig, ResourceMetadata
from ...utils.async_utils import finish_cleanup


class BaseConnectorStep(BaseStep):
    """Subclasses write all payloads in ``collect`` and return their file entries."""

    source: str = ""
    description: str = ""

    @abstractmethod
    def collect(self, directory: Path) -> list[dict]:
        """Write payloads into the private directory; raise if any collection fails."""

    async def execute(self):
        assert self.context is not None
        config = self.app_context.app_config if self.app_context else ApplicationConfig()
        workspace = self.workspace_path.resolve()
        root, error = resolve_path(workspace, config.resource_dir)
        if error or root == workspace or (workspace / config.resource_dir).absolute() != root:
            raise ValueError(error or "resource_dir must stay inside the workspace without symlinks")
        now = datetime.now(ZoneInfo(config.timezone)) if config.timezone else datetime.now().astimezone()
        target = root / now.date().isoformat() / uuid4().hex
        if not _check_path_permission(workspace, target, self.context.get("_allowed_paths")):
            raise PermissionError("no permission to write this resource package")

        cancelled = Event()
        task = asyncio.create_task(asyncio.to_thread(self._collect_and_publish, target, now, cancelled))
        try:
            metadata = await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled.set()
            with suppress(Exception):
                await finish_cleanup(task)
            raise

        result = {"path": self.to_workspace_relative(target / "metadata.json"), "files": metadata["files"]}
        self.context["resource"] = result
        self.context.response.success = True
        self.context.response.answer = result
        self.context.response.metadata.update(result)
        return self.context.response

    def _collect_and_publish(self, target: Path, created_at: datetime, cancelled: Event) -> dict:
        """Stage outside resource/, then atomically rename on the same filesystem."""
        with TemporaryDirectory(prefix=".reme-resource-", dir=self.workspace_path.resolve()) as temporary:
            directory = Path(temporary)
            files = self.collect(directory)
            metadata = ResourceMetadata(
                dataclass=type(self).__name__,
                source=self.source,
                created_at=created_at,
                modified_at=datetime.now(created_at.tzinfo),
                description=self.kwargs.get("description", self.description),
                files=files,
            )
            for item in metadata.files:
                payload = directory / item.path
                if payload.resolve() != payload or not payload.is_file() or payload.stat().st_size == 0:
                    raise ValueError(f"Missing, empty, or symlinked resource payload: {item.path}")
            (directory / "metadata.json").write_text(
                metadata.model_dump_json(indent=2, exclude_none=True) + "\n",
                encoding="utf-8",
            )
            if cancelled.is_set() or (self.context.stop_event is not None and self.context.stop_event.is_set()):
                raise RuntimeError("Resource collection stopped before publication")
            if target.parent.resolve() != target.parent:
                raise ValueError("Resource package parent must not contain symlinks")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.parent.resolve() != target.parent:
                raise ValueError("Resource package parent must not contain symlinks")
            if target.exists():
                raise FileExistsError(f"Resource package already exists: {target}")
            directory.rename(target)
            return metadata.model_dump(mode="json", exclude_none=True)
