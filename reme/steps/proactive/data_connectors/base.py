"""Collect resources off-loop and return text with progressively readable paths."""

import asyncio
from abc import abstractmethod
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from zoneinfo import ZoneInfo

from .._resource import resource_root
from ...base_step import BaseStep
from ...file_io._path import _check_path_permission, validate_filename_component
from ....schema import ApplicationConfig, ConnectorResult
from ....utils.async_utils import finish_cleanup


class BaseConnectorStep(BaseStep):
    """Stage under a private directory; publish resources under ``resource_dir/source``."""

    source: str = ""
    _cancelled: Event

    @abstractmethod
    def collect(self, directory: Path, created_at: datetime) -> ConnectorResult:
        """Return textual context and staged relative paths; raise on collection failure."""

    def resource_target(self) -> Path:
        """Validate the publication destination before collecting any external data."""
        root = resource_root(self)
        error = validate_filename_component(self.source, kind="connector source")
        if error:
            raise ValueError(error)
        target = root / self.source
        if target.resolve() != target:
            raise ValueError("connector directory must not contain symlinks")
        if not _check_path_permission(self.workspace_path, target, self.context.get("_allowed_paths")):
            raise PermissionError("no permission to write connector resources")
        return target

    def collection_time(self) -> datetime:
        """Use the application's timezone consistently for native and file timestamps."""
        config = self.app_context.app_config if self.app_context else ApplicationConfig()
        timezone = config.timezone
        return datetime.now(ZoneInfo(timezone)) if timezone else datetime.now().astimezone()

    async def execute(self):
        target = self.resource_target()
        now = self.collection_time()
        cancelled = self._cancelled = Event()
        task = asyncio.create_task(asyncio.to_thread(self._collect_and_publish, target, now, cancelled))
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled.set()
            with suppress(Exception):
                await finish_cleanup(task)
            raise
        self.context.response.answer = result
        self.context.response.success = True
        return self.context.response

    def _check_stopped(self, cancelled: Event) -> None:
        """Let collectors interrupt pagination and waits between bounded I/O calls."""
        if cancelled.is_set() or (self.context.stop_event is not None and self.context.stop_event.is_set()):
            raise RuntimeError("Resource collection stopped before publication")

    def _check_running(self, target: Path, cancelled: Event) -> None:
        self._check_stopped(cancelled)
        if target.resolve() != target:
            raise ValueError("connector directory must not contain symlinks")

    def _publish(self, source: Path, target: Path, published: list[Path], cancelled: Event) -> None:
        self._check_running(target.parent, cancelled)
        if source.is_dir():
            target.mkdir()
            published.append(target)
            for child in sorted(source.iterdir()):
                self._publish(child, target / child.name, published, cancelled)
        else:
            target.hardlink_to(source)  # Atomic and exclusive, on the workspace's filesystem.
            published.append(target)

    def _collect_and_publish(self, target: Path, created_at: datetime, cancelled: Event) -> dict:
        self._check_running(target, cancelled)
        with TemporaryDirectory(prefix=".reme-resource-", dir=self.workspace_path.resolve()) as temporary:
            directory = Path(temporary)
            result = self.collect(directory, created_at)
            if not isinstance(result, ConnectorResult):
                result = ConnectorResult.model_validate(result)
            for path in directory.rglob("*"):
                if path.is_symlink() or not (path.is_file() or path.is_dir()):
                    raise ValueError(f"Resource payload must be a regular file or directory: {path.name}")
            for name in result.paths:
                payload = directory / name
                if not payload.exists() or (payload.is_file() and payload.stat().st_size == 0):
                    raise ValueError(f"Missing or empty resource payload: {name}")
            self._check_running(target, cancelled)
            if not result.paths:
                return result.model_dump()
            published: list[Path] = []
            try:
                target.mkdir(parents=True, exist_ok=True)
                self._check_running(target, cancelled)
                for name in sorted({Path(path).parts[0] for path in result.paths}):
                    self._publish(directory / name, target / name, published, cancelled)
                self._check_running(target, cancelled)
            except BaseException:
                for path in reversed(published):
                    if path.is_dir():
                        path.rmdir()
                    else:
                        path.unlink()
                raise
            return ConnectorResult(
                context=result.context,
                paths=[self.to_workspace_relative(target / path) for path in result.paths],
            ).model_dump()
