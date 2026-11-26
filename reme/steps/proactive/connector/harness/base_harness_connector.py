"""Incremental Harness collection with provider-owned opaque cursors."""

import asyncio
from abc import abstractmethod
from contextlib import suppress
from datetime import datetime
import os
import shutil
from threading import Event as ThreadEvent

from ..base_connector import BaseConnectorStep, resource_package_id, resource_event, connector_enabled
from .....schema import ApplicationConfig, ResourceMetadata
from .....schema.resource_metadata import RESOURCE_METADATA_FILENAME
from .....utils.files.path_lock import get_path_lock
from .....utils.files.workspace_files import WorkspaceFiles
from .....utils.files.file_sync import sync_directory, sync_file
from .....utils.timezone_utils import resolve_timezone


class BaseHarnessConnectorStep(BaseConnectorStep):
    """Own baseline, bounded reading, publication, cancellation and progress handoff."""

    def __init__(self, max_scan_bytes=32 * 1024 * 1024, **kwargs):
        super().__init__(**kwargs)
        if type(max_scan_bytes) is not int or max_scan_bytes < 1:
            raise ValueError("max_scan_bytes must be a positive integer")
        self.max_scan_bytes = max_scan_bytes
        self.cancelled = ThreadEvent()

    @abstractmethod
    def _index(self):
        """Return a source adapter with load, scan and commit over opaque cursors."""

    @abstractmethod
    def render_dialogue(self, messages, now):
        """Render normalized source messages and attachments for the Agent."""

    def _now(self):
        config = self.app_context.app_config if self.app_context else ApplicationConfig()
        return datetime.now(resolve_timezone(config.timezone))

    def _check_stopped(self):
        if self.cancelled.is_set() or self.context.stop_event is not None and self.context.stop_event.is_set():
            raise RuntimeError("Harness collection stopped")

    def _authorize(self):
        files = WorkspaceFiles(self.workspace_path)
        directory = files.path(self.config_value("resource_dir"))
        if directory == self.workspace_path:
            raise ValueError("resource_dir must be a workspace subdirectory")
        self.require_paths(directory, files.path(self.config_value("reme_agent_dir")))
        return files.checked(directory / self.source)

    async def _offload(self, function, *args):
        task = asyncio.create_task(asyncio.to_thread(function, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            self.cancelled.set()
            with suppress(Exception):
                await task
            raise

    async def initialize(self):
        """Establish a baseline without replaying existing source history."""
        if not await connector_enabled(self, self.connector_job):
            self.context.response.answer = {"skipped": "paused"}
            return self.context.response
        async with await get_path_lock(self._authorize()):
            if not await connector_enabled(self, self.connector_job):
                self.context.response.answer = {"skipped": "paused"}
                return self.context.response
            state = await self._offload(lambda: self._index().load(self._now()))
        self.context.response.answer = {"indexed_files": len(state["files"])}
        return self.context.response

    def _prepare(self):
        index = self._index()
        now = self._now()
        state = index.load(now)
        pending = index.files.read_json(index.pending_path, None)
        if pending is not None:
            return pending
        scan = index.scan(state, self.max_scan_bytes, self.max_file_bytes())
        batch = {
            "id": resource_package_id(now),
            "task_names": self.task_names,
            "delivered_to": [],
            "resource": None,
            **{key: scan[key] for key in ("updates", "message_ids", "next_path", "scanned_bytes", "errors")},
        }
        directory = None
        try:
            if scan["messages"]:
                target = index.directory / now.date().isoformat() / batch["id"]
                self.require_paths(target)
                index.files.checked(target.parent).mkdir(parents=True, exist_ok=True)
                index.files.checked(target).mkdir()
                directory = target
                index.files.write(directory / "dialogue.md", self.render_dialogue(scan["messages"], now))
                index.files.write_jsonl(directory / "dialogue.jsonl", scan["messages"])
                for name, data in scan["images"].items():
                    self._check_stopped()
                    path = index.files.checked(directory / name)
                    self.require_paths(path)
                    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(descriptor, "wb") as handle:
                        handle.write(data)
                        handle.flush()
                        sync_file(handle.fileno())
                if scan["images"]:
                    sync_directory(directory)
                sync_directory(directory.parent)
                metadata = ResourceMetadata(
                    dataclass=type(self).__name__,
                    source=self.source,
                    description=self.description,
                    created_at=now,
                    modified_at=now,
                    files=[{"path": name} for name in ("dialogue.md", "dialogue.jsonl", *scan["images"])],
                )
                batch["resource"] = {
                    "path": directory.relative_to(index.files.workspace).as_posix(),
                    "metadata": metadata.model_dump(mode="json"),
                }
            self._check_stopped()
            index.files.write_json(index.pending_path, batch)
            return batch
        finally:
            if directory is not None and not index.pending_path.exists():
                shutil.rmtree(index.files.checked(directory))

    def _publish(self, batch):
        index = self._index()
        resource = batch["resource"]
        if resource is None:
            return []
        metadata = ResourceMetadata.model_validate(resource["metadata"])
        if metadata.source != self.source:
            raise ValueError("Unexpected Codex dialogue resource source")
        directory = index.files.path(resource["path"])
        manifest = directory / RESOURCE_METADATA_FILENAME
        paths = [manifest, *(directory / entry.path for entry in metadata.files)]
        self.require_paths(directory, *paths)
        for path in paths[1:]:
            path = index.files.checked(path)
            if not path.is_file() or not path.stat().st_size:
                raise ValueError("Codex dialogue payload is missing or empty")
        saved = index.files.read_json(manifest, None)
        if saved is not None and saved != resource["metadata"]:
            raise ValueError("Codex dialogue conflicts with its saved manifest")
        self._check_stopped()
        if saved is None:
            index.files.write_json(manifest, resource["metadata"])
        payloads = [path for path in paths[1:] if path.name != "dialogue.jsonl"]
        event = resource_event(index.files.workspace, directory, metadata, payloads)
        return [event]

    def _commit(self, batch):
        index = self._index()
        index.commit(index.load(self._now()), batch)
        index.files.remove(index.pending_path)

    async def execute(self):
        if not await connector_enabled(self, self.connector_job):
            self.context.response.answer = {"skipped": "paused"}
            return self.context.response
        async with await get_path_lock(self._authorize()):
            if not await connector_enabled(self, self.connector_job):
                self.context.response.answer = {"skipped": "paused"}
                return self.context.response
            batch = await self._offload(self._prepare)
            events = await self._offload(self._publish, batch)
            index = self._index()

            async def save(value):
                await self._offload(index.files.write_json, index.pending_path, value)

            if not await self.deliver_batch(events, batch, save):
                self.context.response.answer = {"skipped": "paused"}
                return self.context.response
            await self._offload(self._commit, batch)
        self.context.response.success = not batch["errors"]
        self.context.response.answer = {
            "event_ids": [event.id for event in events],
            "dialogue_messages": len(batch["message_ids"]),
            "scanned_bytes": batch["scanned_bytes"],
            "errors": batch["errors"],
        }
        if batch.get("rejected"):
            self.context.response.answer["rejected"] = batch["rejected"]
        return self.context.response
