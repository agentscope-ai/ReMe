"""Collect screenshots and Codex dialogue together, then deliver one retryable event batch."""

import asyncio
from contextlib import suppress
from datetime import datetime
from hashlib import sha256
import os
from pathlib import Path
import shutil
from threading import Event as ThreadEvent
from zoneinfo import ZoneInfo

from ._codex_dialogue import CodexDialogueIndex
from .base_connector import resource_package_id
from .screen_capture import ScreenCaptureStep
from ..agent import BaseAgentStep
from ...file_io import get_path_lock
from ....components import R
from ....components.agent_runtime._internal.files import AgentFiles
from ....schema import Event, ResourceMetadata
from ....schema.agent_runtime import validate_agent_id
from ....utils.async_utils import finish_cleanup
from ....utils.file_sync import sync_directory, sync_file


class _WorkObservationBase(BaseAgentStep):
    """Use invocation-owned cancellation and a shared path lock, without a permanent watcher."""

    def __init__(self, codex_home=None, agent_id="work", max_scan_bytes=32 * 1024 * 1024, **kwargs):
        super().__init__(**kwargs)
        if type(max_scan_bytes) is not int or max_scan_bytes < 1:
            raise ValueError("max_scan_bytes must be a positive integer")
        self.codex_home = Path(codex_home or os.environ.get("CODEX_HOME") or Path.home() / ".codex")
        self.agent_id = validate_agent_id(agent_id)
        self.max_scan_bytes = max_scan_bytes
        self.cancelled = ThreadEvent()

    def _now(self):
        timezone = self.config_value("timezone")
        return datetime.now(ZoneInfo(timezone)) if timezone else datetime.now().astimezone()

    def _check_stopped(self):
        if self.cancelled.is_set() or (self.context.stop_event is not None and self.context.stop_event.is_set()):
            raise RuntimeError("Work observation collection stopped")

    def _index(self):
        files = AgentFiles(self.workspace_path)
        directory = files.path(self.config_value("resource_dir")) / "codex_dialogue"
        return CodexDialogueIndex(files, directory, self.codex_home, self._check_stopped)

    def _authorize(self):
        files = AgentFiles(self.workspace_path)
        directory = files.path(self.config_value("resource_dir"))
        if directory == files.workspace:
            raise ValueError("resource_dir must be a workspace subdirectory")
        self.require_paths(
            directory,
            files.path(self.config_value("reme_agent_dir")),
        )
        return directory

    async def _offload(self, function, *args):
        task = asyncio.create_task(asyncio.to_thread(function, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            self.cancelled.set()
            with suppress(Exception):
                await finish_cleanup(task)
            raise


@R.register("work_observation_init_step")
class WorkObservationInitStep(_WorkObservationBase):
    """Establish the baseline at application startup, before the first five-minute collection."""

    def _initialize(self):
        index = self._index()
        state = index.load(self._now())
        return {"indexed_files": len(state["files"])}

    async def execute(self):
        directory = self._authorize()
        async with await get_path_lock(directory):
            self.context.response.answer = await self._offload(self._initialize)
        return self.context.response


@R.register("work_observation_step")
class WorkObservationStep(_WorkObservationBase):
    """Keep payloads, collection progress and unfinished deliveries under their own sources."""

    def _pending_paths(self, files):
        root = files.path(self.config_value("resource_dir"))
        return {source: root / source / "pending.json" for source in ("screenshot", "codex_dialogue")}

    def _resource_directory(self, source, now, package_id):
        files = self._index().files
        directory = files.path(self.config_value("resource_dir")) / source / now.date().isoformat() / package_id
        self.require_paths(directory)
        files.checked(directory.parent).mkdir(parents=True, exist_ok=True)
        files.checked(directory).mkdir()
        return directory

    def _package(self, folder, source, description, entries, now):
        index = self._index()
        metadata = ResourceMetadata(
            dataclass=type(self).__name__,
            source=source,
            description=description,
            created_at=now,
            modified_at=now,
            files=entries,
        )
        for item in metadata.files:
            path = folder / item.path
            if path.resolve() != path or not path.is_file() or not path.stat().st_size:
                raise ValueError("Work resource payload is missing, empty, or symlinked")
            with path.open("rb") as handle:
                sync_file(handle.fileno())
        sync_directory(folder)
        sync_directory(folder.parent)
        relative = folder.relative_to(index.files.workspace).as_posix()
        event = Event(
            id=f"resource-{sha256(relative.encode()).hexdigest()}",
            event_ts=now,
            source=source,
            description=description,
            paths=[f"{relative}/metadata.json", *(f"{relative}/{entry.path}" for entry in metadata.files)],
        )
        return {"source": source, "path": relative, "metadata": metadata.model_dump(mode="json")}, event

    def _prepare(self):
        index = self._index()
        now = self._now()
        state = index.load(now)
        pending = self._pending_paths(index.files)
        sources = {
            source: batch
            for source, path in pending.items()
            if (batch := index.files.read_json(path, None)) is not None
        }
        if sources:
            return sources
        scan = index.scan(state, self.max_scan_bytes, self.max_file_bytes())
        package_id = resource_package_id(now)
        directories = {}
        try:
            sources = {
                source: {"id": package_id, "agent_id": self.agent_id, "resources": [], "events": [], "errors": {}}
                for source in pending
            }
            dialogue_batch = sources["codex_dialogue"]
            dialogue_batch.update({key: scan[key] for key in ("updates", "message_ids", "next_path", "scanned_bytes")})
            dialogue_batch["errors"] = dict(scan["errors"])
            screenshot = self._resource_directory("screenshot", now, package_id)
            directories["screenshot"] = screenshot
            try:
                capture = ScreenCaptureStep(app_context=self.app_context)
                entries = capture.collect(screenshot)
                resource, event = self._package(
                    screenshot,
                    "screenshot",
                    capture.description,
                    entries,
                    now,
                )
                sources["screenshot"]["resources"].append(resource)
                sources["screenshot"]["events"].append(event.model_dump(mode="json"))
            except Exception as exc:
                sources["screenshot"]["errors"]["screenshot"] = str(exc)
                shutil.rmtree(screenshot)
            self._check_stopped()
            if scan["messages"]:
                dialogue = self._resource_directory("codex_dialogue", now, package_id)
                directories["codex_dialogue"] = dialogue
                index.files.write_jsonl(dialogue / "dialogue.jsonl", scan["messages"])
                resource, event = self._package(
                    dialogue,
                    "codex_dialogue",
                    "New Codex user inputs and final answers, grouped by project and turn. "
                    "Read dialogue.jsonl for original message times and text; unfinished turns are not completed work.",
                    [{"path": "dialogue.jsonl"}],
                    now,
                )
                dialogue_batch["resources"].append(resource)
                dialogue_batch["events"].append(event.model_dump(mode="json"))
            self._check_stopped()
            for source, batch in sources.items():
                index.files.write_json(pending[source], batch)
            return sources
        finally:
            for source, directory in directories.items():
                if not pending[source].exists() and directory.exists():
                    shutil.rmtree(index.files.checked(directory))

    def _publish(self, sources):
        """Finish source-owned manifests before submitting their events."""
        index = self._index()
        events = [Event.model_validate(value) for batch in sources.values() for value in batch["events"]]
        for event in events:
            self.require_paths(*(index.files.path(path) for path in event.paths))
        for source, batch in sources.items():
            if batch["agent_id"] != self.agent_id:
                raise ValueError("Pending resources do not match their Agent")
            for resource in batch["resources"]:
                self._check_stopped()
                if resource["source"] != source:
                    raise ValueError("Pending resources do not match their source")
                target = index.files.path(resource["path"])
                self.require_paths(target)
                metadata = ResourceMetadata.model_validate(resource["metadata"])
                for entry in metadata.files:
                    path = index.files.checked(target / entry.path)
                    if not path.is_file() or not path.stat().st_size:
                        raise ValueError("Work resource payload is missing or empty")
                manifest = target / "metadata.json"
                saved = index.files.read_json(manifest, None)
                if saved is not None and saved != resource["metadata"]:
                    raise ValueError("Work resource conflicts with its saved manifest")
                if saved is None:
                    index.files.write_json(manifest, resource["metadata"])
        return [event.model_dump(mode="json") for event in events]

    def _commit(self, sources):
        index = self._index()
        pending = self._pending_paths(index.files)
        if "codex_dialogue" in sources:
            index.commit(index.load(self._now()), sources["codex_dialogue"])
        for source in ("codex_dialogue", "screenshot"):
            if source in sources:
                index.files.checked(pending[source]).unlink()
                sync_directory(pending[source].parent)

    async def execute(self):
        directory = self._authorize()
        if self.get_job("append_event") is None:
            raise RuntimeError("Work observations require the append_event Job")
        if (await self.agent_runtime.definition(self.agent_id)).trigger_type != "event":
            raise ValueError("Work observations require an event Agent")
        async with await get_path_lock(directory):
            sources = await self._offload(self._prepare)
            events = await self._offload(self._publish, sources)
            if events:
                response = await self.run_job(
                    "append_event",
                    agent_id=self.agent_id,
                    events=events,
                    _allowed_paths=self.context.get("_allowed_paths"),
                )
                if not response.success:
                    raise RuntimeError(f"Work event delivery failed: {response.answer}")
            await self._offload(self._commit, sources)
        dialogue = sources.get("codex_dialogue", {})
        errors = {key: value for batch in sources.values() for key, value in batch["errors"].items()}
        self.context.response.success = not errors
        self.context.response.answer = {
            "event_ids": [event["id"] for event in events],
            "dialogue_messages": len(dialogue.get("message_ids", [])),
            "scanned_bytes": dialogue.get("scanned_bytes", 0),
            "errors": errors,
        }
        return self.context.response
