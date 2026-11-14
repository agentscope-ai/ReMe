"""Collect screenshots and Codex dialogue together, then deliver one retryable event batch."""

import asyncio
from contextlib import suppress
from datetime import datetime
from hashlib import sha256
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from threading import Event as ThreadEvent
from uuid import uuid4
from zoneinfo import ZoneInfo

from ._codex_dialogue import CodexDialogueIndex
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
        directory = files.path(self.config_value("metadata_dir")) / "work_observations" / self.agent_id
        return CodexDialogueIndex(files, directory, self.codex_home, self._check_stopped)

    def _authorize(self):
        files = AgentFiles(self.workspace_path)
        directory = files.path(self.config_value("metadata_dir")) / "work_observations" / self.agent_id
        if files.path(self.config_value("resource_dir")) == files.workspace:
            raise ValueError("resource_dir must be a workspace subdirectory")
        self.require_paths(
            directory,
            files.path(self.config_value("resource_dir")),
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
    """Stage source payloads and their next cursors atomically before exposing any event."""

    def _package(self, stage, source, description, entries, now, package_id):
        index = self._index()
        folder = stage / source
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
        index.files.write(folder / "metadata.json", metadata.model_dump_json(indent=2) + "\n")
        sync_directory(folder)
        target = index.files.path(self.config_value("resource_dir")) / source / now.date().isoformat() / package_id
        index.files.checked(target)
        self.require_paths(target)
        relative = target.relative_to(index.files.workspace).as_posix()
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
        pending = index.directory / "pending"
        transaction = index.files.read_json(pending / "batch.json", None)
        if transaction is not None:
            return transaction
        if pending.exists():
            if any(pending.iterdir()):
                raise ValueError("Work observation pending directory is incomplete")
            pending.rmdir()
        scan = index.scan(state, self.max_scan_bytes, self.max_file_bytes())
        package_id = f"{now:%H%M%S}_{uuid4().hex[:12]}"
        index.directory.mkdir(parents=True, exist_ok=True)
        index.files.checked(index.directory)
        with TemporaryDirectory(prefix=".collect-", dir=index.directory) as temporary:
            stage = Path(temporary)
            resources, events = [], []
            errors = dict(scan["errors"])
            screenshot = stage / "screenshot"
            screenshot.mkdir()
            try:
                capture = ScreenCaptureStep(app_context=self.app_context)
                entries = capture.collect(screenshot)
                resource, event = self._package(
                    stage,
                    "screenshot",
                    capture.description,
                    entries,
                    now,
                    package_id,
                )
                resources.append(resource)
                events.append(event.model_dump(mode="json"))
            except Exception as exc:
                errors["screenshot"] = str(exc)
                shutil.rmtree(screenshot)
            self._check_stopped()
            if scan["messages"]:
                dialogue = stage / "codex_dialogue"
                dialogue.mkdir()
                index.files.write_jsonl(dialogue / "dialogue.jsonl", scan["messages"])
                resource, event = self._package(
                    stage,
                    "codex_dialogue",
                    "New Codex user inputs and final answers, grouped by project and turn. "
                    "Read dialogue.jsonl for original message times and text; unfinished turns are not completed work.",
                    [{"path": "dialogue.jsonl"}],
                    now,
                    package_id,
                )
                resources.append(resource)
                events.append(event.model_dump(mode="json"))
            transaction = {
                "id": package_id,
                "agent_id": self.agent_id,
                **{key: scan[key] for key in ("updates", "message_ids", "next_path", "scanned_bytes")},
                "resources": resources,
                "events": events,
                "errors": errors,
            }
            index.files.write_json(stage / "batch.json", transaction)
            self._check_stopped()
            stage.rename(pending)
            sync_directory(index.directory)
        return transaction

    def _publish(self, transaction):
        """An interrupted multi-resource rename resumes from the same staged batch."""
        index = self._index()
        if transaction["agent_id"] != self.agent_id:
            raise ValueError("Work observation batch belongs to a different Agent")
        events = [Event.model_validate(value) for value in transaction["events"]]
        for event in events:
            self.require_paths(*(index.files.path(path) for path in event.paths))
        for resource in transaction["resources"]:
            self._check_stopped()
            source = resource["source"]
            if source not in {"screenshot", "codex_dialogue"}:
                raise ValueError("Invalid work observation source")
            staged = index.files.checked(index.directory / "pending" / source)
            target = index.files.path(resource["path"])
            self.require_paths(target)
            if not target.exists():
                index.files.checked(target.parent)
                target.parent.mkdir(parents=True, exist_ok=True)
                index.files.checked(target.parent)
                staged.rename(target)
                sync_directory(target.parent)
                sync_directory(staged.parent)
            metadata = ResourceMetadata.model_validate_json(index.files.read(target / "metadata.json"))
            if metadata.model_dump(mode="json") != resource["metadata"]:
                raise ValueError("Work resource conflicts with its staged manifest")
            for entry in metadata.files:
                path = index.files.checked(target / entry.path)
                if not path.is_file() or not path.stat().st_size:
                    raise ValueError("Work resource payload is missing or empty")
        return [event.model_dump(mode="json") for event in events]

    def _commit(self, transaction):
        index = self._index()
        state = index.load(self._now())
        index.commit(state, transaction)
        pending = index.directory / "pending"
        index.files.checked(pending / "batch.json").unlink()
        pending.rmdir()
        sync_directory(index.directory)

    async def execute(self):
        directory = self._authorize()
        if self.get_job("append_event") is None:
            raise RuntimeError("Work observations require the append_event Job")
        if (await self.agent_runtime.definition(self.agent_id)).trigger_type != "event":
            raise ValueError("Work observations require an event Agent")
        async with await get_path_lock(directory):
            transaction = await self._offload(self._prepare)
            events = await self._offload(self._publish, transaction)
            if events:
                response = await self.run_job(
                    "append_event",
                    agent_id=self.agent_id,
                    events=events,
                    _allowed_paths=self.context.get("_allowed_paths"),
                )
                if not response.success:
                    raise RuntimeError(f"Work event delivery failed: {response.answer}")
            await self._offload(self._commit, transaction)
        self.context.response.success = not transaction["errors"]
        self.context.response.answer = {
            "event_ids": [event["id"] for event in events],
            "dialogue_messages": len(transaction["message_ids"]),
            "scanned_bytes": transaction["scanned_bytes"],
            "errors": transaction["errors"],
        }
        return self.context.response
