"""Collect new Codex dialogue and deliver a retryable observation to a named Agent."""

import asyncio
from contextlib import suppress
from datetime import datetime
from hashlib import sha256
import os
from pathlib import Path
import shutil
from threading import Event as ThreadEvent

from ....schema.resource_metadata import RESOURCE_METADATA_FILENAME
from ._codex_dialogue import CodexDialogueIndex
from .base_connector import resource_package_id
from ..agent import BaseAgentStep
from ....utils.files.path_lock import get_path_lock
from ....core.registry import R
from ....utils.files.workspace_files import WorkspaceFiles
from ....schema import ApplicationConfig, Event, ResourceMetadata
from ....schema.proactive_runtime import validate_agent_name
from ....utils.async_utils import finish_cleanup
from ....utils.files.file_sync import sync_directory
from ....utils.timezone_utils import resolve_timezone


class _CodexDialogueBase(BaseAgentStep):
    """Own cancellation and serialize source progress without a permanent watcher."""

    def __init__(self, codex_home=None, agent_name="work", max_scan_bytes=32 * 1024 * 1024, **kwargs):
        super().__init__(**kwargs)
        if type(max_scan_bytes) is not int or max_scan_bytes < 1:
            raise ValueError("max_scan_bytes must be a positive integer")
        self.codex_home = Path(codex_home or os.environ.get("CODEX_HOME") or Path.home() / ".codex")
        self.agent_name = validate_agent_name(agent_name)
        self.max_scan_bytes = max_scan_bytes
        self.cancelled = ThreadEvent()

    def _now(self):
        config = self.app_context.app_config if self.app_context is not None else ApplicationConfig()
        return datetime.now(resolve_timezone(config.timezone))

    def _check_stopped(self):
        if self.cancelled.is_set() or (self.context.stop_event is not None and self.context.stop_event.is_set()):
            raise RuntimeError("Codex dialogue collection stopped")

    def _index(self):
        files = WorkspaceFiles(self.workspace_path)
        directory = files.path(self.config_value("resource_dir")) / "codex_dialogue"
        return CodexDialogueIndex(files, directory, self.codex_home, self._check_stopped)

    def _authorize(self):
        files = WorkspaceFiles(self.workspace_path)
        directory = files.path(self.config_value("resource_dir"))
        if directory == files.workspace:
            raise ValueError("resource_dir must be a workspace subdirectory")
        self.require_paths(directory, files.path(self.config_value("reme_agent_dir")))
        return directory / "codex_dialogue"

    async def _offload(self, function, *args):
        task = asyncio.create_task(asyncio.to_thread(function, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            self.cancelled.set()
            with suppress(Exception):
                await finish_cleanup(task)
            raise


@R.register("codex_dialogue_init_step")
class CodexDialogueInitStep(_CodexDialogueBase):
    """Establish the startup baseline without importing existing dialogue."""

    def _initialize(self):
        state = self._index().load(self._now())
        return {"indexed_files": len(state["files"])}

    async def execute(self):
        async with await get_path_lock(self._authorize()):
            self.context.response.answer = await self._offload(self._initialize)
        return self.context.response


@R.register("codex_dialogue_step")
class CodexDialogueStep(_CodexDialogueBase):
    """Commit reading progress only after a frozen dialogue package has been delivered."""

    description = (
        "New Codex user inputs and final answers, grouped by project and turn. "
        "Read dialogue.jsonl for original message times and text; unfinished turns are not completed work."
    )

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
            "agent_name": self.agent_name,
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
                index.files.write_jsonl(directory / "dialogue.jsonl", scan["messages"])
                sync_directory(directory.parent)
                metadata = ResourceMetadata(
                    dataclass=type(self).__name__,
                    source="codex_dialogue",
                    description=self.description,
                    created_at=now,
                    modified_at=now,
                    files=[{"path": "dialogue.jsonl"}],
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
        if batch["agent_name"] != self.agent_name:
            raise ValueError("Pending Codex dialogue does not match its Agent")
        resource = batch["resource"]
        if resource is None:
            return []
        metadata = ResourceMetadata.model_validate(resource["metadata"])
        if metadata.source != "codex_dialogue":
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
        event = Event(
            id=f"resource-{sha256(resource['path'].encode()).hexdigest()}",
            event_ts=metadata.created_at,
            source=metadata.source,
            description=metadata.description,
            paths=[path.relative_to(index.files.workspace).as_posix() for path in paths],
        )
        return [event.model_dump(mode="json")]

    def _commit(self, batch):
        index = self._index()
        index.commit(index.load(self._now()), batch)
        index.files.remove(index.pending_path)

    async def execute(self):
        directory = self._authorize()
        if self.get_job("append_task_events") is None:
            raise RuntimeError("Codex dialogue requires the append_task_events Job")
        await self.proactive_runtime.definition(self.agent_name)
        async with await get_path_lock(directory):
            batch = await self._offload(self._prepare)
            events = await self._offload(self._publish, batch)
            if events:
                response = await self.run_job(
                    "append_task_events",
                    agent_name=self.agent_name,
                    events=events,
                    _allowed_paths=self.context.get("_allowed_paths"),
                )
                if not response.success:
                    raise RuntimeError(f"Codex dialogue delivery failed: {response.answer}")
            await self._offload(self._commit, batch)
        self.context.response.success = not batch["errors"]
        self.context.response.answer = {
            "event_ids": [event["id"] for event in events],
            "dialogue_messages": len(batch["message_ids"]),
            "scanned_bytes": batch["scanned_bytes"],
            "errors": batch["errors"],
        }
        return self.context.response
