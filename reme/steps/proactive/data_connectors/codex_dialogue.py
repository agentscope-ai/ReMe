"""Collect new Codex dialogue and deliver retryable observations to background tasks."""

import asyncio
from contextlib import suppress
from datetime import datetime
from itertools import groupby
import os
from pathlib import Path
import shutil
from threading import Event as ThreadEvent

from ....schema.resource_metadata import RESOURCE_METADATA_FILENAME
from ._codex_dialogue import CodexDialogueIndex
from .base_connector import resource_package_id
from .resource_delivery import resource_event
from ..agent import BaseAgentStep
from ....utils.files.path_lock import get_path_lock
from ....core.registry import R
from ....utils.files.workspace_files import WorkspaceFiles
from ....schema import ApplicationConfig, ResourceMetadata
from ....schema.proactive_agent import validate_task_names
from ....utils.async_utils import finish_cleanup
from ....utils.files.file_sync import sync_directory, sync_file
from ....utils.timezone_utils import resolve_timezone


def dialogue_markdown(messages, now, language):
    """Render messages for reading, keeping stable identifiers for evidence references."""
    chinese = language.lower().startswith("zh")
    title, session, turn, user, progress, final = (
        ("Codex 对话", "会话", "轮次", "用户", "Codex 进度", "Codex 最终回答")
        if chinese
        else ("Codex dialogue", "Session", "Turn", "User", "Codex progress", "Codex final answer")
    )
    sections = [f"# {title}"]
    for (cwd, session_id), session_messages in groupby(messages, key=lambda row: (row["cwd"], row["session_id"])):
        sections.append(f"## `{cwd}`\n\n{session} `{session_id}`")
        for turn_id, turn_messages in groupby(session_messages, key=lambda row: row["turn_id"]):
            sections.append(f"### {turn} `{turn_id}`")
            for row in turn_messages:
                timestamp = datetime.fromisoformat(row["timestamp"]).astimezone(now.tzinfo)
                role = {None: user, "commentary": progress, "final_answer": final}[row.get("phase")]
                sections.append(f"#### {role} · {timestamp:%Y-%m-%d %H:%M:%S %z} · `{row['message_id']}`")
                if row["content"]:
                    sections.append(row["content"].rstrip("\n"))
                images = row.get("image_paths", ())
                for filename in images:
                    number = Path(filename).stem.rsplit("-", 1)[-1]
                    sections.append(f"![Image #{number}]({filename})")
                if missing := row["image_attachments"] - len(images):
                    sections.append(f"{missing} 张图片无法读取。" if chinese else f"Unavailable images: {missing}.")
    return "\n\n".join(sections) + "\n"


class _CodexDialogueBase(BaseAgentStep):
    """Own cancellation and serialize source progress without a permanent watcher."""

    def __init__(self, codex_home=None, task_names=None, max_scan_bytes=32 * 1024 * 1024, **kwargs):
        super().__init__(**kwargs)
        if type(max_scan_bytes) is not int or max_scan_bytes < 1:
            raise ValueError("max_scan_bytes must be a positive integer")
        self.codex_home = Path(codex_home or os.environ.get("CODEX_HOME") or Path.home() / ".codex")
        self.task_names = validate_task_names(["proactive_memory"] if task_names is None else task_names)
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
        "New Codex user messages, progress updates and final answers. "
        "Image links are relative to the dialogue file. Progress updates alone do not establish completed work."
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
                index.files.write(directory / "dialogue.md", dialogue_markdown(scan["messages"], now, self.language))
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
                    source="codex_dialogue",
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
        payloads = [path for path in paths[1:] if path.name != "dialogue.jsonl"]
        event = resource_event(index.files.workspace, directory, metadata, payloads)
        return [event]

    def _commit(self, batch):
        index = self._index()
        index.commit(index.load(self._now()), batch)
        index.files.remove(index.pending_path)

    async def execute(self):
        directory = self._authorize()
        async with await get_path_lock(directory):
            batch = await self._offload(self._prepare)
            events = await self._offload(self._publish, batch)
            targets = validate_task_names(batch["task_names"])
            rejected = batch.get("rejected", {})
            pending = [name for name in targets if name not in batch.get("delivered_to", ()) and name not in rejected]
            if events and pending:
                result = await self.proactive_manager.deliver_events(task_names=pending, events=events)
                delivered, skipped = result["deliveries"], result["rejected"]
                if skipped:
                    self.logger.warning(f"Codex dialogue batch {batch['id']} rejected by tasks: {skipped}")
                    rejected.update(skipped)
                    batch["rejected"] = rejected
                if delivered or skipped:
                    batch.setdefault("delivered_to", []).extend(delivered)
                    index = self._index()
                    await self._offload(index.files.write_json, index.pending_path, batch)
                if result["errors"]:
                    raise RuntimeError(f"Codex dialogue delivery failed: {result['errors']}")
            await self._offload(self._commit, batch)
        self.context.response.success = not batch["errors"]
        self.context.response.answer = {
            "event_ids": [event.id for event in events],
            "dialogue_messages": len(batch["message_ids"]),
            "scanned_bytes": batch["scanned_bytes"],
            "errors": batch["errors"],
        }
        if rejected:
            self.context.response.answer["rejected"] = rejected
        return self.context.response
