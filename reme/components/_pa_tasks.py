"""User-owned task definitions, cumulative Markdown results and consumption receipts."""

# pylint: disable=protected-access

from __future__ import annotations

import asyncio
from datetime import date
import json
import os
from pathlib import Path
import re
from stat import S_ISREG
from typing import TYPE_CHECKING

import frontmatter

from ..constants import DEFAULT_MAX_FILE_BYTES
from ..schema.pa_task import PaTask, PaTaskRun, PaTaskRecovery
from ..schema.screen_capture import CapturedDisplay
from ..utils.async_utils import complete_in_thread
from ..utils.file_utils import get_path_lock

if TYPE_CHECKING:
    from .pa_runtime import PaRuntime


class TaskFiles:
    """Keep results authoritative in files; serialize publication and consumption."""

    def __init__(self, runtime: PaRuntime):
        self.runtime = runtime
        self.lock = asyncio.Lock()
        self.run_locks: dict[str, asyncio.Lock] = {}
        self.scheduled: dict[str, dict] = {}
        self.schedule_error: str | None = None

    def checked(self, path: Path) -> Path:
        """Reject workspace escapes and symlink aliases before any file operation."""
        resolved = path.resolve()
        if resolved != path.absolute() or not resolved.is_relative_to(self.runtime.workspace_path.resolve()):
            raise ValueError("Task files must stay inside the workspace without symlinks")
        return path

    @property
    def root_path(self) -> Path:
        """Keep each task's results, handoff and recovery together."""
        return self.runtime._metadata_file("cron")

    def task_path(self, task_id: str) -> Path:
        """Validate an identity before using it as a directory name."""
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", task_id):
            raise ValueError("Invalid task_id")
        return self.checked(self.root_path / task_id)

    def handoff_path(self, task_id: str) -> Path:
        """Return the latest internal handoff, separate from user results."""
        return self.checked(self.task_path(task_id) / "handoff.md")

    def recovery_path(self, task_id: str) -> Path:
        """Return durable progress for the currently unfinished execution."""
        return self.checked(self.task_path(task_id) / "recovery.json")

    def initialize(self) -> None:
        """Seed missing definitions once; never replace an existing user configuration."""
        legacy = self.checked(self.runtime.cron_path.with_name("tasks.json"))
        if not self.runtime.cron_path.exists() and legacy.exists():
            raise ValueError(
                "Legacy tasks.json requires offline migration: python -m scripts.migrate_pa_tasks WORKSPACE",
            )
        self.runtime._write_text(
            self.runtime.cron_path,
            json.dumps(
                [task.model_dump(mode="json") for task in self.runtime.default_tasks],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            create_only=True,
        )
        self.load()

    def read(self, path: Path) -> str:
        """Read a complete bounded UTF-8 document."""
        target = self.checked(path)
        if not S_ISREG(target.stat().st_mode):
            raise ValueError(f"Task file must be a regular file: {path}")
        with target.open("rb") as handle:
            content = handle.read(DEFAULT_MAX_FILE_BYTES + 1)
        if len(content) > DEFAULT_MAX_FILE_BYTES:
            raise ValueError(f"Task file exceeds the read limit: {path}")
        return content.decode("utf-8")

    @staticmethod
    def markdown(post: frontmatter.Post) -> str:
        """Serialize a complete task document within the same limit used when reading it."""
        content = frontmatter.dumps(post) + "\n"
        if len(content.encode("utf-8")) > DEFAULT_MAX_FILE_BYTES:
            raise ValueError("Task document exceeds the file read limit after metadata")
        return content

    def load(self) -> list[PaTask]:
        """Validate the complete configuration without changing it."""
        return self.parse(self.read(self.runtime.cron_path))

    @staticmethod
    def parse(content: str) -> list[PaTask]:
        """Use the same schema validation for runtime loading and offline migration."""
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("cron.json must contain a list of task definitions")
        tasks = [PaTask.model_validate(item) for item in data]
        if len({task.id for task in tasks}) != len(tasks):
            raise ValueError("cron.json contains duplicate task IDs")
        return tasks

    async def record(self, run: PaTaskRun) -> None:
        """Finish each append before allowing cancellation to leave its caller."""
        await complete_in_thread(self.runtime._append_jsonl, self.runtime.cron_runs_path, run.model_dump(mode="json"))
        if run.status == "running":
            self.runtime.start_task_activity(run)

    def paths(self, task_id: str) -> list[Path]:
        """Find published snapshots, newest first, across local calendar dates."""
        today = self.runtime._now().date()
        paths = []
        for directory in self.task_path(task_id).glob("????-??-??"):
            try:
                day = date.fromisoformat(directory.name)
            except ValueError:
                continue
            if day > today:
                continue
            root = self.checked(directory)
            for path in root.glob("*.md"):
                if re.fullmatch(r"(?:[01][0-9]|2[0-3])[0-5][0-9][0-5][0-9](?:-[0-9a-f]{32})?", path.stem):
                    paths.append(self.checked(path))
        return sorted(paths, key=lambda path: (path.parent.name, path.stem), reverse=True)

    def handoff(self, task_id: str):
        """Read the latest bounded internal handoff if one exists."""
        path = self.handoff_path(task_id)
        return frontmatter.loads(self.read(path)) if path.exists() else frontmatter.Post("")

    def recovery(self, task: PaTask) -> PaTaskRecovery | None:
        """Resume only the same definition; a user's newer request takes precedence."""
        path = self.recovery_path(task.id)
        if not path.exists():
            return None
        saved = PaTaskRecovery.model_validate_json(self.read(path))
        if saved.task != task:
            return None
        target = self.checked(self.runtime.workspace_path / saved.target_path)
        if target.parent.parent != self.task_path(task.id) or saved.run.task_id != task.id:
            raise ValueError("Task recovery identity does not match its directory")
        return saved

    async def checkpoint(self, saved: PaTaskRecovery) -> None:
        """Finish publishing phase progress before cancellation can escape."""
        path = self.recovery_path(saved.task.id)
        content = saved.model_dump_json(indent=2) + "\n"
        if len(content.encode("utf-8")) > DEFAULT_MAX_FILE_BYTES:
            raise ValueError("Task recovery exceeds the file read limit")
        async with await get_path_lock(path):
            await complete_in_thread(self.runtime._write_text, path, content)

    async def save_handoff(self, saved: PaTaskRecovery, content: str) -> None:
        """Atomically save a completed handoff and its successful screenshot baseline."""
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Task handoff must contain a nonempty summary")
        post = frontmatter.Post(content.strip(), task_id=saved.task.id, run_id=saved.run.run_id)
        if saved.prefetched is not None:
            post["screen_capture"] = saved.prefetched["frames"]
        payload = self.markdown(post)
        path = self.handoff_path(saved.task.id)
        async with await get_path_lock(path):
            await complete_in_thread(self.runtime._write_text, path, payload)

    async def clear_recovery(self, saved: PaTaskRecovery) -> None:
        """Remove only the checkpoint belonging to this completed execution."""
        path = self.recovery_path(saved.task.id)

        def clear():
            if path.exists() and PaTaskRecovery.model_validate_json(self.read(path)).run.run_id == saved.run.run_id:
                path.unlink()

        async with await get_path_lock(path):
            await complete_in_thread(clear)

    def screen_baseline(self, task_id: str) -> list[dict]:
        """Recover processed screenshots even when no user-facing result was published."""
        handoff = self.handoff(task_id)
        paths = self.paths(task_id)
        if "screen_capture" not in handoff and not paths:
            return []
        try:
            values = handoff.get("screen_capture")
            if values is None:
                values = frontmatter.loads(self.read(paths[0])).get("screen_capture", [])
            if not isinstance(values, list) or len(values) > 32:
                return []
            return [CapturedDisplay.model_validate(value).model_dump(mode="json") for value in values]
        except (OSError, ValueError):
            return []

    async def publish(
        self,
        staging: Path,
        target: Path,
        task_id: str,
        run_id: str,
        *,
        screen_capture: list[dict] | None = None,
    ) -> None:
        """Publish only a completed, nonempty snapshot, exclusively at its final name."""

        def publish():
            self.checked(target)
            post = frontmatter.loads(self.read(staging))
            if not post.content.strip():
                raise ValueError("Task agent did not write a nonempty summary")
            post.metadata.update(
                task_id=task_id,
                run_id=run_id,
                created_at=self.runtime._now().isoformat(),
                status="unread",
                user_read_at=None,
            )
            post.metadata.pop("screen_capture", None)
            if screen_capture is not None:
                post.metadata["screen_capture"] = [
                    CapturedDisplay.model_validate(value).model_dump(mode="json") for value in screen_capture
                ]
            content = self.markdown(post)
            self.runtime._write_text(staging, content)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(self.checked(staging), self.checked(target))

        async with self.lock:
            async with await get_path_lock(target):
                await complete_in_thread(publish)

    async def consume(self, limit: int, round_id: str) -> list[dict]:
        """Checkpoint full inputs before acknowledging read or superseded snapshots."""
        if type(limit) is not int or not 1 <= limit <= 3:
            raise ValueError("limit must be an integer from 1 to 3")

        def select():
            documents, updates = [], []
            for task in self.load():
                count = 0
                for path in self.paths(task.id):
                    text = self.read(path)
                    post = frontmatter.loads(text)
                    if post.get("status") != "unread":
                        continue
                    if post.get("task_id") != task.id:
                        raise ValueError(f"Task result identity does not match its directory: {path}")
                    if not post.content.strip():
                        raise ValueError(f"Empty task result: {path}")
                    status = "read" if count < limit else "superseded"
                    if status == "read":
                        documents.append(
                            {"task_id": task.id, "path": self.runtime.to_workspace_relative(path), "content": text},
                        )
                    updates.append((path, text, status))
                    count += 1
            return documents, updates

        def acknowledge(path, original, status):
            # Preserve concurrent user edits for a later round instead of overwriting them.
            if self.read(path) != original:
                return
            post = frontmatter.loads(original)
            post["status"] = status
            self.runtime._write_text(self.checked(path), self.markdown(post))

        async with self.lock:
            documents, updates = await asyncio.to_thread(select)
            await self.runtime.checkpoint_proactive(round_id, "task_results", documents)
            for path, original, status in updates:
                async with await get_path_lock(path):
                    await complete_in_thread(acknowledge, path, original, status)
            return [{"task_id": document["task_id"], "path": document["path"]} for document in documents]
