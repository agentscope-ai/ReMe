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
from ..schema.pa_task import PaTask, PaTaskRun
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

    def checked(self, path: Path) -> Path:
        """Reject workspace escapes and symlink aliases before any file operation."""
        resolved = path.resolve()
        if resolved != path.absolute() or not resolved.is_relative_to(self.runtime.workspace_path.resolve()):
            raise ValueError("Task files must stay inside the workspace without symlinks")
        return path

    @property
    def daily_path(self) -> Path:
        """Honor the configured daily directory."""
        return self.checked(self.runtime.workspace_path / self.runtime.app_context.app_config.daily_dir)

    def initialize(self) -> None:
        """Seed missing definitions once; never replace an existing user configuration."""
        self.runtime._write_text(
            self.runtime.tasks_path,
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

    def load(self) -> list[PaTask]:
        """Validate the complete configuration without changing it."""
        data = json.loads(self.read(self.runtime.tasks_path))
        if not isinstance(data, list):
            raise ValueError("tasks.json must contain a list of task definitions")
        tasks = [PaTask.model_validate(item) for item in data]
        if len({task.id for task in tasks}) != len(tasks):
            raise ValueError("tasks.json contains duplicate task IDs")
        return tasks

    async def record(self, run: PaTaskRun) -> None:
        """Finish each append before allowing cancellation to leave its caller."""
        await complete_in_thread(self.runtime._append_jsonl, self.runtime.task_runs_path, run.model_dump(mode="json"))
        if run.status == "running":
            self.runtime.start_task_activity(run)

    def paths(self, task_id: str) -> list[Path]:
        """Find published snapshots, newest first, across local calendar dates."""
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", task_id):
            raise ValueError("Invalid task_id")
        today = self.runtime._now().date()
        paths = []
        for directory in self.daily_path.glob("????-??-??"):
            try:
                day = date.fromisoformat(directory.name)
            except ValueError:
                continue
            if day > today:
                continue
            root = self.checked(directory / "cron" / task_id)
            for path in root.glob("*.md"):
                if re.fullmatch(r"(?:[01][0-9]|2[0-3])[0-5][0-9][0-5][0-9]", path.stem):
                    paths.append(self.checked(path))
        return sorted(paths, key=lambda path: (path.parents[2].name, path.name), reverse=True)

    def screen_baseline(self, task_id: str) -> list[dict]:
        """Recover only images attached by code to the latest successful task summary."""
        paths = self.paths(task_id)
        if not paths:
            return []
        try:
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
            )
            post.metadata.pop("screen_capture", None)
            if screen_capture is not None:
                post.metadata["screen_capture"] = [
                    CapturedDisplay.model_validate(value).model_dump(mode="json") for value in screen_capture
                ]
            content = frontmatter.dumps(post) + "\n"
            if len(content.encode("utf-8")) > DEFAULT_MAX_FILE_BYTES:
                raise ValueError("Task summary exceeds the file read limit after adding metadata")
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
            self.runtime._write_text(self.checked(path), frontmatter.dumps(post) + "\n")

        async with self.lock:
            documents, updates = await asyncio.to_thread(select)
            await self.runtime.checkpoint_proactive(round_id, "task_results", documents)
            for path, original, status in updates:
                async with await get_path_lock(path):
                    await complete_in_thread(acknowledge, path, original, status)
            return [{"task_id": document["task_id"], "path": document["path"]} for document in documents]
