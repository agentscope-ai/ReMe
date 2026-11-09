"""User-owned task definitions, cumulative Markdown results and consumption receipts."""

# pylint: disable=protected-access

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING

import frontmatter
import yaml

from ...constants import DEFAULT_MAX_FILE_BYTES
from ...schema.pa_task import PaTask, PaTaskRun, PaTaskState, PaTaskCompletion
from ...utils.async_utils import complete_in_thread
from ...utils.file_utils import get_path_lock

if TYPE_CHECKING:
    from .runtime import PaRuntime


@dataclass
class TaskExecution:
    """One active run, independent of its task's persistent native session."""

    state: PaTaskState
    history_version: tuple[int, int, int] | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class TaskFiles:
    """Keep results authoritative in files; serialize publication and consumption."""

    def __init__(self, runtime: PaRuntime):
        self.runtime = runtime
        self.lock = asyncio.Lock()
        self.run_locks: dict[str, asyncio.Lock] = {}
        self.scheduled: dict[str, dict] = {}
        self.executions: dict[str, TaskExecution] = {}

    @property
    def root_path(self) -> Path:
        """Keep task results and persistent execution state together."""
        return self.runtime.files.metadata_path("cron")

    def task_path(self, task_id: str) -> Path:
        """Validate an identity before using it as a directory name."""
        return self.runtime.files.identity_path(self.root_path, task_id)

    def state_path(self, task_id: str) -> Path:
        """Keep the session reference and execution checkpoint beside its results."""
        return self.runtime.files.checked(self.task_path(task_id) / "state.json")

    def initialize(self) -> None:
        """Seed missing definitions once; never replace an existing user configuration."""
        legacy = self.runtime.files.checked(self.runtime.cron_path.with_name("tasks.json"))
        if not self.runtime.cron_path.exists() and legacy.exists():
            raise ValueError(
                "Legacy tasks.json requires offline migration: python -m scripts.migrate_pa_tasks WORKSPACE",
            )
        self.runtime.files.write_text(
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

    def load(self) -> list[PaTask]:
        """Validate the complete configuration without changing it."""
        return self.parse(self.runtime.files.read(self.runtime.cron_path))

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
        await complete_in_thread(
            self.runtime.files.append_jsonl,
            self.runtime.cron_runs_path,
            run.model_dump(mode="json"),
        )
        if run.status == "running":
            self.runtime.event_stream.start_task_activity(run)

    def result(self, path: Path, content: str | None = None) -> frontmatter.Post:
        """Validate a result without repairing or rewriting the source document."""
        try:
            post = frontmatter.loads(self.runtime.files.read(path) if content is None else content)
        except (yaml.YAMLError, TypeError) as exc:
            raise ValueError("Invalid task result frontmatter") from exc
        if post.get("task_id") != path.parent.parent.name or not post.content.strip():
            raise ValueError("Invalid task result identity or body")
        return post

    def _result_warning(self, path: Path, error: Exception, warnings: list[str] | None) -> None:
        message = f"Task result {self.runtime.to_workspace_relative(path)}: {error}"
        self.runtime.logger.warning(message)
        if warnings is not None:
            warnings.append(message)

    def paths(self, task_id: str) -> list[Path]:
        """Find candidate paths, newest first; results() validates their contents."""
        today = self.runtime.now().date()
        paths = []
        for directory in self.task_path(task_id).glob("????-??-??"):
            try:
                day = date.fromisoformat(directory.name)
            except ValueError:
                continue
            if day > today:
                continue
            root = self.runtime.files.checked(directory)
            for path in root.glob("*.md"):
                if re.fullmatch(r"(?:[01][0-9]|2[0-3])[0-5][0-9][0-5][0-9](?:[0-9]{6}|-[0-9a-f]{32})?", path.stem):
                    paths.append(self.runtime.files.checked(path))
        return sorted(paths, key=lambda path: (path.parent.name, path.stem), reverse=True)

    def results(
        self,
        task_id: str,
        *,
        warnings: list[str] | None = None,
    ) -> Iterator[tuple[Path, str, frontmatter.Post]]:
        """Read each valid result once, retaining exact source text for checkpoints."""
        for path in self.paths(task_id):
            try:
                content = self.runtime.files.read(path)
                post = self.result(path, content)
            except (OSError, ValueError) as exc:
                self._result_warning(path, exc, warnings)
                continue
            yield path, content, post

    def documents(self, *, warnings: list[str]) -> Iterator[tuple[Path, str, frontmatter.Post]]:
        """Enumerate published results once for all read-only projections."""
        if not self.root_path.exists():
            return
        for root in sorted(self.root_path.iterdir()):
            if not root.is_dir():
                continue
            try:
                yield from self.results(root.name, warnings=warnings)
            except (OSError, ValueError) as exc:
                warnings.append(f"Task {root.name}: {exc}")

    def state(self, task: PaTask) -> PaTaskState:
        """Load a task's native session; discard work superseded by a new definition."""
        path = self.state_path(task.id)
        state = (
            PaTaskState.model_validate_json(self.runtime.files.read(path)) if path.exists() else PaTaskState(task=task)
        )
        if state.task.id != task.id:
            raise ValueError("Task state belongs to another task")
        if state.task != task:
            state = PaTaskState(task=task, session_id=state.session_id)
        return state

    async def save(self, state: PaTaskState) -> None:
        """Atomically checkpoint the session and completed side effects."""
        content = state.model_dump_json(indent=2) + "\n"
        if len(content.encode("utf-8")) > DEFAULT_MAX_FILE_BYTES:
            raise ValueError("Task state exceeds the file read limit")
        path = self.state_path(state.task.id)
        async with await get_path_lock(path):
            await complete_in_thread(self.runtime.files.write_text, path, content)

    def next_start(self, task_id: str) -> datetime:
        """Keep start times distinct even within one clock tick or after a clock rollback."""
        started_at = self.runtime.now()
        path = self.state_path(task_id)
        if path.exists():
            previous = PaTaskState.model_validate_json(self.runtime.files.read(path)).run
            if previous is not None:
                started_at = max(started_at, previous.started_at + timedelta(microseconds=1))
        while self.result_path(task_id, started_at).exists():
            started_at += timedelta(microseconds=1)
        return started_at

    def result_path(self, task_id: str, started_at: datetime) -> Path:
        """Name each snapshot with its task-local date and time, including microseconds."""
        return self.runtime.files.checked(self.task_path(task_id) / f"{started_at:%Y-%m-%d}/{started_at:%H%M%S%f}.md")

    def staging_path(self, run: PaTaskRun) -> Path:
        """Keep a recoverable draft outside the published result list."""
        return self.runtime.files.checked(
            self.task_path(run.task_id) / ".pending" / f"{run.started_at:%Y%m%d-%H%M%S%f}.md",
        )

    def staged_summary(self, run: PaTaskRun) -> str:
        """Accept a complete Markdown body; ReMe supplies authoritative metadata."""
        path = self.staging_path(run)
        if not path.exists():
            return ""
        summary = frontmatter.loads(self.runtime.files.read(path)).content.strip()
        if not summary:
            raise ValueError("Cron summary document must not be empty")
        return summary

    @contextmanager
    def execution_scope(self, state: PaTaskState):
        """Expose the task's current execution only while its serialized worker is active."""
        task_id = state.task.id
        if task_id in self.executions:
            raise ValueError("Cron task is already executing")
        self.executions[task_id] = TaskExecution(state)
        try:
            yield
        finally:
            self.executions.pop(task_id, None)

    def execution(self, task_id: str, state: PaTaskState | None = None) -> TaskExecution:
        """Resolve server-bound tasks and reject revoked or superseded in-flight work."""
        current = self.executions.get(task_id)
        if current is None or (state is not None and current.state is not state):
            raise ValueError("Cron tools require the task's current execution")
        return current

    def _check_definition(self, state: PaTaskState) -> None:
        if state.task not in self.load():
            raise ValueError("Cron task changed or was cancelled")

    async def complete(self, task_id: str, reminder: str = "") -> dict:
        """Publish the task query's draft before optionally announcing it."""
        current = self.execution(task_id)
        async with current.lock:
            state = current.state
            self.execution(task_id, state)
            self._check_definition(state)
            if not isinstance(reminder, str):
                raise ValueError("Task reminder must be a string")
            summary = (
                state.completion.summary
                if state.completion is not None
                else await asyncio.to_thread(self.staged_summary, state.run)
            )
            self.execution(task_id, state)
            reminder = reminder.strip()
            if reminder and (not summary or not state.run.target_session_id):
                raise ValueError("Cron reminder requires a result and a target session")
            if state.completion is not None and state.run.status == "succeeded":
                if state.completion.reminder != reminder:
                    raise ValueError("This run already submitted a different reminder")
                return self.answer(state)
            if reminder:
                self.runtime.reminders.check_history(current.history_version)
            state.completion = PaTaskCompletion(
                summary=summary,
                reminder=reminder,
                history_version=current.history_version,
            )
            await self.save(state)
            return await self.finish(state)

    @staticmethod
    def answer(state: PaTaskState) -> dict:
        """Return publication and submission facts without repeating report contents."""
        run = state.run
        return {
            "task_id": state.task.id,
            "started_at": run.started_at.isoformat(),
            "status": run.status,
            "path": run.result_path,
            "message_id": run.message_id,
            "reason": run.reason,
        }

    async def finish(self, state: PaTaskState) -> dict:
        """Resume a saved submission without another research or summarization turn."""
        self._check_definition(state)
        run, completion = state.run, state.completion
        if completion.summary:
            target = (
                self.runtime.files.checked(self.runtime.workspace_path / run.result_path)
                if run.result_path
                else self.result_path(state.task.id, run.started_at)
            )
            if not target.is_relative_to(self.task_path(state.task.id)):
                raise ValueError("Cron results must stay inside their task directory")
            async with self.lock:
                async with await get_path_lock(target):
                    await complete_in_thread(self._publish, target, state)
            run.result_path = self.runtime.to_workspace_relative(target)
        else:
            run.reason = "no_new_information"
        await self.save(state)
        if completion.reminder:
            result = await self.runtime.reminders.submit_immediate(
                f"cron_{run.task_id}_{run.started_at.isoformat()}",
                completion.reminder,
                session_id=run.target_session_id,
                task_id=run.task_id,
                task_definition=state.task.model_dump(mode="json"),
                cron_state=state,
                result_path=run.result_path,
                history_version=completion.history_version,
                expires_at=self.runtime.now() + timedelta(minutes=15),
            )
            if result["status"] == "discarded":
                raise ValueError("Cron task changed before presentation")
            run.message_id = result["id"]
        run.status, run.error, run.finished_at = "succeeded", None, self.runtime.now()
        await self.save(state)
        return self.answer(state)

    def _publish(self, target: Path, state: PaTaskState) -> None:
        """Own metadata and preserve an existing publication across retry or user reads."""
        self.runtime.files.checked(target)
        run = state.run
        if not target.exists():
            post = frontmatter.Post(
                state.completion.summary,
                task_id=state.task.id,
                started_at=run.started_at.isoformat(),
                created_at=self.runtime.now().isoformat(),
                status="unread",
                user_read_at=None,
            )
            self.runtime.files.write_text(target, self.runtime.files.markdown(post), create_only=True)
        post = self.result(target)
        if post.get("started_at") != run.started_at.isoformat() or post.content != state.completion.summary:
            raise FileExistsError(f"Task snapshot already exists: {target}")

    async def consume(self, limit: int, round_id: str, *, warnings: list[str] | None = None) -> list[dict]:
        """Checkpoint full inputs before acknowledging read or superseded snapshots."""
        if type(limit) is not int or not 1 <= limit <= 3:
            raise ValueError("limit must be an integer from 1 to 3")

        def select():
            documents, updates = [], []
            for task in self.load():
                count = 0
                for path, text, post in self.results(task.id, warnings=warnings):
                    if post.get("status") != "unread":
                        continue
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
            if self.runtime.files.read(path) != original:
                return
            post = frontmatter.loads(original)
            post["status"] = status
            self.runtime.files.write_text(self.runtime.files.checked(path), self.runtime.files.markdown(post))

        async with self.lock:
            documents, updates = await asyncio.to_thread(select)
            await self.runtime.checkpoint_proactive(round_id, "task_results", documents)
            for path, original, status in updates:
                async with await get_path_lock(path):
                    await complete_in_thread(acknowledge, path, original, status)
            return [{"task_id": document["task_id"], "path": document["path"]} for document in documents]
