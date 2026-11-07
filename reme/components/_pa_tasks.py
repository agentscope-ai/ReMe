"""Cron-slot reservation and task execution owned by PaRuntime."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from ..constants import DEFAULT_MAX_FILE_BYTES
from ..schema.pa_task import PaTask, PaTaskRun
from ..schema.response import Response
from ..utils.async_utils import complete_in_thread

if TYPE_CHECKING:
    from .pa_runtime import PaRuntime


class TaskScheduler:
    """Own task serialization; definitions and history remain workspace files."""

    def __init__(self, runtime: PaRuntime):
        self.runtime = runtime
        self.lock = asyncio.Lock()

    def load(self) -> list[PaTask]:
        try:
            with self.runtime.tasks_path.open("rb") as handle:
                content = handle.read(DEFAULT_MAX_FILE_BYTES + 1)
        except FileNotFoundError:
            data = [task.model_dump(mode="json") for task in self.runtime.default_tasks]
            changed = True
        else:
            if len(content) > DEFAULT_MAX_FILE_BYTES:
                raise ValueError("tasks.json exceeds the file read limit")
            data = json.loads(content)
            changed = False
        if not isinstance(data, list):
            raise ValueError("tasks.json must contain a list of task definitions")
        tasks = [PaTask.model_validate(item) for item in data]
        if len({task.id for task in tasks}) != len(tasks):
            raise ValueError("tasks.json contains duplicate task IDs")
        now = self.runtime._now()
        for raw, task in zip(data, tasks):
            if "id" not in raw:
                changed = True
            if task.created_at is None:
                task.created_at = now
                changed = True
        if changed:
            self.runtime._write_json(self.runtime.tasks_path, [task.model_dump(mode="json") for task in tasks])
        return tasks

    def latest(self, task_ids: set[str]) -> dict[str, PaTaskRun]:
        if not task_ids:
            return {}
        try:
            descriptor = os.open(self.runtime.task_runs_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            return {}
        latest = {}
        with os.fdopen(descriptor, "rb") as handle:
            for line in self.runtime._reverse_jsonl_lines(handle):
                run = PaTaskRun.model_validate_json(line)
                if run.task_id in task_ids and run.task_id not in latest:
                    latest[run.task_id] = run
                if latest.keys() >= task_ids:
                    break
        return latest

    async def run(
        self, runner: Callable[..., Awaitable[Response]], *, stop_event: asyncio.Event | None = None,
    ) -> list[dict]:
        """Reserve each due slot before serial execution; coalesce missed slots to the latest one."""
        from croniter import croniter

        async with self.lock:
            tasks = await complete_in_thread(self.load)
            latest = await asyncio.to_thread(self.latest, {task.id for task in tasks})
            for previous in latest.values():
                if previous.status == "running":
                    previous.status = "interrupted"
                    previous.finished_at = self.runtime._now()
                    previous.error = "Previous execution has no completion record; outcome unknown"
                    await complete_in_thread(self.runtime._append_jsonl, self.runtime.task_runs_path, previous.model_dump(mode="json"))
            results = []
            for task in tasks:
                if stop_event is not None and stop_event.is_set():
                    break
                if not task.enabled:
                    continue
                now = self.runtime._now()
                scheduled_at = croniter(task.cron, now + timedelta(microseconds=1)).get_prev(datetime)
                previous = latest.get(task.id)
                if scheduled_at <= task.created_at or (previous is not None and previous.scheduled_at >= scheduled_at):
                    continue
                run = PaTaskRun(
                    task_id=task.id, cron=task.cron, query=task.query,
                    scheduled_at=scheduled_at, started_at=now,
                )
                await complete_in_thread(self.runtime._append_jsonl, self.runtime.task_runs_path, run.model_dump(mode="json"))
                try:
                    response = await runner(query=task.query, stop_event=stop_event)
                    run.status = "succeeded" if response.success else "failed"
                    run.result = response.answer
                    run.agent_session_id = response.metadata.get("session_id")
                    if not response.success:
                        run.error = str(response.answer)[:1000]
                except asyncio.CancelledError:
                    run.status = "interrupted"
                    run.error = "Execution cancelled; outcome may be incomplete"
                    raise
                except Exception as exc:
                    run.status = "failed"
                    run.error = str(exc)[:1000]
                finally:
                    run.finished_at = self.runtime._now()
                    await complete_in_thread(self.runtime._append_jsonl, self.runtime.task_runs_path, run.model_dump(mode="json"))
                latest[task.id] = run
                results.append(run.model_dump(mode="json"))
            return results
