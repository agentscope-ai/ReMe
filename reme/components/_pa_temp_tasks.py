"""Persistent one-shot tasks; runtime workers own execution, never the creation tool."""

# pylint: disable=protected-access

import asyncio
from datetime import datetime, timedelta
from hashlib import sha256
import json
import math

import frontmatter

from ..constants import DEFAULT_MAX_FILE_BYTES
from ..schema.pa_temp_task import PaTempTask
from ..utils.async_utils import complete_in_thread


class TempTasks:
    """Serialize task definition changes and reject stale worker revisions."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.lock = asyncio.Lock()

    @property
    def path(self):
        """Locate the durable one-shot definitions and execution state."""
        return self.runtime._metadata_file("temp_tasks.json")

    @property
    def root_path(self):
        """Keep one-shot documents beside the cron results under PA ownership."""
        return self.runtime._metadata_file("temporary")

    def initialize(self):
        """Seed absent state and recover completion from announced deliveries."""
        self.runtime._write_text(self.path, "[]\n", create_only=True)
        records = self.load()
        deliveries = {record.id: record for record in self.runtime.reminders.read() if record.delivery == "immediate"}
        changed = False
        for task in records:
            delivery = deliveries.get(f"temp_{task.id}_{task.revision}")
            if (
                delivery is not None
                and task.enabled
                and delivery.status != "discarded"
                and task.status not in {"completed", "cancelled"}
            ):
                task.status, task.message_id = "completed", delivery.id
                task.finished_at = delivery.created_at
                changed = True
        if changed:
            self._save(records)

    def load(self):
        """Validate task identities and ownership of all document references."""
        values = json.loads(self.runtime.tasks.read(self.path))
        if not isinstance(values, list):
            raise ValueError("temp_tasks.json must contain a list")
        records = [PaTempTask.model_validate(value) for value in values]
        if len({record.id for record in records}) != len(records):
            raise ValueError("Duplicate temporary task IDs")
        for record in records:
            for field, name in (("handoff_path", "handoff.md"), ("result_path", "result.md")):
                value = getattr(record, field)
                if value is not None and self.runtime.tasks.checked(
                    self.runtime.workspace_path / value,
                ) != self.document_path(record.id, name):
                    raise ValueError("Temporary task document belongs to a different task")
        return records

    def get(self, task_id):
        """Read one task by stable identity without changing it."""
        return next((record for record in self.load() if record.id == task_id), None)

    def document_path(self, task_id, name):
        """Resolve an allowed task document inside the one-shot data directory."""
        if name not in {"handoff.md", "result.md"}:
            raise ValueError("Unknown temporary task document")
        # Keep cron identities and temporary task documents in separate directories.
        self.runtime.tasks.task_path(task_id)
        return self.runtime.tasks.checked(self.root_path / task_id / name)

    async def recover_text(self, task, name):
        """Recover a published document whose progress update was interrupted."""
        path = self.document_path(task.id, name)
        if not path.exists():
            return task
        post = frontmatter.loads(await asyncio.to_thread(self.runtime.tasks.read, path))
        if post.get("task_id") != task.id or post.get("revision") != task.revision or not post.content.strip():
            return task
        changes = {"handoff_path" if name == "handoff.md" else "result_path": self.runtime.to_workspace_relative(path)}
        changes["status"] = "scheduled" if name == "handoff.md" else "sending"
        if name == "result.md":
            changes["agent_session_id"] = post.get("agent_session_id")
        return await self.patch(task.id, task.revision, **changes)

    def _save(self, records):
        text = json.dumps([record.model_dump(mode="json") for record in records], ensure_ascii=False, indent=2) + "\n"
        if len(text.encode("utf-8")) > DEFAULT_MAX_FILE_BYTES:
            raise ValueError("Temporary task state exceeds the file read limit")
        self.runtime._write_text(self.path, text)

    def deadline(self, run_at=None, delay_minutes: float | None = None) -> datetime:
        """Resolve one timezone-aware deadline from an absolute time or delay."""
        if run_at is not None and delay_minutes is not None:
            raise ValueError("Specify either run_at or delay_minutes")
        if run_at is not None:
            value = datetime.fromisoformat(run_at.replace("Z", "+00:00")) if isinstance(run_at, str) else run_at
            if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("run_at must include a timezone")
            return value
        delay = 0.0 if delay_minutes is None else delay_minutes
        if type(delay) not in (int, float) or not math.isfinite(delay) or delay < 0:
            raise ValueError("delay_minutes must be a non-negative finite number")
        return self.runtime._now() + timedelta(minutes=float(delay))

    async def create(self, query, *, run_at=None, delay_minutes: float | None = None, request_id=None):
        """Register one Main-bound task with retry-safe request identity."""
        source = self.runtime.session_id
        if not source:
            raise ValueError("Create a temporary task from an existing Main conversation")
        now = self.runtime._now()
        deadline = self.deadline(run_at, delay_minutes)
        delay_minutes = float(delay_minutes) if delay_minutes is not None else None
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        signature = sha256(
            json.dumps([source, query.strip(), str(run_at), delay_minutes], ensure_ascii=False).encode(),
        ).hexdigest()
        activity = self.runtime._activities.get("chat")
        if request_id is None and activity is not None:
            key = json.dumps([source, activity.turn_id, query, str(run_at), delay_minutes], ensure_ascii=False)
            request_id = sha256(key.encode()).hexdigest()
        async with self.lock:
            records = await asyncio.to_thread(self.load)
            existing = next((record for record in records if request_id and record.request_id == request_id), None)
            if existing is not None:
                if existing.request_signature != signature:
                    raise ValueError("request_id already belongs to another task")
                return existing
            record = PaTempTask(
                query=query,
                run_at=deadline,
                source_session_id=source,
                created_at=now,
                updated_at=now,
                request_id=request_id,
                request_signature=signature,
            )
            records.append(record)
            await complete_in_thread(self._save, records)
            return record

    async def manage(self, task_id, action, *, query=None, run_at=None, delay_minutes: float | None = None):
        """Revise a task while invalidating workers bound to its previous revision."""
        if action not in {"update", "pause", "resume", "cancel"}:
            raise ValueError("action must be update, pause, resume or cancel")
        if action != "update" and any(value is not None for value in (query, run_at, delay_minutes)):
            raise ValueError("Only update accepts query or time changes")
        if action == "update" and all(value is None for value in (query, run_at, delay_minutes)):
            raise ValueError("update requires query or a new execution time")
        async with self.lock:
            records = await asyncio.to_thread(self.load)
            record = next((value for value in records if value.id == task_id), None)
            if record is None:
                raise ValueError("Unknown temporary task")
            if record.status == "cancelled":
                if action == "cancel":
                    return record
                raise ValueError("A completed or cancelled task cannot be restarted; create a new task")
            if record.status == "completed" and action != "cancel":
                raise ValueError("A completed task cannot be restarted; create a new task")
            if action == "pause" and record.status == "paused":
                return record
            if action == "resume" and record.enabled and record.status != "failed":
                return record
            data = record.model_dump()
            data.update(revision=record.revision + 1, updated_at=self.runtime._now(), error=None, retry_at=None)
            if action == "cancel":
                data.update(enabled=False, status="cancelled", finished_at=record.finished_at or self.runtime._now())
            elif action == "pause":
                data.update(enabled=False, status="paused")
            else:
                if query is not None:
                    data.update(
                        query=query,
                        handoff_path=None,
                        result_path=None,
                        agent_session_id=None,
                        message_id=None,
                        started_at=None,
                        finished_at=None,
                        delayed=False,
                    )
                if run_at is not None or delay_minutes is not None:
                    data["run_at"] = self.deadline(run_at, delay_minutes)
                    data.update(result_path=None, started_at=None, finished_at=None, delayed=False)
                # A new revision gets a fresh native tool binding; retain completed results, not an old lease.
                data.update(
                    enabled=True,
                    status="scheduled" if data["handoff_path"] else "pending",
                    agent_session_id=None,
                )
            updated = PaTempTask.model_validate(data)
            records = [updated if value.id == task_id else value for value in records]
            await complete_in_thread(self._save, records)
            await self.runtime.announce_messages()
            return updated

    async def patch(self, task_id, revision, **changes):
        """Save worker progress only while its revision remains active."""
        async with self.lock:
            records = await asyncio.to_thread(self.load)
            current = next((value for value in records if value.id == task_id), None)
            if (
                current is None
                or current.revision != revision
                or not current.enabled
                or current.status in {"cancelled", "completed"}
            ):
                raise asyncio.CancelledError()
            updated = PaTempTask.model_validate({**current.model_dump(), **changes, "updated_at": self.runtime._now()})
            await complete_in_thread(self._save, [updated if value.id == task_id else value for value in records])
            return updated

    async def save_text(self, task_id, revision, name, text, **changes):
        """Publish a bounded task document before checkpointing its state."""
        if name not in {"handoff.md", "result.md"} or not isinstance(text, str) or not text.strip():
            raise ValueError("Temporary task documents must contain a nonempty body")
        if len(text.encode("utf-8")) > DEFAULT_MAX_FILE_BYTES:
            raise ValueError("Temporary task document exceeds the file read limit")
        async with self.lock:
            records = await asyncio.to_thread(self.load)
            current = next((value for value in records if value.id == task_id), None)
            if (
                current is None
                or current.revision != revision
                or not current.enabled
                or current.status in {"completed", "cancelled"}
            ):
                raise asyncio.CancelledError()
            return await self._save_text(records, current, name, text, **changes)

    async def _save_text(self, records, task, name, text, **changes):
        """Write the document before its progress checkpoint while holding the task lock."""
        path = self.document_path(task.id, name)
        post = frontmatter.Post(
            text.strip(),
            task_id=task.id,
            revision=task.revision,
            agent_session_id=changes.get("agent_session_id"),
        )
        payload = self.runtime.tasks.markdown(post)
        await complete_in_thread(self.runtime._write_text, path, payload)
        field = "handoff_path" if name == "handoff.md" else "result_path"
        updated = PaTempTask.model_validate(
            {
                **task.model_dump(),
                **changes,
                field: self.runtime.to_workspace_relative(path),
                "updated_at": self.runtime._now(),
            },
        )
        await complete_in_thread(self._save, [updated if value.id == task.id else value for value in records])
        return updated

    async def send(self, task_id, revision, content):
        """Checkpoint and deliver the active task's result without another model turn."""
        if not isinstance(content, str) or not content.strip() or len(content.strip()) > 3980:
            raise ValueError("Temporary result must contain 1-3980 characters")
        content = content.strip()
        async with self.lock:
            records = await asyncio.to_thread(self.load)
            task = next((value for value in records if value.id == task_id), None)
            if task is None or task.revision != revision or not task.enabled:
                raise ValueError("Chat message requires an active temporary task")
            if task.status == "completed" and task.message_id:
                return await self._submit(task, content)
            if task.status not in {"running", "sending"}:
                raise ValueError("Chat message requires a running temporary task")
            if task.result_path is None:
                task = await self._save_text(records, task, "result.md", content, status="sending")
            else:
                saved = await asyncio.to_thread(self.runtime.tasks.read, self.runtime.workspace_path / task.result_path)
                if frontmatter.loads(saved).content != content:
                    raise ValueError("Temporary task already saved a different result")
            result = await self._submit(task, content)
            if result["status"] == "discarded":
                raise ValueError("Temporary task changed before presentation")
            updated = task.model_copy(
                update={
                    "message_id": result["id"],
                    "status": "completed",
                    "finished_at": self.runtime._now(),
                    "updated_at": self.runtime._now(),
                },
            )
            await complete_in_thread(self._save, [updated if value.id == task_id else value for value in records])
            return result

    async def _submit(self, task, content):
        if task.delayed and isinstance(content, str):
            label = "【已延迟】" if self.runtime.app_context.app_config.language == "zh" else "[Delayed]"
            if not content.startswith(label):
                content = label + "\n" + content
        return await self.runtime.reminders.submit_immediate(
            f"temp_{task.id}_{task.revision}",
            content,
            session_id=task.source_session_id,
            task_id=task.id,
            task_revision=task.revision,
        )
