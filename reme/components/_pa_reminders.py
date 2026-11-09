"""Reminder state and deferred delivery owned by PaRuntime."""

# pylint: disable=protected-access

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta
import re
from typing import TYPE_CHECKING

from .codex_app_server import SessionChangedError
from ..schema.pa_proactive import PaNotificationWindow, PaProactiveConfig
from ..schema.reminder import ReminderKind, ReminderRecord
from ..utils.async_utils import complete_in_thread

if TYPE_CHECKING:
    from .pa_runtime import PaRuntime

_GREETINGS = {
    "morning": re.compile(r"早安|早上好|\bgood\s+morning\b", re.IGNORECASE),
    "midday": re.compile(r"午安|中午好|\bgood\s+afternoon\b", re.IGNORECASE),
    "evening": re.compile(r"晚安|晚上好|\bgood\s*(?:night|evening)\b|\bnighty[ -]?night\b", re.IGNORECASE),
}


class ReminderDelivery:
    """Serialize latest-decision updates and delivery without owning agent rounds."""

    def __init__(self, runtime: PaRuntime):
        self.runtime = runtime
        self.lock = asyncio.Lock()
        self.pending: ReminderRecord | None = None
        self.idle: tuple[str, float] | None = None

    @staticmethod
    def _window(config: PaProactiveConfig, now: datetime, records: list[ReminderRecord]) -> PaNotificationWindow:
        start, end = config.active_window(now)
        closing = next(
            (record for record in records if record.kind == "evening" and start <= record.created_at < end),
            None,
        )
        return PaNotificationWindow(
            starts_at=start,
            ends_at=end,
            closing_message_id=f"reme-proactive-{closing.id}" if closing is not None else None,
            enabled=config.enabled,
        )

    async def _refresh_window(self, config: PaProactiveConfig | None = None) -> PaNotificationWindow:
        config = config or await self.runtime.get_proactive_config()
        records = await asyncio.to_thread(self.read)
        window = self._window(config, self.runtime._now(), records)
        self.runtime.notifications = window
        self.runtime.publish_status()
        return window

    async def window(self, config: PaProactiveConfig) -> PaNotificationWindow:
        """Project the current window from the journal, serialized with decisions."""
        async with self.lock:
            return await self._refresh_window(config)

    def read(self) -> list[ReminderRecord]:
        """Read the latest state of each reminder, newest creation first."""
        latest = {}
        try:
            descriptor = os.open(self.runtime.chat_inbox_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            return []
        with os.fdopen(descriptor, "rb") as handle:
            for line in self.runtime._reverse_jsonl_lines(handle):
                record = ReminderRecord.model_validate_json(line)
                latest.setdefault(record.id, record)
        return sorted(latest.values(), key=lambda record: record.created_at, reverse=True)

    def restore(self) -> None:
        """Restore the newest pending reminder and discard superseded ones."""
        pending = [record for record in self.read() if record.status == "pending" and record.delivery == "idle"]
        self.pending = pending[0] if pending else None
        self.idle = None
        for record in pending[1:]:
            record = record.model_copy(
                update={
                    "status": "discarded",
                    "reason": "superseded",
                    "updated_at": self.runtime._now(),
                },
            )
            self.runtime._append_jsonl(self.runtime.chat_inbox_path, record.model_dump(mode="json"))

    @staticmethod
    def answer(record: ReminderRecord) -> dict:
        """Expose persisted state together with its delivery outcome."""
        return {"sent": record.status == "sent", **record.model_dump(mode="json")}

    def history_version(self) -> tuple[int, int, int]:
        """Identify a journal snapshot while the caller holds the delivery lock."""
        try:
            stat = self.runtime.chat_inbox_path.stat()
            return stat.st_ino, stat.st_size, stat.st_mtime_ns
        except FileNotFoundError:
            return 0, 0, 0

    def check_history(self, version) -> None:
        """Reject stale or missing history before making a new decision durable."""
        if version is None or version != self.history_version():
            raise ValueError("Reminder history changed or was not read; call get_chat_messages again")

    async def _retry(self, record: ReminderRecord, error: OSError) -> None:
        record = record.model_copy(
            update={
                "retry_at": self.runtime._now() + timedelta(seconds=5),
                "updated_at": self.runtime._now(),
                "error": str(error)[:1000],
            },
        )
        await complete_in_thread(
            self.runtime._append_jsonl,
            self.runtime.chat_inbox_path,
            record.model_dump(mode="json"),
        )
        if self.pending is not None and self.pending.id == record.id:
            self.pending = record

    async def _finish(self, status: str, reason: str | None = None, error: str | None = None) -> ReminderRecord:
        record = self.pending
        now = self.runtime._now()
        record = record.model_copy(
            update={
                "status": status,
                "reason": reason,
                "error": error,
                "updated_at": now,
                "sent_at": now if status == "sent" else None,
                "retry_at": None,
            },
        )
        await complete_in_thread(
            self.runtime._append_jsonl,
            self.runtime.chat_inbox_path,
            record.model_dump(mode="json"),
        )
        self.pending = None
        self.idle = None
        await self._refresh_window()
        await self.runtime.announce_messages()
        return record

    async def submit(self, content: str, kind: ReminderKind, *, session_id: str | None, message_id: str) -> dict:
        """Persist a validated round decision; the caller holds the submission lock."""
        # pylint: disable=too-many-return-statements
        records = await asyncio.to_thread(self.read)
        existing = next((record for record in records if record.id == message_id), None)
        if existing is not None:
            if existing.content != content or existing.kind != kind:
                raise ValueError("message_id already belongs to different content")
            return self.answer(existing)
        if not content:
            if self.pending is not None and self.pending.session_id == session_id:
                await self._finish("discarded", "withdrawn")
            return {"id": message_id, "sent": False, "status": "skipped", "reason": "empty_content"}
        feedback_version = self.runtime.current_round.feedback_version
        if feedback_version != await asyncio.to_thread(self.runtime.inbox.version):
            if self.pending is not None:
                await self._finish("discarded", "feedback_changed")
            return {"id": message_id, "sent": False, "status": "skipped", "reason": "feedback_changed"}
        config = await self.runtime.get_proactive_config()
        now = self.runtime._now()
        if not config.enabled:
            return {"id": message_id, "sent": False, "status": "skipped", "reason": "disabled"}
        window = self._window(config, now, records)
        if not config.is_active(now):
            return {"id": message_id, "sent": False, "status": "skipped", "reason": "outside_active_hours"}
        if window.closing_message_id is not None:
            return {"id": message_id, "sent": False, "status": "skipped", "reason": "day_closed"}
        if kind != "normal":
            if any(
                record.kind == kind and record.created_at.astimezone(now.tzinfo).date() == now.date()
                for record in records
            ):
                return {"id": message_id, "sent": False, "status": "skipped", "reason": "period_used"}
            if self.runtime.current_round.hint_period != kind or self.runtime.hints._period(config, now) != kind:
                raise ValueError("A special reminder requires this round's matching sampled hint")
        if any(pattern.search(content) for period, pattern in _GREETINGS.items() if period != kind):
            raise ValueError("Greeting wording requires the matching special reminder kind")
        record = ReminderRecord(
            id=message_id,
            content=content,
            kind=kind,
            created_at=now,
            updated_at=now,
            expires_at=min(now + timedelta(minutes=config.interval_minutes), window.ends_at),
            session_id=session_id,
            reason="waiting_for_idle",
            feedback_version=feedback_version,
        )
        previous = self.pending
        await complete_in_thread(
            self.runtime._append_jsonl,
            self.runtime.chat_inbox_path,
            record.model_dump(mode="json"),
        )
        self.pending = record
        self.idle = None
        await self._refresh_window(config)
        if previous is not None:
            previous = previous.model_copy(
                update={
                    "status": "discarded",
                    "reason": "superseded",
                    "updated_at": now,
                },
            )
            await complete_in_thread(
                self.runtime._append_jsonl,
                self.runtime.chat_inbox_path,
                previous.model_dump(mode="json"),
            )
        result = await self._deliver()
        return self.answer(result or record)

    async def deliver_pending(self) -> None:
        """Serialize an idle check for the current pending reminder."""
        async with self.lock:
            records = await asyncio.to_thread(self.read)
            for record in reversed(records):
                if (
                    record.delivery == "immediate"
                    and record.status == "pending"
                    and (record.retry_at is None or record.retry_at <= self.runtime._now())
                ):
                    try:
                        await self._deliver_immediate(record)
                    except (OSError, ValueError) as exc:
                        self.runtime.logger.warning(f"Immediate delivery {record.id} is waiting for valid state: {exc}")
            if self.pending is not None:
                await self._deliver()

    @staticmethod
    def message(record):
        """Project a delivery record into the shared chat-message shape."""
        from ..schema.agent_session import SessionMessage

        return SessionMessage(
            id=f"reme-proactive-{record.id}",
            role="assistant",
            source="proactive",
            content=record.content,
            created_at=record.created_at.isoformat(),
            expires_at=record.expires_at.isoformat() if record.expires_at else None,
            delivery=record.delivery,
            origin_session_id=record.session_id,
        )

    def _task_state(self):
        try:
            cron = {task.id: task.model_dump(mode="json") for task in self.runtime.tasks.load() if task.enabled}
        except (OSError, ValueError):
            cron = self.runtime.tasks.scheduled.copy()
        try:
            temporary = {task.id: task for task in self.runtime.temp_tasks.load()}
        except (OSError, ValueError):
            temporary = {}
        return cron, temporary

    @staticmethod
    def _matches_task(record, cron, temporary):
        if record.task_definition is not None:
            return cron.get(record.task_id) == record.task_definition
        task = temporary.get(record.task_id)
        return (
            task is not None and task.enabled and task.revision == record.task_revision and task.status != "cancelled"
        )

    def inbox_messages(self):
        """Project messages whose delivery and owning task still permit display."""
        records = [
            record
            for record in self.read()
            if record.status == "sent"
            or (record.delivery == "immediate" and record.announced_at is not None and record.status != "discarded")
        ][:100]
        cron, temporary = self._task_state()
        try:
            version = self.runtime.inbox.version()
        except (OSError, ValueError):
            version = None
        messages = []
        for record in reversed(records):
            eligible = (
                self._matches_task(record, cron, temporary)
                if record.delivery == "immediate"
                else (version is not None and record.feedback_version == version)
            )
            eligible = eligible and (record.expires_at is None or self.runtime._now() < record.expires_at)
            messages.append(self.message(record).model_copy(update={"notification_eligible": eligible}))
        return messages

    async def submit_immediate(
        self,
        identifier,
        content,
        *,
        session_id,
        task_id,
        task_revision=None,
        task_definition=None,
        cron_run_id=None,
        result_path=None,
        history_version=None,
        expires_at=None,
    ):
        """Persist a retry-safe result; presentation precedes native insertion."""
        if not isinstance(content, str) or not content.strip() or len(content.strip()) > 4000:
            raise ValueError("Immediate content must contain 1-4000 characters")
        async with self.lock:
            existing = next((record for record in await asyncio.to_thread(self.read) if record.id == identifier), None)
            if existing is not None:
                if existing.content != content.strip() or existing.session_id != session_id:
                    raise ValueError("This task already submitted different chat content")
                return self.answer(existing)
            if task_definition is not None:
                self.runtime.tasks.message_context(cron_run_id)
                self.check_history(history_version)
            now = self.runtime._now()
            record = ReminderRecord(
                id=identifier,
                content=content.strip(),
                kind="normal",
                delivery="immediate",
                task_id=task_id,
                task_revision=task_revision,
                task_definition=task_definition,
                result_path=result_path,
                expires_at=expires_at,
                created_at=now,
                updated_at=now,
                session_id=session_id,
                reason="waiting_for_session_write",
            )
            await complete_in_thread(
                self.runtime._append_jsonl,
                self.runtime.chat_inbox_path,
                record.model_dump(mode="json"),
            )
            record = await self._announce_immediate(record)
            return self.answer(record)

    async def _immediate_active(self, record):
        cron, temporary = await asyncio.to_thread(self._task_state)
        return self._matches_task(record, cron, temporary)

    async def _announce_immediate(self, record):
        active = await self._immediate_active(record)
        if not active or record.announced_at is None:
            record = record.model_copy(
                update={
                    "status": record.status if active else "discarded",
                    "reason": record.reason if active else "task_changed",
                    "announced_at": self.runtime._now() if active else record.announced_at,
                    "updated_at": self.runtime._now(),
                },
            )
            await complete_in_thread(
                self.runtime._append_jsonl,
                self.runtime.chat_inbox_path,
                record.model_dump(mode="json"),
            )
        await self.runtime.announce_messages()
        return record

    async def _deliver_immediate(self, record):
        if record.expires_at is not None and self.runtime._now() >= record.expires_at:
            expired = record.model_copy(
                update={"status": "discarded", "reason": "expired", "updated_at": self.runtime._now()},
            )
            await complete_in_thread(
                self.runtime._append_jsonl,
                self.runtime.chat_inbox_path,
                expired.model_dump(mode="json"),
            )
            await self.runtime.announce_messages()
            return
        if record.announced_at is None or not await self._immediate_active(record):
            record = await self._announce_immediate(record)
            if record.status == "discarded":
                return
        if record.session_id == self.runtime.session_id and self.runtime.session_lock.locked():
            return
        try:
            await self.runtime.agent_wrapper.append_session(
                record.session_id,
                record.content,
                message_id=record.id,
                timeout=0.5,
                delivery="immediate",
            )
        except (SessionChangedError, TimeoutError):
            return
        except OSError as exc:
            await self._retry(record, exc)
            return
        except Exception as exc:
            record = record.model_copy(
                update={"status": "failed", "reason": "session_write_failed", "error": str(exc)[:1000]},
            )
        else:
            record = record.model_copy(
                update={
                    "status": "sent",
                    "sent_at": self.runtime._now(),
                    "reason": None,
                    "retry_at": None,
                    "error": None,
                },
            )
        record = record.model_copy(update={"updated_at": self.runtime._now()})
        await complete_in_thread(
            self.runtime._append_jsonl,
            self.runtime.chat_inbox_path,
            record.model_dump(mode="json"),
        )

    async def _deliver(self) -> ReminderRecord | None:
        # Terminal-state guard clauses keep the delivery rules explicit.
        # pylint: disable=too-many-return-statements
        record = self.pending
        if record is None:
            return None
        if record.session_id is None:
            return await self._finish("discarded", "no_session")
        if record.session_id != self.runtime.session_id:
            return await self._finish("discarded", "session_changed")
        config = await self.runtime.get_proactive_config()
        if not config.enabled:
            return await self._finish("discarded", "disabled")
        if not config.is_active(self.runtime._now()):
            return await self._finish("discarded", "outside_active_hours")
        if self.runtime._now() >= record.expires_at:
            return await self._finish("discarded", "expired")
        if record.feedback_version != await asyncio.to_thread(self.runtime.inbox.version):
            return await self._finish("discarded", "feedback_changed")
        if record.retry_at is not None and record.retry_at > self.runtime._now():
            return None
        lock = self.runtime.session_lock
        if lock.locked():
            self.idle = None
            return None
        try:
            async with lock:
                try:
                    snapshot = await self.runtime.agent_wrapper.read_session(record.session_id)
                    if record.session_id != self.runtime.session_id:
                        return await self._finish("discarded", "session_changed")
                    if snapshot.state.status not in {"idle", "not_loaded"}:
                        self.idle = None
                        return None
                    turn_id = snapshot.last_turn_id or ""
                    clock = asyncio.get_running_loop().time()
                    if self.idle is None or self.idle[0] != turn_id:
                        self.idle = (turn_id, clock)
                        return None
                    if clock - self.idle[1] < 10:
                        return None
                    config = await self.runtime.get_proactive_config()
                    if record.feedback_version != await asyncio.to_thread(self.runtime.inbox.version):
                        return await self._finish("discarded", "feedback_changed")
                    if not config.is_active(self.runtime._now()):
                        return await self._finish("discarded", "outside_active_hours")
                    if self.runtime._now() >= record.expires_at:
                        return await self._finish("discarded", "expired")
                    window = await self._refresh_window(config)
                    if not window.allows(self.runtime._now(), f"reme-proactive-{record.id}"):
                        return await self._finish("discarded", "day_closed")
                    timeout = min(1, (record.expires_at - self.runtime._now()).total_seconds())
                    if timeout <= 0:
                        return await self._finish("discarded", "expired")
                    await self.runtime.agent_wrapper.append_session(
                        record.session_id,
                        record.content,
                        message_id=record.id,
                        timeout=timeout,
                        expected_turn_id=turn_id,
                    )
                except (SessionChangedError, TimeoutError):
                    self.idle = None
                    return None
                except OSError as exc:
                    await self._retry(record, exc)
                    return None
                except Exception as exc:
                    return await self._finish("failed", "delivery_failed", str(exc)[:1000])
                return await self._finish("sent")
        finally:
            self.runtime.publish_status()

    async def history(self, limit: int = 10, offset: int = 0) -> tuple[list[dict], tuple[int, int, int]]:
        """Read current reminder states, newest creation first; never consume history."""
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 to 100")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        async with self.lock:
            version = self.history_version()
            records = await asyncio.to_thread(self.read)
            self.check_history(version)
            return [self.answer(record) for record in records[offset : offset + limit]], version
