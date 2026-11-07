"""Reminder state and deferred delivery owned by PaRuntime."""

from __future__ import annotations

import asyncio
import os
from datetime import timedelta
from typing import TYPE_CHECKING

from .codex_app_server import SessionChangedError
from ..schema.reminder import ReminderRecord
from ..utils.async_utils import complete_in_thread

if TYPE_CHECKING:
    from .pa_runtime import PaRuntime


class ReminderDelivery:
    """Serialize latest-decision updates and delivery without owning agent rounds."""

    def __init__(self, runtime: PaRuntime):
        self.runtime = runtime
        self.lock = asyncio.Lock()
        self.pending: ReminderRecord | None = None
        self.idle: tuple[str, float] | None = None

    def read(self) -> list[ReminderRecord]:
        latest = {}
        try:
            descriptor = os.open(self.runtime.reminders_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            return []
        with os.fdopen(descriptor, "rb") as handle:
            for line in self.runtime._reverse_jsonl_lines(handle):
                record = ReminderRecord.model_validate_json(line)
                latest.setdefault(record.id, record)
        return sorted(latest.values(), key=lambda record: record.created_at, reverse=True)

    def restore(self) -> None:
        pending = [record for record in self.read() if record.status == "pending"]
        self.pending = pending[0] if pending else None
        self.idle = None
        for record in pending[1:]:
            record = record.model_copy(update={
                "status": "discarded", "reason": "superseded", "updated_at": self.runtime._now(),
            })
            self.runtime._append_jsonl(self.runtime.reminders_path, record.model_dump(mode="json"))

    @staticmethod
    def answer(record: ReminderRecord) -> dict:
        return {"sent": record.status == "sent", **record.model_dump(mode="json")}

    async def _finish(self, status: str, reason: str | None = None, error: str | None = None) -> ReminderRecord:
        record = self.pending
        now = self.runtime._now()
        record = record.model_copy(update={
            "status": status, "reason": reason, "error": error, "updated_at": now,
            "sent_at": now if status == "sent" else None,
        })
        await complete_in_thread(self.runtime._append_jsonl, self.runtime.reminders_path, record.model_dump(mode="json"))
        self.pending = None
        self.idle = None
        return record

    async def submit(self, content: str, *, session_id: str | None, message_id: str) -> dict:
        """Persist a validated round decision; the caller holds the submission lock."""
        records = await asyncio.to_thread(self.read)
        existing = next((record for record in records if record.id == message_id), None)
        if existing is not None:
            if existing.content != content:
                raise ValueError("message_id already belongs to different content")
            return self.answer(existing)
        if not content:
            if self.pending is not None and self.pending.session_id == session_id:
                await self._finish("discarded", "withdrawn")
            return {"id": message_id, "sent": False, "status": "skipped", "reason": "empty_content"}
        config = await self.runtime.get_proactive_config()
        now = self.runtime._now()
        record = ReminderRecord(
            id=message_id, content=content, created_at=now, updated_at=now,
            expires_at=now + timedelta(minutes=config.interval_minutes),
            session_id=session_id, reason="waiting_for_idle",
        )
        previous = self.pending
        await complete_in_thread(self.runtime._append_jsonl, self.runtime.reminders_path, record.model_dump(mode="json"))
        self.pending = record
        self.idle = None
        if previous is not None:
            previous = previous.model_copy(update={
                "status": "discarded", "reason": "superseded", "updated_at": now,
            })
            await complete_in_thread(self.runtime._append_jsonl, self.runtime.reminders_path, previous.model_dump(mode="json"))
        result = await self._deliver(immediate=True)
        return self.answer(result or record)

    async def deliver_pending(self) -> None:
        if self.pending is not None:
            async with self.lock:
                await self._deliver()

    async def _deliver(self, *, immediate: bool = False) -> ReminderRecord | None:
        record = self.pending
        if record is None:
            return None
        if record.session_id is None:
            return await self._finish("discarded", "no_session")
        if record.session_id != self.runtime.session_id:
            return await self._finish("discarded", "session_changed")
        config = await self.runtime.get_proactive_config()
        if not config.is_active(self.runtime._now()):
            return await self._finish("discarded", "outside_active_hours")
        if self.runtime._now() >= record.expires_at:
            return await self._finish("discarded", "expired")
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
                    if not immediate:
                        clock = asyncio.get_running_loop().time()
                        if self.idle is None or self.idle[0] != turn_id:
                            self.idle = (turn_id, clock)
                            return None
                        if clock - self.idle[1] < 10:
                            return None
                    config = await self.runtime.get_proactive_config()
                    if not config.is_active(self.runtime._now()):
                        return await self._finish("discarded", "outside_active_hours")
                    if self.runtime._now() >= record.expires_at:
                        return await self._finish("discarded", "expired")
                    await self.runtime.agent_wrapper.append_session(
                        record.session_id, record.content, message_id=record.id,
                        timeout=1, expected_turn_id=turn_id,
                    )
                except (SessionChangedError, TimeoutError):
                    self.idle = None
                    return None
                except Exception as exc:
                    return await self._finish("failed", "delivery_failed", str(exc)[:1000])
                return await self._finish("sent")
        finally:
            self.runtime.publish_status()

    async def history(self, limit: int = 10, offset: int = 0) -> list[dict]:
        """Read current reminder states, newest creation first; never consume history."""
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 to 100")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        async with self.lock:
            records = await asyncio.to_thread(self.read)
            return [self.answer(record) for record in records[offset:offset + limit]]
