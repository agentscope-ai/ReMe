"""Bounded PA event replay and activity snapshots, rebuilt from session owners."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable
from hashlib import sha256
import re
from typing import TYPE_CHECKING
from uuid import uuid4

from ...schema.agent_session import SessionMessage, SessionSnapshot, SessionStatus
from ...schema.agent_activity import AgentActivity
from ...schema.pa_event import PaEvent
from ...schema.pa_task import PaTaskRun

if TYPE_CHECKING:
    from .runtime import PaRuntime

_CURSOR = re.compile(r"([0-9a-f]{32}):(0|[1-9][0-9]{0,19})\Z")


class PaEventStream:
    """Keep one bounded replay buffer and reset slow readers to a fresh snapshot."""

    def __init__(self, runtime: PaRuntime, max_events: int, max_buffer_bytes: int, heartbeat_seconds: float):
        self.runtime = runtime
        self.max_events = max_events
        self.max_buffer_bytes = max_buffer_bytes
        self.heartbeat_seconds = heartbeat_seconds
        self._epoch = uuid4().hex
        self._sequence = 0
        self._revision = 0
        self._buffer: deque[tuple[int, str, int]] = deque()
        self._buffer_bytes = 0
        self._changed = asyncio.Event()
        self._closed = True
        self._unsubscribe: Callable[[], None] | None = None
        self._unsubscribe_activity: Callable[[], None] | None = None
        self._native_status: SessionStatus | None = None
        self._activities: dict[str, AgentActivity] = {}
        self._task_activities: dict[str, AgentActivity] = {}
        self._published_status: tuple | None = None
        self._fingerprints: dict[str, bytes] = {}
        self._needs_refresh = False
        self._reset_reason = "cursor_expired"

    def activity(self, source: str) -> AgentActivity | None:
        """Read the source's latest application-owned activity."""
        return self._activities.get(source)

    def start(self) -> None:
        """Attach listeners only after the runtime has restored its selection."""
        self._closed = False
        self._unsubscribe_activity = self.runtime.agent_wrapper.app_server.watch_activity(self.observe_task_activity)
        self._invalidate("cursor_expired")
        self.select_session()

    async def announce_messages(self) -> None:
        """Offer an immediate result independently of the selected chat's generation lock."""
        if self._closed:
            return
        messages = await asyncio.to_thread(self.runtime.reminders.inbox_messages)
        if self._closed:
            return
        self._revision += 1
        status, flags = self._activity()
        self._publish(
            "status_changed",
            status=status,
            active_flags=flags,
            activities=self._activities,
            task_activities=self._task_activities,
            inbox_messages=messages,
        )

    @property
    def cursor(self) -> str:
        """Return the current replay watermark."""
        return f"{self._epoch}:{self._sequence}"

    def _wake(self) -> None:
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()

    def _invalidate(self, reason: str) -> None:
        self._epoch = uuid4().hex
        self._sequence = 0
        self._revision += 1
        self._buffer.clear()
        self._buffer_bytes = 0
        self._fingerprints.clear()
        self._reset_reason = reason
        self._needs_refresh = True
        self._wake()

    def _publish(self, event_type: str, **values) -> None:
        self._sequence += 1
        event = PaEvent(
            type=event_type,
            cursor=self.cursor,
            session_id=self.runtime.session_id,
            notifications=self.runtime.notifications,
            **values,
        )
        serialized = event.model_dump_json(exclude_none=True)
        size = len(serialized.encode("utf-8"))
        self._buffer.append((self._sequence, serialized, size))
        self._buffer_bytes += size
        while len(self._buffer) > self.max_events or self._buffer_bytes > self.max_buffer_bytes:
            self._buffer_bytes -= self._buffer.popleft()[2]
        self._wake()

    def _activity(self) -> tuple[str | None, list[str]]:
        if self.runtime.session_lock.locked():
            return "active", self._native_status.active_flags if self._native_status is not None else []
        if self._native_status is not None:
            return self._native_status.status, self._native_status.active_flags
        return None, []

    def publish_status(self) -> None:
        """Merge foreground serialization with the native session's activity."""
        if self._closed:
            return
        status, flags = self._activity()
        signature = (
            self.runtime.session_id,
            status,
            tuple(flags),
            tuple((source, value.model_dump_json()) for source, value in sorted(self._activities.items())),
            tuple((task_id, value.model_dump_json()) for task_id, value in sorted(self._task_activities.items())),
            self.runtime.notifications.model_dump_json() if self.runtime.notifications is not None else None,
        )
        if signature != self._published_status:
            self._revision += 1
            self._published_status = signature
            self._publish(
                "status_changed",
                status=status,
                active_flags=flags,
                activities=self._activities,
                task_activities=self._task_activities,
            )

    def start_task_activity(self, run: PaTaskRun) -> None:
        """Keep a run busy through both model queries and Markdown publication."""
        if self._closed:
            return
        self._task_activities[run.task_id] = AgentActivity(
            session_id=run.agent_session_id or run.task_id,
            turn_id=run.started_at.isoformat(),
            source="task",
            phase="working",
            task_id=run.task_id,
            started_at=run.started_at,
        )
        self.publish_status()

    def observe_task_activity(self, change: AgentActivity | None) -> None:
        """Task runs outlive selected-chat changes and never enter its transcript."""
        if self._closed:
            return
        if change is None:
            self._task_activities.clear()
        elif (
            change.source == "task"
            and change.task_id in self._task_activities
            and change.started_at == self._task_activities[change.task_id].started_at
        ):
            if change.phase == "completed":
                change = change.model_copy(update={"phase": "working", "tool": None})
            self._task_activities[change.task_id] = change
        else:
            return
        self.publish_status()

    def finish_task_activity(self, task_id: str) -> None:
        """Retire the live presentation even when durable final recording fails."""
        self._task_activities.pop(task_id, None)
        self.publish_status()

    def select_session(self) -> None:
        """Follow the committed selection, dropping listeners for superseded sessions."""
        if self._closed:
            return
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        session_id = self.runtime.session_id
        self._activities.clear()
        self._native_status = SessionStatus(session_id=session_id, status="idle") if session_id else None
        self._fingerprints.clear()
        self._revision += 1
        self._needs_refresh = False
        if session_id and self.runtime.agent_wrapper.app_server is not None:

            def receive(change):
                if not self._closed and self.runtime.session_id == session_id:
                    self.observe(change)

            self._unsubscribe = self.runtime.agent_wrapper.app_server.watch_session(session_id, receive)
        self._publish("session_changed")
        self.publish_status()

    @staticmethod
    def _fingerprint(message: SessionMessage) -> bytes:
        return sha256(message.model_dump_json().encode("utf-8")).digest()

    def observe(self, change: AgentActivity | SessionMessage | SessionSnapshot | SessionStatus | None) -> None:
        """Accept published history/status from the session owner without blocking it."""
        if self._closed:
            return
        self._revision += 1
        session_id = self.runtime.session_id
        if change is None:
            self._activities.clear()
            self._native_status = SessionStatus(session_id=session_id, status="error") if session_id else None
            self._invalidate("upstream_disconnected")
        elif isinstance(change, AgentActivity):
            if change.session_id != session_id or change.source == "task":
                return
            self._activities[change.source] = change
        elif isinstance(change, SessionMessage):
            fingerprint = self._fingerprint(change)
            if self._fingerprints.get(change.id) != fingerprint:
                self._fingerprints[change.id] = fingerprint
                self._publish("message_upsert", message=change)
        elif change.session_id == session_id:
            if isinstance(change, SessionSnapshot):
                self._native_status = change.state
                for message in change.messages:
                    self.observe(message)
                self._needs_refresh = False
            else:
                self._native_status = change
        self.publish_status()

    async def _snapshot(self) -> PaEvent | None:
        # Retry if a switch or owner notification crosses the history read.
        # There is no await between the final version check and its watermark.
        while not self._closed:
            revision, session_id = self._revision, self.runtime.session_id
            snapshot, error = None, None
            try:
                await self.runtime.reminders.window(await self.runtime.get_proactive_config())
            except (OSError, ValueError) as exc:
                self.runtime.notifications = None
                self.runtime.logger.warning(f"Ordinary notifications paused by invalid configuration: {exc}")
            try:
                if session_id is not None:
                    async with asyncio.timeout(30):
                        snapshot = await self.runtime.agent_wrapper.read_session(session_id)
            except Exception as exc:
                error = str(exc)
            inbox_messages = await asyncio.to_thread(self.runtime.reminders.inbox_messages)
            if self._closed:
                return None
            if revision != self._revision or session_id != self.runtime.session_id:
                continue
            self._needs_refresh = error is not None
            if snapshot is not None:
                self._native_status = snapshot.state
            elif session_id is not None:
                self._native_status = SessionStatus(session_id=session_id, status="error")
            self.publish_status()
            status, flags = self._activity()
            return PaEvent(
                type="snapshot",
                cursor=None if error is not None else self.cursor,
                session_id=session_id,
                status=status,
                active_flags=flags,
                activities=self._activities,
                task_activities=self._task_activities,
                notifications=self.runtime.notifications,
                inbox_messages=inbox_messages,
                snapshot=snapshot,
                error=error,
            )
        return None

    def _position(self, cursor: str | None) -> tuple[int, str | None]:
        match = _CURSOR.fullmatch(cursor) if cursor is not None else None
        if match is None:
            return 0, "invalid_cursor"
        epoch, sequence = match[1], int(match[2])
        oldest = self._buffer[0][0] if self._buffer else self._sequence + 1
        if epoch != self._epoch:
            return 0, self._reset_reason
        if sequence > self._sequence or sequence < oldest - 1:
            return 0, "cursor_expired"
        return sequence, None

    async def events(self, cursor: str | None = None) -> AsyncIterator[PaEvent]:
        """Replay changes, reset gaps to snapshots, and wait without polling healthy history."""
        if cursor is not None and (not isinstance(cursor, str) or len(cursor) > 128):
            raise ValueError("cursor must be a string of at most 128 characters")
        retry_snapshot = False
        while not self._closed:
            changed = self._changed
            if retry_snapshot:
                # No delta may advance a reader beyond history it has not recovered.
                try:
                    await asyncio.wait_for(changed.wait(), timeout=self.heartbeat_seconds)
                except asyncio.TimeoutError:
                    pass
                snapshot = await self._snapshot()
                if snapshot is None:
                    return
                retry_snapshot = snapshot.error is not None
                cursor = snapshot.cursor
                yield snapshot
                continue
            sequence, reason = self._position(cursor)
            if reason is not None:
                if cursor is not None:
                    yield PaEvent(type="reset", reason=reason)
                snapshot = await self._snapshot()
                if snapshot is None:
                    return
                retry_snapshot = snapshot.error is not None
                cursor = snapshot.cursor
                yield snapshot
                continue
            pending = next((data for seq, data, _size in self._buffer if seq > sequence), None)
            if pending is not None:
                event = PaEvent.model_validate_json(pending)
                cursor = event.cursor
                yield event
                continue
            try:
                await asyncio.wait_for(changed.wait(), timeout=self.heartbeat_seconds)
            except asyncio.TimeoutError:
                if self._needs_refresh:
                    snapshot = await self._snapshot()
                    if snapshot is None:
                        return
                    retry_snapshot = snapshot.error is not None
                    cursor = snapshot.cursor
                    yield snapshot
                else:
                    yield PaEvent(type="heartbeat", session_id=self.runtime.session_id)

    async def close(self) -> None:
        """Detach listeners and wake subscriptions before application shutdown."""
        self._closed = True
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        if self._unsubscribe_activity is not None:
            self._unsubscribe_activity()
            self._unsubscribe_activity = None
        self._buffer.clear()
        self._buffer_bytes = 0
        self._fingerprints.clear()
        self._activities.clear()
        self._task_activities.clear()
        self._wake()
