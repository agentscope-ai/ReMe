"""Application-owned, bounded notifications for the selected PA conversation."""

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable
from hashlib import sha256
import math
import re
from uuid import uuid4

from .agent_wrapper import CodexAgentWrapper
from .base_component import BaseComponent
from .component_registry import R
from ..schema.agent_session import SessionMessage, SessionSnapshot, SessionStatus
from ..schema.pa_event import PaEvent

_CURSOR = re.compile(r"([0-9a-f]{32}):(0|[1-9][0-9]{0,19})\Z")


@R.register("local")
class PaRuntime(BaseComponent):
    """Keep notifications rebuildable from session files, without per-client queues.

    Synchronous publication and cursor checks run on the application's event
    loop. A slow reader falls behind the shared buffer and receives a fresh
    snapshot. Existing PA steps share the same application-lifetime state.
    """

    component_type = "pa"

    def __init__(
        self,
        agent_wrapper: str = "default",
        max_events: int = 512,
        max_buffer_bytes: int = 4 * 1024 * 1024,
        heartbeat_seconds: float = 15,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        for name, value in (("max_events", max_events), ("max_buffer_bytes", max_buffer_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(heartbeat_seconds, bool)
            or not isinstance(heartbeat_seconds, (int, float))
            or not math.isfinite(heartbeat_seconds)
            or heartbeat_seconds <= 0
        ):
            raise ValueError("heartbeat_seconds must be positive and finite")
        self.agent_wrapper = self.bind(agent_wrapper, CodexAgentWrapper, optional=False)
        self.max_events = max_events
        self.max_buffer_bytes = max_buffer_bytes
        self.heartbeat_seconds = heartbeat_seconds
        self.state: dict = {}
        self._epoch = uuid4().hex
        self._sequence = 0
        self._revision = 0
        self._buffer: deque[tuple[int, str, int]] = deque()
        self._buffer_bytes = 0
        self._changed = asyncio.Event()
        self._closed = True
        self._unsubscribe: Callable[[], None] | None = None
        self._native_status: SessionStatus | None = None
        self._published_status: tuple | None = None
        self._fingerprints: dict[str, bytes] = {}
        self._needs_refresh = False
        self._reset_reason = "cursor_expired"

    @property
    def cursor(self) -> str:
        """Return the current replay watermark."""
        return f"{self._epoch}:{self._sequence}"

    async def _start(self) -> None:
        if self.app_context is None or not isinstance(self.agent_wrapper, CodexAgentWrapper):
            raise RuntimeError("PaRuntime requires an ApplicationContext and CodexAgentWrapper")
        self.state = self.app_context.metadata.setdefault("pa", {"session_id": None, "lock": asyncio.Lock()})
        self._closed = False
        self._invalidate("cursor_expired")
        self.select_session()

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
        event = PaEvent(type=event_type, cursor=self.cursor, session_id=self.state["session_id"], **values)
        serialized = event.model_dump_json(exclude_none=True)
        size = len(serialized.encode("utf-8"))
        self._buffer.append((self._sequence, serialized, size))
        self._buffer_bytes += size
        while len(self._buffer) > self.max_events or self._buffer_bytes > self.max_buffer_bytes:
            self._buffer_bytes -= self._buffer.popleft()[2]
        self._wake()

    def _activity(self) -> tuple[str | None, list[str]]:
        if self.state["lock"].locked():
            return "active", []
        if self._native_status is not None:
            return self._native_status.status, self._native_status.active_flags
        return None, []

    def publish_status(self) -> None:
        """Merge foreground serialization with the native session's activity."""
        if self._closed:
            return
        self._revision += 1
        status, flags = self._activity()
        signature = (self.state["session_id"], status, tuple(flags))
        if signature != self._published_status:
            self._published_status = signature
            self._publish("status_changed", status=status, active_flags=flags)

    def select_session(self) -> None:
        """Follow the committed selection, dropping listeners for superseded sessions."""
        if self._closed:
            return
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        session_id = self.state["session_id"]
        self._native_status = SessionStatus(session_id=session_id, status="idle") if session_id else None
        self._fingerprints.clear()
        self._revision += 1
        self._needs_refresh = False
        if session_id and self.agent_wrapper.app_server is not None:

            def receive(change):
                if not self._closed and self.state["session_id"] == session_id:
                    self.observe(change)

            self._unsubscribe = self.agent_wrapper.app_server.watch_session(session_id, receive)
        self._publish("session_changed")
        self.publish_status()

    @staticmethod
    def _fingerprint(message: SessionMessage) -> bytes:
        return sha256(message.model_dump_json().encode("utf-8")).digest()

    def observe(self, change: SessionMessage | SessionSnapshot | SessionStatus | None) -> None:
        """Accept published history/status from the session owner without blocking it."""
        if self._closed:
            return
        self._revision += 1
        session_id = self.state["session_id"]
        if change is None:
            self._native_status = SessionStatus(session_id=session_id, status="error") if session_id else None
            self._invalidate("upstream_disconnected")
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
            revision, session_id = self._revision, self.state["session_id"]
            snapshot, error = None, None
            try:
                if session_id is not None:
                    async with asyncio.timeout(30):
                        snapshot = await self.agent_wrapper.read_session(session_id)
            except Exception as exc:
                error = str(exc)
            if self._closed:
                return None
            if revision != self._revision or session_id != self.state["session_id"]:
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
                    yield PaEvent(type="heartbeat", session_id=self.state["session_id"])

    async def _close(self) -> None:
        self._closed = True
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        self._buffer.clear()
        self._buffer_bytes = 0
        self._fingerprints.clear()
        self._wake()
