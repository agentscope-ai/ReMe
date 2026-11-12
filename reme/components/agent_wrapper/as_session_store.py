"""File-native AgentScope sessions, serialized writes and nonblocking history reads."""

import asyncio
import math
import re
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from agentscope.message import AssistantMsg, Msg, UserMsg
from agentscope.permission import PermissionContext, PermissionMode
from agentscope.state import AgentState

from ...enumeration import ChunkEnum
from ...schema import SessionMessage, SessionSnapshot, SessionStatus, StreamChunk
from ...utils.agent_state_io import AsSessionData, AsStateHandler

_UUID = re.compile(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}", re.IGNORECASE)
_MESSAGE_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")


@dataclass
class _SessionRuntime:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0
    session: AsSessionData | None = None


def _message(msg: Msg, turn_id: str | None = None) -> SessionMessage:
    """Project stored native messages without exposing internal system messages."""
    chunks = []
    for block in msg.content:
        native = block.model_dump(mode="json")
        source = native["source"] if block.type == "data" else {}
        payload = next((native[key] for key in ("text", "thinking", "input", "output") if key in native), None)
        if block.type == "data":
            payload = {"url": source["url"]} if "url" in source else source.get("data")
        chunks.append(
            StreamChunk(
                chunk_type={
                    "text": ChunkEnum.CONTENT,
                    "thinking": ChunkEnum.THINK,
                    "tool_call": ChunkEnum.TOOL_CALL,
                    "tool_result": ChunkEnum.TOOL_RESULT,
                }.get(block.type, ChunkEnum.DATA),
                event="end",
                chunk=payload,
                block_id=block.id,
                message_id=msg.id,
                turn_id=turn_id,
                tool_call_id=block.id if block.type in {"tool_call", "tool_result"} else None,
                tool_call_name=native.get("name") if block.type == "tool_call" else None,
                media_type=source.get("media_type"),
                metadata={"native": native, **({"visibility": "diagnostic"} if block.type == "hint" else {})},
            ),
        )
    return SessionMessage(
        id=msg.id,
        role=msg.role,
        content=msg.get_text_content() or "",
        created_at=msg.created_at,
        turn_id=turn_id,
        chunks=chunks,
    )


def record_inputs(session: AsSessionData, inputs: Any) -> None:
    """Record user/observation inputs independently of the SDK's mutable context."""
    messages = [inputs] if isinstance(inputs, Msg) else inputs if isinstance(inputs, list) else []
    for msg in messages:
        if isinstance(msg, Msg) and msg.role != "system":
            message = _message(msg)
            for chunk in message.chunks:
                chunk.session_id = session.state.session_id
            session.messages.append(message)


def record_chunk(session: AsSessionData, chunk: StreamChunk) -> None:
    """Retain display events and reconcile text snapshots by their native block IDs."""
    turn_id = chunk.turn_id or session.last_turn_id
    if turn_id is None:
        return
    session.last_turn_id = turn_id
    for message in reversed(session.messages):
        if message.turn_id is not None:
            break
        if message.source == "chat":
            message.turn_id = turn_id
            for event in message.chunks:
                event.turn_id = turn_id
    message = next((msg for msg in reversed(session.messages) if msg.id == turn_id and msg.role == "assistant"), None)
    if message is None:
        message = SessionMessage(id=turn_id, role="assistant", content="", turn_id=turn_id, completed=False)
        session.messages.append(message)
    if message.completed:
        message.chunks = [event for event in message.chunks if event.chunk_type != ChunkEnum.REPLY_END]
    message.completed = chunk.chunk_type == ChunkEnum.REPLY_END
    message.chunks.append(chunk.model_copy(deep=True))
    if chunk.chunk_type != ChunkEnum.CONTENT or not isinstance(chunk.chunk, str):
        return
    if chunk.event == "delta":
        message.content += chunk.chunk
        return
    blocks: dict[str | None, str] = {}
    for event in message.chunks:
        if event.chunk_type == ChunkEnum.CONTENT and isinstance(event.chunk, str):
            key = event.block_id
            blocks[key] = blocks.get(key, "") + event.chunk if event.event == "delta" else event.chunk
    message.content = "".join(blocks.values())


def record_end(session: AsSessionData, terminal: StreamChunk, native: Msg | None) -> None:
    """Settle history with the SDK's final text, which may correct streamed deltas."""
    if terminal.turn_id is None:
        return
    message = next((msg for msg in reversed(session.messages) if msg.id == terminal.turn_id), None)
    if native is not None and native.finished_reason == "completed":
        chunks = message.chunks if message is not None else []
        start = next(
            (
                index
                for index in range(len(chunks) - 1, -1, -1)
                if chunks[index].chunk_type == ChunkEnum.USAGE and chunks[index].event == "start"
            ),
            0,
        )
        block_ids = list(
            dict.fromkeys(event.block_id for event in chunks[start:] if event.chunk_type == ChunkEnum.CONTENT),
        )
        for block_id in block_ids[:-1]:
            record_chunk(session, StreamChunk(chunk="", event="end", block_id=block_id, turn_id=terminal.turn_id))
        record_chunk(
            session,
            StreamChunk(
                chunk=native.get_text_content() or "",
                event="end",
                session_id=terminal.session_id,
                turn_id=terminal.turn_id,
                message_id=terminal.turn_id,
                block_id=block_ids[-1] if block_ids else f"{terminal.turn_id}:final",
            ),
        )
    record_chunk(session, terminal)
    if native is not None and session.messages:
        session.messages[-1].created_at = native.created_at


class AsSessionStore:
    """One application's owner for sessions sharing an AgentScope storage directory."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._runtimes: dict[str, _SessionRuntime] = {}

    def in_use(self, session_id: str) -> bool:
        """Keep active and queued sessions out of retention cleanup."""
        return session_id in self._runtimes

    @staticmethod
    def validate_id(value: str, field_name: str = "session_id") -> str:
        """Normalize UUIDs before resolving paths or creating runtime resources."""
        if not isinstance(value, str) or not _UUID.fullmatch(value):
            raise ValueError(f"{field_name} must be a valid UUID: {value!r}")
        return value.lower()

    def path(self, session_id: str) -> Path:
        """Contain lookups within the configured session directory, including external storage."""
        session_id = self.validate_id(session_id)
        root = self.root.resolve()
        path = root / f"{session_id}.jsonl"
        if not path.resolve().is_relative_to(root):
            raise ValueError("AgentScope session path escapes its configured storage directory")
        return path

    def options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Keep explicit new-session IDs distinct from resume and fork requests."""
        kwargs = dict(kwargs)
        for key in ("session_id", "resume"):
            if kwargs.get(key):
                kwargs[key] = self.validate_id(kwargs[key], key)
        if kwargs.get("session_id") and kwargs.get("resume") and not kwargs.get("fork_session"):
            raise ValueError("session_id cannot be used with resume unless fork_session=True")
        if not kwargs.get("resume") or kwargs.get("fork_session"):
            kwargs["session_id"] = kwargs.get("session_id") or str(uuid4())
        if kwargs.get("fork_session") and kwargs.get("session_id") == kwargs.get("resume"):
            raise ValueError("A fork must have a different session_id")
        return kwargs

    @asynccontextmanager
    async def _lease(self, session_id: str) -> AsyncIterator[_SessionRuntime]:
        """Serialize mutations and discard idle runtime bookkeeping."""
        self.path(session_id)
        runtime = self._runtimes.setdefault(session_id, _SessionRuntime())
        runtime.users += 1
        try:
            async with runtime.lock:
                yield runtime
        finally:
            runtime.users -= 1
            if runtime.users == 0:
                self._runtimes.pop(session_id)

    async def load(self, session_id: str) -> AsSessionData:
        """Load an atomic snapshot, seeding display history for legacy state files."""
        session_id = self.validate_id(session_id)
        path = self.path(session_id)
        try:
            session = await AsStateHandler(path).load_session()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"AgentScope session not found: {session_id}") from exc
        if session.state.session_id.lower() != session_id:
            raise ValueError("AgentScope session ID does not match its file")
        if not session.messages:
            for msg in session.state.context:
                if msg.role == "system":
                    continue
                session.messages.append(_message(msg, msg.id if msg.role == "assistant" else None))
                if msg.role == "assistant":
                    session.last_turn_id = msg.id
        return session

    async def save(self, session: AsSessionData) -> None:
        """Commit native state and display provenance in one atomic replacement."""
        await AsStateHandler(self.path(session.state.session_id)).dump_session(session)

    @asynccontextmanager
    async def open(self, options: dict[str, Any], mode: PermissionMode) -> AsyncIterator[AsSessionData]:
        """Hold source/target leases through loading, execution, and final persistence."""
        source = options.get("resume")
        target = options.get("session_id") or source
        async with AsyncExitStack() as stack:
            runtimes = {
                key: await stack.enter_async_context(self._lease(key)) for key in sorted({source, target} - {None})
            }
            if source:
                session = await self.load(source)
                if source != target:
                    if self.path(target).exists():
                        raise FileExistsError(f"AgentScope fork target already exists: {target}")
                    session.state.session_id = target
                    for message in session.messages:
                        for chunk in message.chunks:
                            chunk.session_id = target
            else:
                # Explicit session_id starts fresh model context, as before; retain its audit history.
                session = await self.load(target) if self.path(target).exists() else AsSessionData(AgentState())
                session.state = AgentState(session_id=target)
            session.state.permission_context = PermissionContext(mode=mode)
            runtime = runtimes[target]
            runtime.session = session
            try:
                yield session
            finally:
                runtime.session = None

    async def exists(self, session_id: str) -> bool:
        """Check resumability without constructing an agent or invoking its model."""
        path = self.path(session_id)
        runtime = self._runtimes.get(session_id)
        return path.is_file() or (runtime is not None and runtime.session is not None)

    async def read(self, session_id: str) -> AsSessionData:
        """Read current display history without waiting behind the generation lease."""
        session_id = self.validate_id(session_id)
        self.path(session_id)
        runtime = self._runtimes.get(session_id)
        if runtime is not None:
            if runtime.session is not None:
                return deepcopy(runtime.session)
            if not self.path(session_id).exists():
                return AsSessionData(AgentState(session_id=session_id))
        return await self.load(session_id)

    async def get_status(self, session_id: str) -> SessionStatus:
        """Inspect live work without copying history or waiting for state loading."""
        session_id = self.validate_id(session_id)
        self.path(session_id)
        runtime = self._runtimes.get(session_id)
        if runtime is not None and runtime.lock.locked():
            if runtime.session is not None:
                return self.status(runtime.session)
            return SessionStatus(session_id=session_id, status="active")
        return self.status(await self.load(session_id))

    def status(self, session: AsSessionData) -> SessionStatus:
        """Distinguish live work and parked tools from appendable idle sessions."""
        session_id = session.state.session_id
        runtime = self._runtimes.get(session_id)
        active = runtime is not None and runtime.lock.locked()
        tail = session.state.context[-1] if session.state.context else None
        waiting = tail is not None and session.state.has_awaiting_tool_calls(tail.name)
        return SessionStatus(
            session_id=session_id,
            status="active" if active or waiting else "idle",
            active_flags=["waitingOnApprovalOrExternalExecution"] if waiting else [],
        )

    async def snapshot(self, session_id: str) -> SessionSnapshot:
        """Return detached display data, including live partial assistant output."""
        session_id = self.validate_id(session_id)
        session = await self.read(session_id)
        return SessionSnapshot(
            session_id=session_id,
            state=self.status(session),
            messages=session.messages,
            path=str(self.path(session_id)),
            last_turn_id=session.last_turn_id,
        )

    async def append(
        self,
        session_id: str,
        content: str,
        *,
        message_id: str,
        role: Literal["user", "assistant"],
        timeout: float,
        expected_turn_id: str | None,
        delivery: str,
        name: str,
    ) -> SessionMessage:
        """Commit one observation and its idempotency key without calling a model."""
        session_id = self.validate_id(session_id)
        if role not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        if not isinstance(message_id, str) or not _MESSAGE_ID.fullmatch(message_id):
            raise ValueError("message_id must contain 1-128 letters, digits, underscores or hyphens")
        if not isinstance(content, str) or not content.strip() or len(content) > 100_000:
            raise ValueError("content must contain 1-100000 characters")
        if delivery not in {"idle", "immediate"}:
            raise ValueError("Invalid chat delivery policy")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        async with asyncio.timeout(timeout):
            async with self._lease(session_id):
                session = await self.load(session_id)
                native_id = f"reme-proactive-{message_id}"
                accepted = next((msg for msg in session.messages if msg.id == native_id), None)
                if accepted is not None:
                    if accepted.source != "proactive" or (accepted.content, accepted.role) != (content, role):
                        raise ValueError("message_id already belongs to different content or role")
                    return accepted
                tail = session.state.context[-1] if session.state.context else None
                if tail is not None and session.state.has_awaiting_tool_calls(tail.name):
                    raise RuntimeError("Session must be idle before appending")
                if expected_turn_id is not None and (session.last_turn_id or "") != expected_turn_id:
                    raise RuntimeError("Session changed before appending")
                factory = AssistantMsg if role == "assistant" else UserMsg
                native = factory(name=name if role == "assistant" else "user", content=content, id=native_id)
                message = _message(native, session.last_turn_id)
                message.source = "proactive"
                message.delivery = delivery
                message.origin_session_id = session_id
                native.metadata["reme"] = message.model_dump(mode="json", exclude={"chunks"})
                session.state.context.append(native)
                session.messages.append(message)
                await self.save(session)
                return message
