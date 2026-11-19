"""File-native AgentScope sessions, serialized writes and nonblocking history reads."""

import asyncio
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from agentscope.message import (
    AssistantMsg,
    Msg,
    ToolCallBlock,
    ToolCallState,
    ToolResultBlock,
    ToolResultState,
    UserMsg,
)
from agentscope.permission import PermissionContext, PermissionMode
from agentscope.state import AgentState

from .session_append import SessionAppends, check_guard, check_identity, validate_append

from ...enumeration import ChunkEnum
from ...schema import SessionMessage, SessionSnapshot, SessionStatus, StreamChunk
from ...utils.agent_state_io import AsSessionData, AsStateHandler

_UUID = re.compile(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}", re.IGNORECASE)


@dataclass
class _SessionRuntime:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0
    session: AsSessionData | None = None
    appends: SessionAppends = field(default_factory=SessionAppends)
    running: bool = False


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
    if chunk.event == "end" and chunk.block_id is not None and chunk.chunk is not None:
        message.chunks = [
            event
            for event in message.chunks
            if (event.chunk_type, event.block_id) != (chunk.chunk_type, chunk.block_id)
        ]
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
    if native is not None and message is not None:
        message.created_at = native.created_at


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
        if kwargs.get("fork_session") and not kwargs.get("resume"):
            raise ValueError("fork_session=True requires resume")
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
    async def _use(self, session_id: str) -> AsyncIterator[_SessionRuntime]:
        """Pin the owner while callers wait without taking the generation lock."""
        self.path(session_id)
        runtime = self._runtimes.setdefault(session_id, _SessionRuntime())
        runtime.users += 1
        try:
            yield runtime
        finally:
            runtime.users -= 1
            if runtime.users == 0 and not runtime.appends.pending:
                self._runtimes.pop(session_id)

    @asynccontextmanager
    async def _lease(self, session_id: str) -> AsyncIterator[_SessionRuntime]:
        """Serialize generation and idle mutations; active appends use their own queue."""
        async with self._use(session_id) as runtime, runtime.lock:
            yield runtime

    async def load(self, session_id: str) -> AsSessionData:
        """Load an atomic snapshot and project native-only state into display messages."""
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

    def finish(self, session):
        """New arrivals wait for the idle lease once final persistence begins."""
        self._runtimes[session.state.session_id].running = False

    @asynccontextmanager
    async def open(
        self,
        options: dict[str, Any],
        mode: PermissionMode,
        *,
        active: bool = False,
    ) -> AsyncIterator[AsSessionData]:
        """Lock only the target; a fork copies live state without waiting for its source."""
        source = options.get("resume")
        target = options.get("session_id") or source
        async with self._lease(target) as runtime:
            if source:
                session = await self.read(source) if source != target else await self.load(source)
                if source != target:
                    if self.path(target).exists():
                        raise FileExistsError(f"AgentScope fork target already exists: {target}")
                    session.state.session_id = target
                    self._seal_fork(session)
                    for message in session.messages:
                        for chunk in message.chunks:
                            chunk.session_id = target
            else:
                # Explicit session_id starts fresh model context, as before; retain its audit history.
                session = await self.load(target) if self.path(target).exists() else AsSessionData(AgentState())
                session.state = AgentState(session_id=target)
            session.state.permission_context = PermissionContext(mode=mode)
            runtime.session = session
            runtime.running = active
            try:
                yield session
            finally:
                runtime.session = None
                runtime.running = False
                runtime.appends.fail(RuntimeError("AgentScope execution ended before append persistence"))

    @staticmethod
    def _seal_fork(session):
        """Copy data, never executing tasks: finish dangling tools only in the detached branch."""
        for native in session.state.context:
            completed = {block.id for block in native.get_content_blocks("tool_result")}
            for call in list(native.get_content_blocks("tool_call")):
                if isinstance(call, ToolCallBlock) and call.id not in completed:
                    call.state = ToolCallState.FINISHED
                    native.content.append(
                        ToolResultBlock(
                            id=call.id,
                            name=call.name,
                            output="Source turn was still running when forked.",
                            state=ToolResultState.INTERRUPTED,
                        ),
                    )
        for message in session.messages:
            if not message.completed:
                message.completed = True
                message.chunks.append(
                    StreamChunk(
                        chunk_type=ChunkEnum.REPLY_END,
                        event="end",
                        session_id=session.state.session_id,
                        turn_id=message.turn_id,
                        status="interrupted",
                    ),
                )

    async def exists(self, session_id: str) -> bool:
        """Check resumability without constructing an agent or invoking its model."""
        session_id = self.validate_id(session_id)
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

    @asynccontextmanager
    async def guard(self, session_id: str, expected_revision: str):
        """Protect a decision against generation and queued observation writers."""
        async with self._lease(session_id) as runtime, runtime.appends.lock:
            check_guard(await self.snapshot(session_id), None, expected_revision)
            yield

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
        expected_revision: str | None = None,
    ) -> SessionMessage:
        """Queue during generation; idle appends and SDK boundaries commit the same records."""
        session_id = self.validate_id(session_id)
        validate_append(content, message_id, role, timeout, delivery)
        async with asyncio.timeout(timeout):
            async with self._use(session_id) as runtime:
                if delivery == "immediate":
                    async with runtime.appends.lock:
                        if runtime.running and runtime.session is not None:
                            pending, accepted = self._prepare_append(
                                runtime,
                                runtime.session,
                                content,
                                message_id,
                                role,
                                delivery,
                                name,
                                expected_turn_id,
                                expected_revision,
                            )
                        else:
                            pending = accepted = None
                    if accepted is not None:
                        return accepted
                    if pending is not None:
                        return await pending.wait()
                async with runtime.lock:
                    session = await self.load(session_id)
                    async with runtime.appends.lock:
                        pending, accepted = self._prepare_append(
                            runtime,
                            session,
                            content,
                            message_id,
                            role,
                            delivery,
                            name,
                            expected_turn_id,
                            expected_revision,
                        )
                    if accepted is not None:
                        return accepted
                    await self.drain(session)
                    return await pending.wait()

    def _prepare_append(
        self,
        runtime,
        session,
        content,
        message_id,
        role,
        delivery,
        name,
        expected_turn_id,
        expected_revision,
    ):
        identifier = f"reme-proactive-{message_id}"
        pending = runtime.appends.get(identifier, content, role)
        if pending is not None:
            return pending, None
        accepted = next((message for message in session.messages if message.id == identifier), None)
        if accepted is not None:
            check_identity(accepted, content, role)
            return None, accepted.model_copy(deep=True)
        self._require_appendable(session)
        snapshot = SessionSnapshot(
            session_id=session.state.session_id,
            state=self.status(session),
            messages=session.messages,
            last_turn_id=session.last_turn_id,
        )
        check_guard(snapshot, expected_turn_id, expected_revision)
        factory = AssistantMsg if role == "assistant" else UserMsg
        native = factory(name=name if role == "assistant" else "user", content=content, id=identifier)
        message = _message(native, session.last_turn_id)
        message.source, message.delivery = "proactive", delivery
        message.origin_session_id = session.state.session_id
        native.metadata["reme"] = message.model_dump(mode="json", exclude={"chunks"})
        return runtime.appends.add(message, native), None

    @staticmethod
    def _require_appendable(session):
        tail = session.state.context[-1] if session.state.context else None
        if tail is not None and session.state.has_awaiting_tool_calls(tail.name):
            raise RuntimeError("Session must be idle before appending")

    async def drain(self, session, agent=None):
        """Observe only before model input or after a reply, never during SDK compression.

        AgentScope replaces context after awaited compression. Concurrent observe()
        would lose messages. The wrapper suspends its SDK generator at these safe
        boundaries, then persists before resolving any pending receipt.
        """
        runtime = self._runtimes.get(session.state.session_id)
        if runtime is None or not runtime.appends.pending:
            return
        async with runtime.appends.lock:
            try:
                self._require_appendable(session)
            except RuntimeError as exc:
                runtime.appends.fail(exc)
                return
            try:
                for pending in list(runtime.appends.pending.values()):
                    if not any(message.id == pending.message.id for message in session.messages):
                        if agent is None:
                            session.state.context.append(pending.payload)
                        else:
                            await agent.observe(pending.payload)
                        session.messages.append(pending.message)
                await self.save(session)
                for pending in list(runtime.appends.pending.values()):
                    runtime.appends.resolve(pending.message)
            except BaseException as exc:
                runtime.appends.fail(exc)
                raise
