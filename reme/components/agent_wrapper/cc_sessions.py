"""Claude native transcripts and application-owned session leases/provenance."""

import asyncio
import json
import math
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from pydantic import BaseModel, Field

from ...schema import SessionMessage, SessionStatus, StreamChunk
from ...utils.async_utils import complete_in_thread


def native_id(session_id: str, name: str) -> str:
    """Reserve stable CLI UUIDs for seed, flush and append records."""
    return str(uuid5(NAMESPACE_URL, f"reme:claude:{session_id}:{name}"))


class AppendRecord(BaseModel):
    """A durable intent whose native UUID allows recovery after an interrupted commit."""

    native_id: str
    message: SessionMessage
    accepted: bool = False


class SessionData(BaseModel):
    """Keep native location and ReMe-only provenance; conversation stays in JSONL."""

    cwd: str
    config_dir: str
    last_turn_id: str | None = None
    waiting: bool = False
    appends: list[AppendRecord] = Field(default_factory=list)
    hidden_ids: list[str] = Field(default_factory=list)


@dataclass
class SessionRuntime:
    """Request-owned SDK activity shared by wrappers using the same session directory."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0
    state: str = "running"
    input: SessionMessage | None = None
    chunks: list[StreamChunk] = field(default_factory=list)

    def observe(self, chunk: StreamChunk) -> None:
        """Track native waits and retain only the current main model message."""
        native = chunk.metadata.get("native") or {}
        if native.get("subtype") == "session_state_changed":
            self.state = native.get("data", {}).get("state", "running")
        if chunk.metadata.get("parent_tool_use_id") is not None:
            return
        if chunk.chunk_type in {"content", "think", "tool_call"} and chunk.message_id:
            if self.chunks and self.chunks[0].message_id != chunk.message_id:
                self.chunks.clear()
            self.chunks.append(chunk.model_copy(deep=True))
        elif chunk.chunk_type == "tool_result" and self.chunks:
            self.chunks.append(chunk.model_copy(deep=True))


class CcSessions:
    """Serialize native mutations while allowing lock-free reads of atomic metadata."""

    def __init__(self, root: Path, workspace: Path) -> None:
        self.root = root
        self.workspace = workspace
        self.runtimes: dict[str, SessionRuntime] = {}

    @staticmethod
    def validate_id(value: str) -> str:
        """Accept only UUIDs before resolving any session-owned path."""
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F-]{36}", value):
            raise ValueError("session_id must be a valid UUID")
        return str(UUID(value))

    def path(self, session_id: str) -> Path:
        """Contain ReMe-owned metadata in the explicitly configured storage directory."""
        session_id = self.validate_id(session_id)
        root = self.root.resolve()
        path = root / f"{session_id}.json"
        if not path.resolve().is_relative_to(root):
            raise ValueError("Claude session metadata escapes its configured storage directory")
        return path

    @staticmethod
    def location(options: Any) -> SessionData:
        """Match the effective subprocess cwd and CLAUDE_CONFIG_DIR without global env changes."""
        cwd = Path(options.cwd).resolve()
        config_dir = Path(options.env["CLAUDE_CONFIG_DIR"]).expanduser()
        return SessionData(cwd=str(cwd), config_dir=str((cwd / config_dir).resolve()))

    async def load(self, session_id: str, default: SessionData) -> SessionData:
        """Read native location and provenance, including unfinished append intents."""
        path = self.path(session_id)

        def read():
            data = (
                SessionData.model_validate_json(path.read_bytes()) if path.is_file() else default.model_copy(deep=True)
            )
            config_dir = Path(data.config_dir).resolve()
            if config_dir != Path(default.config_dir).resolve() and not config_dir.is_relative_to(
                self.workspace.resolve(),
            ):
                raise ValueError("Claude session config directory is outside its configured location")
            return data

        return await asyncio.to_thread(read)

    async def save(self, session_id: str, data: SessionData) -> None:
        """Finish one atomic metadata replacement before allowing another session writer."""
        path = self.path(session_id)
        payload = data.model_dump_json()

        def write():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            try:
                temporary.write_text(payload, encoding="utf-8")
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

        await complete_in_thread(write)

    @staticmethod
    def native_path(session_id: str, data: SessionData) -> Path:
        """Use the SDK's canonical project key and reject escaping transcript symlinks."""
        from claude_agent_sdk import project_key_for_directory

        root = Path(data.config_dir).expanduser().resolve()
        path = root / "projects" / project_key_for_directory(data.cwd) / f"{session_id}.jsonl"
        if not path.exists():
            matches = list((root / "projects").glob(f"*/{session_id}.jsonl"))
            if len(matches) > 1:
                raise ValueError("Claude session exists in multiple project directories")
            if matches:
                path = matches[0]
        if not path.resolve().is_relative_to(root):
            raise ValueError("Claude transcript escapes its configured config directory")
        return path

    async def entries(self, session_id: str, data: SessionData, store: Any = None) -> list[dict] | None:
        """Read the complete raw history; tolerate only an unfinished final JSONL record."""
        self.validate_id(session_id)
        if store is not None:
            from claude_agent_sdk import project_key_for_directory

            return await store.load({"project_key": project_key_for_directory(data.cwd), "session_id": session_id})
        path = self.native_path(session_id, data)

        def read():
            if not path.is_file():
                return None
            lines = path.read_text(encoding="utf-8").split("\n")
            result = []
            for index, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    if index == len(lines) - 1:
                        break
                    raise
                if not isinstance(entry, dict):
                    raise ValueError("Invalid Claude transcript entry")
                result.append(entry)
            return result

        return await asyncio.to_thread(read)

    async def exists(self, session_id: str, data: SessionData, store: Any = None) -> bool:
        """Keep local existence/status probes independent of transcript size."""
        if store is not None:
            return bool(await self.entries(session_id, data, store))
        path = self.native_path(session_id, data)
        return await asyncio.to_thread(path.is_file)

    async def latest(self, data: SessionData, store: Any = None) -> str | None:
        """Resolve native --continue to a main session before taking its writer lease."""
        from claude_agent_sdk import project_key_for_directory
        from .cc_session_store import CcFileSessionStore

        listing_store = store if store is not None else CcFileSessionStore(Path(data.config_dir) / "projects")
        sessions = await listing_store.list_sessions(project_key_for_directory(data.cwd))
        for candidate in sorted(sessions, key=lambda item: item["mtime"], reverse=True):
            try:
                session_id = self.validate_id(candidate["session_id"])
            except ValueError:
                continue
            entries = await self.entries(session_id, data, store)
            if entries and not entries[0].get("isSidechain"):
                return session_id
        return None

    @asynccontextmanager
    async def lease(self, session_id: str) -> AsyncIterator[SessionRuntime]:
        """Hold the same lease for reply, compaction, native injection and publication."""
        session_id = self.validate_id(session_id)
        runtime = self.runtimes.setdefault(session_id, SessionRuntime())
        runtime.users += 1
        try:
            async with runtime.lock:
                runtime.state = "running"
                runtime.input = None
                runtime.chunks.clear()
                yield runtime
        finally:
            runtime.users -= 1
            if runtime.users == 0:
                self.runtimes.pop(session_id)

    def status(self, session_id: str, data: SessionData) -> SessionStatus:
        """A result alone cannot declare a leased or approval-blocked session idle."""
        runtime = self.runtimes.get(session_id)
        waiting = (
            runtime.state == "requires_action" if runtime is not None and runtime.input is not None else data.waiting
        )
        return SessionStatus(
            session_id=session_id,
            status="active" if waiting or (runtime is not None and runtime.lock.locked()) else "idle",
            active_flags=["waitingOnApprovalOrInput"] if waiting else [],
        )

    @staticmethod
    def validate_append(content: str, message_id: str, role: str, timeout: float, delivery: str) -> None:
        """Reject invalid append requests before taking a lease or starting the SDK."""
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        if not isinstance(message_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", message_id):
            raise ValueError("Invalid message_id")
        if not isinstance(content, str) or not content.strip() or len(content) > 100_000:
            raise ValueError("content must contain 1-100000 characters")
        if delivery not in {"idle", "immediate"}:
            raise ValueError("Invalid chat delivery policy")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")


def input_frame(session_id: str, identifier: str, content: str, role: str = "user") -> dict:
    """Native no-inference input; assistant frames are committed by a subsequent user flush."""
    message = {"role": role, "content": [{"type": "text", "text": content}]}
    if role == "assistant":
        message.update(
            id=f"msg_{identifier}",
            type="message",
            model="<synthetic>",
            stop_reason="end_turn",
            stop_sequence=None,
            usage={"input_tokens": 0, "output_tokens": 0},
        )
    return {
        "type": role,
        "session_id": session_id,
        "uuid": identifier,
        "parent_tool_use_id": None,
        "message": message,
        "isSynthetic": True,
        "shouldQuery": False,
        "client_composed": True,
    }
