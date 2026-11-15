"""Atomic SDK snapshots referencing committed prefixes of append-only display history."""

import asyncio
from copy import deepcopy
import json
import os
from dataclasses import dataclass, field
from uuid import uuid4
from pathlib import Path

from agentscope.message import Msg, UserMsg
from agentscope.state import AgentState

from .async_utils import complete_in_thread
from .file_sync import sync_directory, sync_file
from ..schema.agent_session import SessionMessage


@dataclass
class AsSessionData:
    """Native model state and the history that must survive its compression."""

    state: AgentState
    messages: list[SessionMessage] = field(default_factory=list)
    last_turn_id: str | None = None
    history_path: Path | None = field(default=None, repr=False)
    saved_messages: list[SessionMessage] = field(default_factory=list, repr=False)


class AsStateHandler:
    """Serialize / deserialize AgentState to a JSONL file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @classmethod
    def for_session(cls, directory: str | Path, session_id: str) -> "AsStateHandler":
        """Create a handler for ``<directory>/<session_id>.jsonl``."""
        if not session_id or Path(session_id).name != session_id:
            raise ValueError(f"Invalid session_id: {session_id!r}")
        return cls(Path(directory) / f"{session_id}.jsonl")

    def exists(self) -> bool:
        """Return whether the state file exists."""
        return self.path.is_file()

    async def dump(self, state: AgentState) -> Path:
        """Write a standalone SDK state without application display history."""
        return await self.dump_session(AsSessionData(state))

    async def dump_session(self, session: AsSessionData) -> Path:
        """Finish serialization, history append and snapshot publication off the event loop."""
        await complete_in_thread(self._dump_session, session)
        return self.path

    def history_path(self) -> Path:
        """Keep paired history under the configured state directory, without symlink redirects."""
        root = self.path.parent.resolve()
        path = root / "history" / self.path.name
        if path.parent.is_symlink() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("AgentScope history path escapes its configured storage directory")
        if path.exists() and not path.is_file():
            raise ValueError("AgentScope history must be a regular file")
        return path

    def _dump_session(self, session: AsSessionData) -> None:
        history = self.history_path()
        committed = 0
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open(encoding="utf-8") as stream:
                metadata = Msg.model_validate_json(stream.readline()).metadata or {}
                committed = self._history_size(metadata)
        start = 0
        if session.history_path == history:
            for previous, current in zip(session.saved_messages, session.messages):
                if previous != current:
                    break
                start += 1
        changed = start != len(session.messages) or start != len(session.saved_messages)
        raw = b""
        if changed or session.history_path != history:
            record = {"start": start, "messages": [m.model_dump(mode="json") for m in session.messages[start:]]}
            raw = (json.dumps(record) + "\n").encode("utf-8")
        state = session.state
        header = UserMsg(
            name="__state__",
            content=state.summary or "",
            metadata={
                "agent_state": state.model_dump(mode="json", exclude={"context"}),
                "history_bytes": committed + len(raw),
                "last_turn_id": session.last_turn_id,
            },
        )
        records = [header.model_dump_json(), *(msg.model_dump_json() for msg in state.context)]
        content = "\n".join(records) + "\n"
        history.parent.mkdir(parents=True, exist_ok=True)
        if raw or history.exists():
            with history.open("r+b" if history.exists() else "w+b") as stream:
                if stream.seek(0, os.SEEK_END) < committed:
                    raise ValueError("AgentScope history is shorter than its committed prefix")
                # Only an uncommitted tail can be removed. The old snapshot stays readable on failure.
                stream.truncate(committed)
                stream.seek(committed)
                stream.write(raw)
                stream.flush()
                sync_file(stream.fileno())
            sync_directory(history.parent)
        self._write(content)
        session.history_path = history
        session.saved_messages = [*session.saved_messages[:start], *deepcopy(session.messages[start:])]

    def _write(self, content: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_path, self.path)
            sync_directory(self.path.parent)
        finally:
            tmp_path.unlink(missing_ok=True)

    async def load(self) -> AgentState:
        """Read native state through the same parser used by session operations."""
        return (await self.load_session()).state

    async def load_session(self) -> AsSessionData:
        """Read a complete session; Unicode separators inside JSON strings stay intact."""
        return await asyncio.to_thread(self._load_session)

    @staticmethod
    def _history_size(metadata: dict) -> int:
        size = metadata.get("history_bytes", 0)
        if type(size) is not int or size < 0:
            raise ValueError("Invalid AgentScope history prefix")
        return size

    def _load_session(self) -> AsSessionData:
        content = self.path.read_text(encoding="utf-8")
        lines = content.split("\n") if content else []
        if not lines:
            return AsSessionData(AgentState())

        header = Msg.model_validate_json(lines[0])
        summary: str | list = (
            list(header.content)
            if any(getattr(b, "type", None) == "data" for b in header.content)
            else header.get_text_content() or ""
        )

        metadata = header.metadata or {}
        scalars = metadata.get("agent_state") or {
            k: metadata.get(k, d) for k, d in [("session_id", ""), ("reply_id", ""), ("cur_iter", 0)]
        }
        state = AgentState.model_validate(
            {
                **scalars,
                "summary": scalars.get("summary", summary),
                "context": [Msg.model_validate_json(line) for line in lines[1:] if line.strip()],
            },
        )
        messages = [SessionMessage.model_validate(message) for message in metadata.get("messages", [])]
        history = None
        if "history_bytes" in metadata:
            history = self.history_path()
            remaining = self._history_size(metadata)
            if remaining:
                with history.open("rb") as stream:
                    while remaining:
                        line = stream.readline(remaining)
                        if not line.endswith(b"\n"):
                            raise ValueError("Incomplete committed AgentScope history")
                        remaining -= len(line)
                        record = json.loads(line)
                        start = record["start"]
                        if type(start) is not int or not 0 <= start <= len(messages):
                            raise ValueError("Invalid AgentScope history offset")
                        messages[start:] = [SessionMessage.model_validate(item) for item in record["messages"]]
        return AsSessionData(
            state,
            messages,
            metadata.get("last_turn_id"),
            history,
            deepcopy(messages),
        )
