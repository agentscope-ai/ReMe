"""Atomically persist AgentScope state and display history in one JSONL file.

The header contains the summary, remaining SDK state, and display history.
Subsequent records contain the current model context. Legacy headers remain readable.
"""

import asyncio
import os
from dataclasses import dataclass, field
from uuid import uuid4
from pathlib import Path

from agentscope.message import Msg, UserMsg
from agentscope.state import AgentState

from .async_utils import complete_in_thread
from ..schema.agent_session import SessionMessage


@dataclass
class AsSessionData:
    """Native model state and the history that must survive its compression."""

    state: AgentState
    messages: list[SessionMessage] = field(default_factory=list)
    last_turn_id: str | None = None


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
        """Commit state, history and proactive message IDs together, including on cancellation."""
        await complete_in_thread(self._dump_session, session)
        return self.path

    def _dump_session(self, session: AsSessionData) -> None:
        """Serialize growing history off the event loop while its owner holds the lease."""
        state = session.state
        header = UserMsg(
            name="__state__",
            content=state.summary or "",
            metadata={
                "agent_state": state.model_dump(mode="json", exclude={"context"}),
                "messages": [message.model_dump(mode="json") for message in session.messages],
                "last_turn_id": session.last_turn_id,
            },
        )
        records = [header.model_dump_json(), *(msg.model_dump_json() for msg in state.context)]
        self._write("\n".join(records) + "\n")

    def _write(self, content: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_path, self.path)
        finally:
            tmp_path.unlink(missing_ok=True)

    async def load(self) -> AgentState:
        """Read native state through the same parser used by session operations."""
        return (await self.load_session()).state

    async def load_session(self) -> AsSessionData:
        """Read a complete session; Unicode separators inside JSON strings stay intact."""
        content = await asyncio.to_thread(self.path.read_text, encoding="utf-8")
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
        return AsSessionData(
            state,
            [SessionMessage.model_validate(message) for message in metadata.get("messages", [])],
            metadata.get("last_turn_id"),
        )
