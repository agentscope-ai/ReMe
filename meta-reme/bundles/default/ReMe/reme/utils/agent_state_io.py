"""AgentState JSONL dump / load.

Format:
  Line 1 — header Msg: AgentState.summary as content, state scalars in metadata.
  Lines 2+ — AgentState.context, one Msg per line.
"""

import os
from uuid import uuid4
from pathlib import Path

import aiofiles
from agentscope.message import Msg, UserMsg
from agentscope.state import AgentState

_META_KEYS = ("session_id", "reply_id", "cur_iter")


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

    async def load_or_none(self) -> AgentState | None:
        """Load state if the file exists, otherwise return ``None``."""
        if not self.exists():
            return None
        return await self.load()

    async def delete(self) -> bool:
        """Delete the state file if present. Returns whether a file was removed."""
        if not self.exists():
            return False
        self.path.unlink()
        return True

    async def dump(self, state: AgentState) -> Path:
        """Write *state* to ``self.path`` in JSONL format."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = UserMsg(
            name="__state__",
            content=state.summary or "",
            metadata={k: getattr(state, k) for k in _META_KEYS},
        )
        tmp_path = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
            await f.write(header.model_dump_json() + "\n")
            for msg in state.context:
                await f.write(msg.model_dump_json() + "\n")
        os.replace(tmp_path, self.path)
        return self.path

    async def load(self) -> AgentState:
        """Read an AgentState back from ``self.path``."""
        async with aiofiles.open(self.path, encoding="utf-8") as f:
            lines = (await f.read()).splitlines()
        if not lines:
            return AgentState()

        header = Msg.model_validate_json(lines[0])
        summary: str | list = (
            list(header.content)
            if any(getattr(b, "type", None) == "data" for b in header.content)
            else header.get_text_content() or ""
        )

        metadata = header.metadata or {}
        return AgentState(
            **{k: metadata.get(k, d) for k, d in [("session_id", ""), ("reply_id", ""), ("cur_iter", 0)]},
            summary=summary,
            context=[Msg.model_validate_json(line) for line in lines[1:] if line.strip()],
        )
