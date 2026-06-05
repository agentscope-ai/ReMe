"""AgentState JSONL dump / load.

Format:
  Line 1 — header Msg: AgentState.summary as content, state scalars in metadata.
  Lines 2+ — AgentState.context, one Msg per line.
"""

from pathlib import Path

import aiofiles
from agentscope.message import Msg, UserMsg
from agentscope.state import AgentState

_META_KEYS = ("session_id", "reply_id", "cur_iter")


class AsStateHandler:
    """Serialize / deserialize AgentState to a JSONL file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    async def dump(self, state: AgentState) -> Path:
        """Write *state* to ``self.path`` in JSONL format."""
        header = UserMsg(
            name="__state__",
            content=state.summary or "",
            metadata={k: getattr(state, k) for k in _META_KEYS},
        )
        async with aiofiles.open(self.path, "w", encoding="utf-8") as f:
            await f.write(header.model_dump_json() + "\n")
            for msg in state.context:
                await f.write(msg.model_dump_json() + "\n")
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

        return AgentState(
            **{k: header.metadata.get(k, d) for k, d in [("session_id", ""), ("reply_id", ""), ("cur_iter", 0)]},
            summary=summary,
            context=[Msg.model_validate_json(line) for line in lines[1:] if line.strip()],
        )
