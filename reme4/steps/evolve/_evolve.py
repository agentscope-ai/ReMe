"""Shared helpers for evolve steps."""

import datetime
import zoneinfo
from typing import Literal

from agentscope.agent import ReActAgent
from agentscope.message import Msg


def now(timezone: str | None = None) -> datetime.datetime:
    """Return current datetime in the given IANA timezone, falling back to local."""
    if not timezone:
        return datetime.datetime.now()
    try:
        return datetime.datetime.now(zoneinfo.ZoneInfo(timezone))
    except Exception:
        return datetime.datetime.now()


def format_history(messages: list[Msg], include_timestamp: bool = True) -> str:
    """Render a conversation slice as a human-readable transcript."""
    lines: list[str] = []
    for msg in messages:
        text = (msg.get_text_content() or "").strip()
        if not text:
            continue
        speaker = msg.name or msg.role or "?"
        header = f"[{speaker} @ {msg.timestamp}]" if include_timestamp else f"[{speaker}]"
        lines.append(f"{header}\n{text}")
    return "\n\n".join(lines) or "(empty)"


class FlexReActAgent(ReActAgent):
    """ReActAgent subclass that allows structured output without forcing tool_choice='required'."""

    async def _reasoning(
        self,
        tool_choice: Literal["auto", "none", "required"] | None = None,
    ) -> Msg:
        if tool_choice == "required":
            tool_choice = None
        return await super()._reasoning(tool_choice)
