"""Transient agent execution state, independent of conversation content."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class AgentActivity(BaseModel):
    """One foreground turn or background proactive run observed by the application."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    turn_id: str
    source: Literal["chat", "proactive"] = "chat"
    phase: Literal["thinking", "working", "remembering", "answering", "waiting", "completed", "failed", "interrupted"]
    tool: str | None = None
