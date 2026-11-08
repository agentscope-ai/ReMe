"""Transient agent execution state, independent of conversation content."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class AgentActivity(BaseModel):
    """One agent turn; task preparation uses run_id until native IDs are available."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    turn_id: str
    source: Literal["chat", "proactive", "task"] = "chat"
    phase: Literal["thinking", "working", "remembering", "answering", "waiting", "completed", "failed", "interrupted"]
    tool: str | None = None
    run_id: str | None = None
    task_id: str | None = None

    @model_validator(mode="after")
    def validate_task_identity(self):
        """Tasks must identify the run independently of their native session."""
        if self.source == "task" and (not self.run_id or not self.task_id):
            raise ValueError("Task activity requires run_id and task_id")
        return self
