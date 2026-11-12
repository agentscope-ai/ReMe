"""Transient agent execution state, independent of conversation content."""

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, model_validator


class AgentActivity(BaseModel):
    """One agent turn; task identity and start time also cover preparation."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    execution_session_id: str | None = None
    turn_id: str
    source: Literal["chat", "proactive", "task"] = "chat"
    phase: Literal["thinking", "working", "remembering", "answering", "waiting", "completed", "failed", "interrupted"]
    tool: str | None = None
    task_id: str | None = None
    started_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_task_identity(self):
        """Tasks must identify the run independently of their native session."""
        if self.source == "task" and (not self.task_id or self.started_at is None):
            raise ValueError("Task activity requires task_id and started_at")
        return self
