"""User-owned, one-shot proactive tasks and their durable execution progress."""

from typing import Literal
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, field_validator


class PaTempTask(BaseModel):
    """Persist an independent delegation and its single execution attempt."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid4().hex, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    request_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,128}$")
    request_signature: str | None = None
    query: str = Field(min_length=1)
    run_at: AwareDatetime
    source_session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    created_at: AwareDatetime
    updated_at: AwareDatetime
    enabled: StrictBool = True
    revision: int = Field(default=0, ge=0, le=2147483647)
    status: Literal[
        "scheduled",
        "running",
        "sending",
        "paused",
        "completed",
        "failed",
        "cancelled",
    ] = "scheduled"
    agent_session_id: str | None = None
    queried: bool = False
    result_path: str | None = None
    message_id: str | None = None
    started_at: AwareDatetime | None = None
    finished_at: AwareDatetime | None = None
    delayed: bool = False
    error: str | None = None

    @property
    def runnable(self) -> bool:
        """Resume only enabled tasks with unfinished work."""
        return self.enabled and self.status in {"scheduled", "running", "sending"}

    @field_validator("query")
    @classmethod
    def validate_query(cls, value):
        """Keep one-shot work nonblank and free of surrounding whitespace."""
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()
