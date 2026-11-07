"""File-native scheduled tasks and append-only execution events."""

from typing import Any, Literal
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, field_validator


class PaTask(BaseModel):
    """A stable task identity with a five-field cron schedule."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid4().hex, pattern=r"^[A-Za-z0-9_-]{1,128}$")
    cron: str
    query: str = Field(min_length=1)
    enabled: StrictBool = True
    created_at: AwareDatetime | None = None

    @field_validator("cron")
    @classmethod
    def validate_cron(cls, value: str) -> str:
        """Keep schedule syntax consistent with minute-based task definitions."""
        from croniter import croniter

        if len(value.split()) != 5 or not croniter.is_valid(value):
            raise ValueError("cron must be a valid five-field expression")
        return value

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Reject whitespace-only tasks."""
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()


class PaTaskRun(BaseModel):
    """Start and finish events share the same run_id and scheduled_at."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default_factory=lambda: uuid4().hex)
    task_id: str
    cron: str
    query: str
    scheduled_at: AwareDatetime
    started_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    status: Literal["running", "succeeded", "failed", "interrupted"] = "running"
    result: Any = None
    error: str | None = None
    agent_session_id: str | None = None
