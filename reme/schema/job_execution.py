"""Application-lifetime job execution snapshots."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

JobExecutionStatus = Literal["running", "succeeded", "failed", "stopped", "cancelled"]


class JobExecution(BaseModel):
    """One execution; step paths describe the currently active nested call."""

    id: int
    started_at: datetime
    finished_at: datetime | None = None
    status: JobExecutionStatus = "running"
    error: str | None = None
    step_path: list[str] = Field(default_factory=list)
    step_started_at: datetime | None = None
    failed_step: list[str] | None = None
