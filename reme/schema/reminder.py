"""Append-only reminder history records."""

from uuid import uuid4
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

ReminderKind = Literal["normal", "morning", "midday", "evening"]


class ReminderRecord(BaseModel):
    """A reminder snapshot; later JSONL events with the same ID replace its state."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid4().hex)
    content: str = Field(min_length=1, max_length=4000)
    kind: ReminderKind
    delivery: Literal["idle", "immediate"] = "idle"
    task_id: str | None = None
    task_revision: int | None = None
    task_definition: dict | None = None
    announced_at: AwareDatetime | None = None
    feedback_version: str | None = None
    status: Literal["pending", "sent", "discarded", "failed"] = "pending"
    created_at: AwareDatetime
    updated_at: AwareDatetime
    expires_at: AwareDatetime | None = None
    sent_at: AwareDatetime | None = None
    session_id: str | None = None
    reason: str | None = None
    error: str | None = None
    retry_at: AwareDatetime | None = None
    result_path: str | None = None

    @model_validator(mode="after")
    def validate_expiration(self):
        """Require an expiry for ordinary proactive messages."""
        if self.delivery == "idle" and self.expires_at is None:
            raise ValueError("Ordinary proactive messages require an expiration")
        return self
