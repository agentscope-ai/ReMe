"""Append-only reminder history records."""

from uuid import uuid4
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

ReminderKind = Literal["normal", "morning", "midday", "evening"]


class ReminderRecord(BaseModel):
    """A reminder snapshot; later JSONL events with the same ID replace its state."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid4().hex)
    content: str = Field(min_length=1, max_length=200)
    kind: ReminderKind
    status: Literal["pending", "sent", "discarded", "failed"] = "pending"
    created_at: AwareDatetime
    updated_at: AwareDatetime
    expires_at: AwareDatetime
    sent_at: AwareDatetime | None = None
    session_id: str | None = None
    reason: str | None = None
    error: str | None = None
