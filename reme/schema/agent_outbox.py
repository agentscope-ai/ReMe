"""Durable sending intentions, separate from receiver processing receipts."""

from typing import Literal
from hashlib import sha256

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from .agent_inbox import NotificationRef
from .event import Event
from .proactive_runtime import AgentName, ChatMessageId


class NotificationDelivery(BaseModel):
    """A committed record awaiting durable admission to a receiver Inbox."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    id: str = Field(min_length=1, max_length=128)
    notification: NotificationRef
    target_agent: AgentName
    status: Literal["prepared", "delivered"] = "prepared"
    created_at: AwareDatetime
    delivered_at: AwareDatetime | None = None
    last_error: str | None = None

    @model_validator(mode="after")
    def validate_delivery(self):
        """Require sending identities, payloads and terminal timestamps to agree."""
        expected = "delivery-" + sha256(f"{self.notification.id}\0{self.target_agent}".encode()).hexdigest()
        if self.id != expected:
            raise ValueError("Notification delivery identity does not match its payload")
        if (self.status == "delivered") != (self.delivered_at is not None):
            raise ValueError("Notification status and delivery time disagree")
        return self


class ChatDelivery(BaseModel):
    """A frozen assistant message whose native delivery can be recovered."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    id: str = Field(min_length=1, max_length=128)
    notification_ids: list[ChatMessageId] = Field(min_length=1, max_length=100)
    record_versions: dict[str, str] = Field(min_length=1)
    session_id: str = Field(min_length=1)
    chat_revision: int = Field(ge=0)
    session_revision: str | None = None
    session_turn_id: str | None = None
    content: str = Field(min_length=1, max_length=100000)
    message_id: ChatMessageId
    status: Literal["prepared", "sent", "acknowledged", "aborted"] = "prepared"
    created_at: AwareDatetime
    delivered_at: AwareDatetime | None = None
    acknowledged_at: AwareDatetime | None = None
    last_error: str | None = None

    @model_validator(mode="after")
    def validate_delivery(self):
        """Require sending identities, payloads and terminal timestamps to agree."""
        if len(set(self.notification_ids)) != len(self.notification_ids):
            raise ValueError("Notification ids must be distinct")
        if {NotificationRef.identity(path) for path in self.record_versions} != set(self.notification_ids):
            raise ValueError("Record versions must identify the selected notifications")
        key = "\0".join((self.session_id, *sorted(self.notification_ids)))
        if self.message_id != "reminder-" + sha256(key.encode()).hexdigest():
            raise ValueError("Native message identity does not match its batch")
        if (self.status in {"sent", "acknowledged"}) != (self.delivered_at is not None):
            raise ValueError("Chat status and delivery time disagree")
        if (self.status == "acknowledged") != (self.acknowledged_at is not None):
            raise ValueError("Chat status and acknowledgement time disagree")
        return self

    @field_validator("record_versions")
    @classmethod
    def validate_paths(cls, value):
        """Require canonical workspace-relative source references."""
        Event.validate_paths(list(value))
        return value

    @field_validator("content")
    @classmethod
    def nonblank(cls, value):
        """Reject a blank user-facing Chat reply."""
        if not value.strip():
            raise ValueError("Chat delivery content must not be blank")
        return value
