"""Shared agent identities, native sessions, notifications and inbox contracts."""

from typing import Annotated, Literal
import re

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from ..constants import WINDOWS_RESERVED_DEVICE_NAMES
from .event import Event


def validate_agent_name(value: str) -> str:
    """Keep application identities portable as workspace directory names."""
    valid_length = isinstance(value, str) and 1 <= len(value) <= 80
    if (
        not valid_length
        or len(value.encode("utf-8")) > 192
        or not re.fullmatch(r"[\w-]+", value)
        or value.upper() in WINDOWS_RESERVED_DEVICE_NAMES
    ):
        raise ValueError("agent_name must contain only letters, numbers, underscores or hyphens")
    return value


AgentName = Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[\w-]+$"), AfterValidator(validate_agent_name)]
AgentType = Literal["chat", "proactive", "cron"]
ChatMessageId = Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]{1,128}$")]
InboxPriority = Literal["P0", "P1", "P2"]


class NotificationConfig(BaseModel):
    """Receiver-owned wakeup delays in seconds."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    p0_delay_seconds: float = Field(default=0, ge=0)
    p1_idle_seconds: float = Field(default=10, ge=0)
    p2_idle_seconds: float = Field(default=1800, ge=0)

    @field_validator("p0_delay_seconds", "p1_idle_seconds", "p2_idle_seconds", mode="before")
    @classmethod
    def numeric_seconds(cls, value):
        """Reject booleans and strings rather than treating them as durations."""
        if type(value) not in (int, float):
            raise ValueError("inbox delays must be numbers")
        return value


class AgentSession(BaseModel):
    """One durable native session reference; temporary branches never select the current session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=128)
    created_at: AwareDatetime
    temporary: bool = False
    signature: str | None = None


class ChatAgentConfig(BaseModel):
    """Startup settings for Remy's personal memory."""

    model_config = ConfigDict(extra="forbid")

    initial_memory: str = Field(default="", max_length=100000)


class AgentNotification(BaseModel):
    """Notification content shared by agent output and durable inbox messages."""

    model_config = ConfigDict(extra="forbid")

    priority: InboxPriority
    message: str = Field(min_length=1, max_length=100000)
    paths: list[str] = Field(min_length=1, max_length=1, description="Path to the source daily record.")

    @field_validator("message")
    @classmethod
    def nonblank_message(cls, value: str) -> str:
        """Require user-facing reminder text."""
        if not value.strip():
            raise ValueError("reminder message must not be blank")
        return value

    @field_validator("paths")
    @classmethod
    def relative_paths(cls, value: list[str]) -> list[str]:
        """Apply the same workspace-relative contract as event inputs."""
        return Event.validate_paths(value)


class AgentReminder(AgentNotification):
    """Reminder content and its delivery and Chat processing state."""

    delivered_at: AwareDatetime | None = None
    read_at: AwareDatetime | None = None
    chat_message_ids: list[ChatMessageId] = Field(default_factory=list)


class InboxWakeRequest(BaseModel):
    """A framework-selected batch eligible for automatic processing."""

    model_config = ConfigDict(extra="forbid")

    priority: InboxPriority
    msg_ids: list[str] = Field(min_length=1, max_length=100)

    @field_validator("msg_ids")
    @classmethod
    def relative_paths(cls, value: list[str]) -> list[str]:
        """Chat batches reference workspace-relative daily records."""
        return Event.validate_paths(value)


class SendChatMessageRequest(BaseModel):
    """Submit user-facing text, or an empty decision to stay quiet."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(max_length=100000)
