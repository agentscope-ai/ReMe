"""Personal-agent session registry and durable conversation cursors."""

from pydantic import BaseModel, ConfigDict, Field


class PaSessionCursor(BaseModel):
    """One recent chat session and its durable conversation-read position."""

    model_config = ConfigDict(extra="forbid", strict=True)

    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    last_active_at: str
    last_read_message_id: str | None = None
