"""Cursor-based notifications for the application's current PA session."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .agent_session import SessionMessage, SessionSnapshot
from .agent_activity import AgentActivity
from .pa_proactive import PaNotificationWindow


class PaEvent(BaseModel):
    """A replayable change or a connection-local snapshot, reset, or heartbeat.

    Cursors are opaque and scoped to one runtime epoch. Only snapshots and
    replayable changes advance the client's cursor. Messages are complete
    replacements keyed by session ID and message ID, rather than token deltas.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["snapshot", "session_changed", "message_upsert", "status_changed", "reset", "heartbeat"]
    cursor: str | None = None
    session_id: str | None = None
    status: Literal["active", "idle", "not_loaded", "error"] | None = None
    active_flags: list[str] = Field(default_factory=list)
    activities: dict[Literal["chat", "proactive"], AgentActivity] = Field(default_factory=dict)
    task_activities: dict[str, AgentActivity] = Field(default_factory=dict)
    notifications: PaNotificationWindow | None = None
    message: SessionMessage | None = None
    snapshot: SessionSnapshot | None = None
    inbox_messages: list[SessionMessage] | None = None
    reason: str | None = None
    error: str | None = None
