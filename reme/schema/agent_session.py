"""Backend-neutral session history and activity contracts."""

from typing import Literal
from hashlib import sha256

from pydantic import BaseModel, Field, computed_field

from .stream_chunk import StreamChunk


class SessionStatus(BaseModel):
    """Activity observed by the session owner, independently of an HTTP stream."""

    session_id: str
    status: Literal["active", "idle", "not_loaded", "error"]
    active_flags: list[str] = Field(default_factory=list)


class SessionMessage(BaseModel):
    """One displayed message; proactive provenance survives session reloads."""

    id: str
    role: Literal["user", "assistant"]
    content: str
    source: Literal["chat", "proactive"] = "chat"
    created_at: str | None = None
    turn_id: str | None = None
    completed: bool = True
    delivery: Literal["idle", "immediate"] = "idle"
    origin_session_id: str | None = None
    chunks: list[StreamChunk] = Field(default_factory=list)


class SessionSnapshot(BaseModel):
    """Ordered display history, including messages absent from native UI history."""

    session_id: str
    state: SessionStatus
    messages: list[SessionMessage] = Field(default_factory=list)
    path: str | None = None
    last_turn_id: str | None = None

    @computed_field
    @property
    def revision(self) -> str:
        """Opaque content guard, stable across reloads and independent of activity status."""
        digest = sha256((self.last_turn_id or "").encode("utf-8"))
        for message in self.messages:
            digest.update(b"\n")
            digest.update(message.model_dump_json().encode("utf-8"))
        return digest.hexdigest()
