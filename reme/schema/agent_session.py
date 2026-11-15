"""Backend-neutral session history and activity contracts."""

from hashlib import sha256
import json
from typing import Literal

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
        """Opaque content version, independent of activity and stable across reloads."""
        value = {
            "last_turn_id": self.last_turn_id,
            "messages": [message.model_dump(mode="json") for message in self.messages],
        }
        return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
