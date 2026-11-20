"""Client messages projected from native history and live execution alike."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .stream_chunk import StreamStatus


class ClientModel(BaseModel):
    """Every serialized field belongs to the public client contract."""

    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


ItemStatus = Literal["running", "completed", "failed", "interrupted", "waiting", "unknown"]


class TextItem(ClientModel):
    """One ordered text block."""

    id: str
    status: ItemStatus
    type: Literal["text"] = "text"
    text: str
    phase: Literal["commentary", "final_answer"] | None = None


class SummaryItem(ClientModel):
    """A public reasoning summary or plan, never private reasoning."""

    id: str
    status: ItemStatus
    type: Literal["summary"] = "summary"
    text: str
    kind: Literal["reasoning", "plan"]


class ToolAction(ClientModel):
    """Provider-independent tool semantics used by the client."""

    kind: Literal["read", "list", "search", "command", "edit", "webSearch", "browse", "image", "tool"]
    target: str | None = None
    query: str | None = None


class ToolItem(ClientModel):
    """A tool call and its result occupy the same ordered block."""

    id: str
    status: ItemStatus
    type: Literal["tool"] = "tool"
    name: str
    input: str = ""
    output: str = ""
    actions: list[ToolAction] = Field(default_factory=list)


class ArtifactItem(ClientModel):
    """A displayable attachment."""

    id: str
    status: ItemStatus
    type: Literal["artifact"] = "artifact"
    label: str
    url: str
    media_type: str


TranscriptItem = Annotated[TextItem | SummaryItem | ToolItem | ArtifactItem, Field(discriminator="type")]


class AgentMessage(ClientModel):
    """A complete replacement, shared by subscriptions and task-question streams."""

    id: str
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    source: Literal["chat", "proactive"] = "chat"
    created_at: str | None = None
    turn_id: str | None = None
    completed: bool = True
    delivery: Literal["idle", "immediate"] = "idle"
    origin_session_id: str | None = None
    notification_eligible: bool = False
    items: list[TranscriptItem] = Field(default_factory=list)
    outcome: StreamStatus | None = None
    error: str | None = None
