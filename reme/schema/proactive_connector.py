"""Connector configuration, management requests and shared source views."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StrictBool, StrictStr, field_validator, model_validator

from .agent_message import ClientModel
from .event import Event
from .proactive_agent import validate_runtime_names, TIME_EVENT_SOURCE
from .proactive_runtime import AgentName
from .proactive_trace import SourceRef


class SSEConnectorBinding(BaseModel):
    """SSE is the transport; its data fields contain CloudEvents JSON."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["sse"] = "sse"
    url: HttpUrl


class ConnectorSource(BaseModel):
    """User-owned subscription definition, independent of delivery checkpoints."""

    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(pattern=r"^source_[0-9a-f]{32}$")
    name: StrictStr = Field(min_length=1, max_length=80)
    runtime_names: list[AgentName]
    binding: SSEConnectorBinding
    enabled: StrictBool = True

    @field_validator("runtime_names", mode="before")
    @classmethod
    def validate_targets(cls, value):
        """An empty target list means collection without event delivery."""
        return validate_runtime_names(value, allow_empty=True)


class ConnectorView(SourceRef, ClientModel):
    """One source representation shared by management and the flow overview."""

    model_config = ConfigDict(extra="forbid", frozen=False, json_schema_serialization_defaults_required=True)
    runtime_names: list[str] = Field(default_factory=list)
    enabled: bool | None = None
    available: bool | None = None
    last_event_id: str | None = None
    connected: bool | None = None
    binding: SSEConnectorBinding | None = None
    schedule: str | None = None
    next_trigger_at: str | None = None
    last_collected_at: str | None = None
    last_admitted_at: str | None = None
    pending_delivery_count: int | None = None
    last_error: str | None = None


class ManageConnectorRequest(BaseModel):
    """Four operations with disjoint inputs and partial updates."""

    model_config = ConfigDict(extra="forbid")
    action: Literal["list", "create", "update", "send"]
    source_id: str | None = Field(default=None, pattern=r"^(?:source_[0-9a-f]{32}|preset:[\w-]{1,80})$")
    name: StrictStr | None = Field(default=None, min_length=1, max_length=80)
    runtime_names: list[AgentName] | None = None
    enabled: StrictBool | None = None
    binding: SSEConnectorBinding | None = None
    events: list[Event] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_action(self):
        """Omitted update fields stay untouched; explicit null is not a patch value."""
        supplied = self.model_fields_set - {"action"}
        if any(getattr(self, key) is None for key in supplied):
            raise ValueError("Connector parameters cannot be null")
        if self.action == "list":
            if supplied:
                raise ValueError("list accepts only action")
        elif self.action == "create":
            if supplied != {"name", "binding", "runtime_names"}:
                raise ValueError("create requires name, binding and runtime_names")
            validate_runtime_names(self.runtime_names)
        elif self.action == "update":
            changes = supplied - {"source_id"}
            if not self.source_id or not changes or not changes <= {"name", "enabled", "runtime_names"}:
                raise ValueError("update requires source_id and changes to name, enabled or runtime_names")
            if "runtime_names" in changes:
                validate_runtime_names(self.runtime_names, allow_empty=True)
        else:
            if supplied != {"runtime_names", "events"}:
                raise ValueError("send requires runtime_names and events")
            validate_runtime_names(self.runtime_names)
            if any(event.source == TIME_EVENT_SOURCE for event in self.events):
                raise ValueError("time is a reserved framework source")
        return self
