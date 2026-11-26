"""Generic durable inbox items and execution receipts."""

from hashlib import sha256

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, model_validator

from .event import Event
from .proactive_runtime import AgentName


class InboxItem(BaseModel):
    """A stable identity with JSON content owned by the input producer."""

    model_config = ConfigDict(extra="allow", frozen=True, allow_inf_nan=False, revalidate_instances="always")

    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)
    id: str = Field(min_length=1, max_length=128)


class InboxReceipt(BaseModel):
    """The durable commit point for one processed input batch."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    item_ids: list[str] = Field(min_length=1)
    consume_ts: AwareDatetime
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class NotificationRef(BaseModel):
    """Stable delivery identity; reminder text and priority remain in the daily record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=1, ge=1, le=1)
    id: str = Field(min_length=1, max_length=128)
    source_agent: AgentName
    record_path: str
    created_at: AwareDatetime

    @staticmethod
    def identity(path: str) -> str:
        """Keep long workspace paths out of queue identifiers."""
        Event.validate_paths([path])
        return "notification-" + sha256(path.encode()).hexdigest()

    @model_validator(mode="after")
    def validate_identity(self):
        """Prevent one path from being retried under a different queue identity."""
        if self.id != self.identity(self.record_path):
            raise ValueError("notification id must identify its source record")
        return self

    @classmethod
    def from_record(cls, source_agent, record_path, created_at):
        """Freeze the reference once the source input batch is committed."""
        return cls(
            id=cls.identity(record_path),
            source_agent=source_agent,
            record_path=record_path,
            created_at=created_at,
        )
