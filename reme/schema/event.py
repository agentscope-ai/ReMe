"""Immutable observations; consumption receipts are stored separately."""

from pathlib import PurePosixPath, PureWindowsPath
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, field_validator


class Event(BaseModel):
    """A small observation referencing workspace files, rather than embedding their contents."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False, revalidate_instances="always")

    id: str = Field(default_factory=lambda: uuid4().hex, min_length=1, max_length=128)
    event_ts: AwareDatetime
    source: str = Field(min_length=1)
    description: str
    paths: list[str] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("paths")
    @classmethod
    def validate_paths(cls, values: list[str]) -> list[str]:
        """Keep resource references workspace-relative and portable."""
        for value in values:
            path = PurePosixPath(value)
            if any(
                (
                    not value,
                    value.startswith("~"),
                    "\\" in value,
                    PureWindowsPath(value).anchor,
                    path.is_absolute(),
                    ".." in path.parts,
                    path.as_posix() != value,
                    value == ".",
                ),
            ):
                raise ValueError("paths must contain normalized workspace-relative file paths")
        return values


class EventConsumption(BaseModel):
    """One durable acknowledgement of a batch within a single log segment."""

    model_config = ConfigDict(extra="forbid")

    event_ids: list[str] = Field(min_length=1)
    consume_ts: AwareDatetime
