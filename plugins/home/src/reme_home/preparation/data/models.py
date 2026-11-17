"""Normalized benchmark data persisted by Home preparation."""

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1
DatasetName = Literal["beam", "longmemeval"]


def utc_now() -> datetime:
    """Return an aware creation timestamp."""
    return datetime.now(timezone.utc)


class ContractModel(BaseModel):
    """Reject unknown persisted fields."""

    model_config = ConfigDict(extra="forbid")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode normalized values deterministically."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def fingerprint(value: Any) -> str:
    """Hash canonical content."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


class QuerySpec(ContractModel):
    """A question, reference answer and benchmark-specific scoring metadata."""

    query_id: str = Field(min_length=1)
    question: str
    golden_answer: Any
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionSpec(ContractModel):
    """An ordered conversation session with normalized message timestamps."""

    session_id: str = Field(min_length=1)
    messages: list[dict[str, Any]]
    metadata: dict[str, Any] = Field(default_factory=dict)


class CaseSpec(ContractModel):
    """One selectable benchmark case with its sessions and questions."""

    case_id: str = Field(min_length=1)
    sessions: list[SessionSpec]
    queries: list[QuerySpec]
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetManifest(ContractModel):
    """Selection identity, converter version and normalized content fingerprints."""

    # Accept legacy files without emitting a version that has no reader dispatch.
    schema_version: int = Field(default=SCHEMA_VERSION, ge=1, exclude=True)
    dataset: DatasetName
    converter_version: int = 1
    variant: str | None = None
    case_ids: list[str]
    source_fingerprint: str = Field(min_length=1)
    normalized_fingerprint: str = Field(min_length=1)
    case_count: int = Field(ge=0)
    query_count: int = Field(ge=0)
    created_at: datetime = Field(default_factory=utc_now)
