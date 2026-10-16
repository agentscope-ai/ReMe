"""Bounded, JSON-native contracts for optional task-time memory assistance."""

import time
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class _ContextModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_assignment=True)


class ContextTaskSnapshot(_ContextModel):
    """Host-visible task state; never include credentials or hidden reasoning."""

    goal: str = Field(min_length=1, max_length=6000)
    recent_context: str = Field(default="", max_length=12000)
    current_tool: str = Field(default="", max_length=2000)
    tool_result_excerpt: str = Field(default="", max_length=4000)
    query_terms: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(default_factory=list, max_length=8)
    seen_evidence: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(
        default_factory=list,
        max_length=128,
    )


class ContextEvent(_ContextModel):
    """Immutable-by-copy snapshot published at a host execution boundary."""

    session_id: str = Field(min_length=1, max_length=200)
    task_id: str = Field(min_length=1, max_length=200)
    event_id: str = Field(min_length=1, max_length=200)
    revision: int = Field(ge=0, strict=True)
    event_type: Literal["task_start", "tool_start", "tool_result"]
    task_snapshot: ContextTaskSnapshot
    created_at: float = Field(default_factory=time.time, gt=0)


class ContextAgentOptions(_ContextModel):
    """Per-evaluation bounds; session-wide limits belong to the coordinator."""

    max_candidates: int = Field(default=8, ge=1, le=16, strict=True)
    max_hint_candidates: int = Field(default=2, ge=1, le=4, strict=True)
    max_request_tokens: int = Field(default=4000, ge=1000, le=32000, strict=True)
    max_excerpt_chars: int = Field(default=1200, ge=100, le=4000, strict=True)
    max_file_bytes: int = Field(default=1024 * 1024, ge=1024, le=20 * 1024 * 1024, strict=True)
    total_timeout_ms: int = Field(default=800, ge=1, le=60000, strict=True)
    candidate_threshold: float = Field(default=0.7, ge=0, le=1)
    intervene_threshold: float = Field(default=0.7, ge=0, le=1)
    vector_weight: float = Field(default=0.7, ge=0, le=1)
    hint_ttl_seconds: float = Field(default=30, gt=0, le=300)


class MemoryCandidate(_ContextModel):
    """Evidence read from a permission-checked source, rather than cached text."""

    candidate_id: str
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    content_version: str
    excerpt: str


class ContextHint(_ContextModel):
    """A source-backed suggestion; hosts insert it as evidence, not instruction."""

    hint_id: str
    session_id: str
    task_id: str
    based_on_revision: int = Field(ge=0)
    expires_at: float
    evidence: list[MemoryCandidate] = Field(min_length=1, max_length=4)
    text: str


class ContextEvaluationResult(_ContextModel):
    """Complete primary Job answer, including successful silent decisions."""

    schema_version: Literal[1] = 1
    event_id: str
    task_id: str
    based_on_revision: int
    action: Literal["silent", "hint"] = "silent"
    reason_code: str
    hint: ContextHint | None = None
    stats: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_action(self):
        """Prevent callers from treating an incomplete hint as deliverable."""
        if (self.action == "hint") != (self.hint is not None):
            raise ValueError("hint must be present exactly when action is hint")
        return self
