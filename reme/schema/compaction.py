"""JSON contracts for recoverable, tool-only context compaction."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


class CompactionOptions(BaseModel):
    """Decision request limits and optional host-counted output budget."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    drop_threshold: float = Field(default=0.1, ge=0, le=1)
    preserve_recent_messages: int = Field(default=6, ge=0, strict=True)
    max_state_tokens: int = Field(default=25000, gt=0, strict=True)
    max_request_tokens: int = Field(default=30000, gt=0, strict=True)
    evidence_chars: int = Field(default=2000, gt=0, strict=True)
    preview_chars: int = Field(default=300, ge=0, strict=True)
    max_output_tokens: int | None = Field(default=None, gt=0, strict=True)
    max_concurrency: int = Field(default=4, ge=1, le=32, strict=True)
    total_timeout_seconds: float = Field(default=30, gt=0)


class CompactionRequest(BaseModel):
    """The host supplies immutable AgentScope snapshots and resolvable archive references."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    messages: list[dict[str, JsonValue]]
    goal: str = ""
    next_action: str = ""
    source_refs: dict[str, str] = Field(default_factory=dict)
    protected_message_ids: list[str] = Field(default_factory=list)
    protected_call_ids: list[str] = Field(default_factory=list)
    eligible_tool_names: list[str] = Field(default_factory=list)
    options: CompactionOptions = Field(default_factory=CompactionOptions)

    @field_validator("source_refs")
    @classmethod
    def validate_refs(cls, refs):
        """References are opaque host identifiers, not paths ReMe opens."""
        if any(not key.strip() or not value.strip() for key, value in refs.items()):
            raise ValueError("Source references need non-empty call IDs and values.")
        return refs

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, messages):
        """Reject ambiguous envelopes while retaining unknown block fields."""
        for message in messages:
            if message.get("role") not in ("user", "assistant", "system"):
                raise ValueError("Each message needs a user, assistant, or system role.")
            content = message.get("content")
            if not isinstance(content, list) or any(not isinstance(block, dict) for block in content):
                raise ValueError("Each message needs a list of content block objects.")
        return messages


class CallDecision(BaseModel):
    """Every replacement identifies its original tool pair and recoverable source."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    id: str
    call_id: str
    action: Literal["keep", "preview"]
    reason: str
    probability: float | None = Field(default=None, ge=0, le=1)
    source_ref: str | None = None


class CompactionResult(BaseModel):
    """A proposed snapshot; the host owns archiving, recovery, and commit."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal[2] = 2
    messages: list[dict[str, JsonValue]]
    decisions: list[CallDecision] = Field(default_factory=list)
    changed: bool
    reason: Literal["compacted", "no_candidates", "no_change", "budget_exceeded"]
    budget_met: bool | None = None
    stats: dict[str, JsonValue] = Field(default_factory=dict)
