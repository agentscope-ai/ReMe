"""Proactive refresh schemas.

Defines the topic model (v2), the LLM extract contract, the chain-shared
context state, and the ``daily/_proactive.yaml`` truth-source file model.
See ``PROACTIVE_SPEC.md`` sections F1/A2/A3 for the full contracts.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator

TOPIC_KINDS = ("follow_up", "interest_extend")
UPDATE_ACTIONS = ("keep", "update", "resolve")


def clamp_confidence(value) -> float:
    """Coerce confidence into [0, 1]; any conversion failure falls back to 0.5."""
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


class ProactiveTopic(BaseModel):
    """One proactive topic; every field has a default so v1 files parse seamlessly.

    Fallback rules (A2): invalid ``kind`` -> ``interest_extend``; unparseable
    ``confidence`` -> 0.5. Missing ``id``, ``first_seen`` and
    ``last_evidence_at`` are context-dependent and therefore resolved by the
    loaders, not here.
    """

    id: str = ""
    title: str = ""
    reason: str = ""
    kind: str = "interest_extend"
    confidence: float = 0.5
    first_seen: str = ""
    last_evidence_at: str = ""
    evidence: str = ""
    keywords: list[str] = Field(default_factory=list)
    paths: list[str] = Field(default_factory=list)

    @field_validator("kind", mode="before")
    @classmethod
    def _fallback_kind(cls, value):
        text = str(value or "").strip()
        return text if text in TOPIC_KINDS else "interest_extend"

    @field_validator("confidence", mode="before")
    @classmethod
    def _fallback_confidence(cls, value):
        return clamp_confidence(value)

    @field_validator("keywords", "paths", mode="before")
    @classmethod
    def _clean_str_list(cls, value):
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]


class TopicUpdate(BaseModel):
    """LLM verdict for one carried-forward topic (F2.3)."""

    id: str = ""
    action: str = "keep"
    evidence: str = ""
    reason: str = ""
    confidence: float | None = None

    @field_validator("action", mode="before")
    @classmethod
    def _fallback_action(cls, value):
        text = str(value or "").strip()
        return text if text in UPDATE_ACTIONS else "keep"

    @field_validator("confidence", mode="before")
    @classmethod
    def _clean_confidence(cls, value):
        if value is None or str(value).strip() == "":
            return None
        return clamp_confidence(value)


class ProactiveExtractOutput(BaseModel):
    """Structured output contract for ``proactive_extract_step`` (A3)."""

    follow_ups: list[ProactiveTopic] = Field(default_factory=list)
    extends: list[ProactiveTopic] = Field(default_factory=list)
    updates: list[TopicUpdate] = Field(default_factory=list)


class ProactiveState(BaseModel):
    """Chain-shared proactive context state (``context['proactive']``).

    Extract fills material/carry-forward/LLM output fields; topics fills the
    filtering/rendering fields; finish records the catalog checkpoint.
    ``file_skip_reason`` is metadata/log only and never persisted to
    interests.yaml (v5 simplification R7).
    """

    date: str = ""
    daily_dir: str = "daily"
    workspace: str = ""
    scan_days: int = 2
    carry_forward_days: int = 14
    material_paths: list[str] = Field(default_factory=list)
    changed_paths: list[str] = Field(default_factory=list)
    resource_paths: list[str] = Field(default_factory=list)
    carry_forward_all: list[ProactiveTopic] = Field(default_factory=list)
    carry_forward_prompt: list[ProactiveTopic] = Field(default_factory=list)
    llm_calls: int = 0
    follow_ups: list[dict] = Field(default_factory=list)
    extends: list[dict] = Field(default_factory=list)
    updates: list[dict] = Field(default_factory=list)
    early_exit: str = ""
    updates_applied: int = 0
    updates_resolved: int = 0
    candidates_in: int = 0
    candidates: list[dict] = Field(default_factory=list)
    dropped_missing: int = 0
    dropped_duplicate: int = 0
    dropped_known: int = 0
    topics_out: list[dict] = Field(default_factory=list)
    push: bool = False
    file_skip_reason: str = ""
    interests_path: str = ""
    interests_written: bool = False
    checkpoint_paths: list[str] = Field(default_factory=list)
    duration_ms: int = 0
    errors: list[str] = Field(default_factory=list)


class ProactiveStateFile(BaseModel):
    """On-disk truth-source ``daily/_proactive.yaml`` (F1.3, v5: 3 sections).

    ``resolved`` tombstones carry ``first_seen`` so a resurrected topic can
    keep its original age anchor (F2.4 reopen channel).
    """

    model_config = ConfigDict(extra="ignore")

    version: int = 1
    open_topics: list[ProactiveTopic] = Field(default_factory=list)
    resolved: list[dict] = Field(default_factory=list)


class ProactiveResult(BaseModel):
    """Result of reading daily interest topics (F5)."""

    date: str = ""
    path: str = ""
    topics: list[dict] = Field(default_factory=list)
    content: str = ""
    skipped: bool = False
    error: str = ""
    summary: str = ""
    push: bool | None = None
    generated_at: str = ""
