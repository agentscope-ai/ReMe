"""Memory — the typed shape of a vault frontmatter.

Single class covers every memory in the vault. The four orthogonal axes
(`lifecycle / scope / source / role`) define the *behavioral shape* —
services read off them to decide decay rules, retrieval ranking, merge
eligibility, etc., instead of switching on a `category` string.

Role-conditional fields (`confidence`, `status`, `origin_session_id`)
are validated declaratively via `model_validator(mode='after')`.

Legacy migration: a `model_validator(mode='before')` recognizes the old
`category` field and back-fills the 4 axes via `LEGACY_AXES_FROM_CATEGORY`,
so vault files written before this schema continue to parse without a
migration script.

Lives in `reme2/memory/schema/` (with the services that consume it),
NOT in `reme2/component/` — the engine is domain-agnostic and never
imports this module.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# The four behavioral axes
# ---------------------------------------------------------------------------


class Lifecycle(StrEnum):
    """Mutability + decay disposition of a memory.

    streaming — write-once events that decay/archive after a freshness window.
    evolving  — long-lived memories that get continuously edited; never decay.
    frozen    — immutable references / artifacts; never edited, never decay.
    """

    STREAMING = "streaming"
    EVOLVING = "evolving"
    FROZEN = "frozen"


class Scope(StrEnum):
    """Referential range of a memory.

    instance — refers to a specific moment / object / occurrence.
    class    — abstracts over many instances; a concept / role / pattern.
    """

    INSTANCE = "instance"
    CLASS = "class"


class Source(StrEnum):
    """Provenance of the memory.

    auto    — captured by the system from a session / tool output.
    curated — written or shaped by a human (or LLM curator) on purpose.
    derived — computed from other memories (Maintainer products, summaries).
    """

    AUTO = "auto"
    CURATED = "curated"
    DERIVED = "derived"


class Role(StrEnum):
    """Cognitive role the memory plays.

    Drives schema-aware ranking + role-specific field validation.
    Adding a new role is a one-line change here plus (optionally) a preset.
    """

    OBSERVATION = "observation"   # what happened (events live here)
    CLAIM = "claim"               # an assertion needing confidence (thesis, model)
    QUESTION = "question"         # an open inquiry needing an answer
    PROFILE = "profile"           # entity description (person, org, system)
    CONCEPT = "concept"           # abstract idea / definition (company, sector, concept)
    METHOD = "method"             # procedure / how-to
    REFERENCE = "reference"       # pointer to external thing (tool, paper, code)
    FUNDAMENTALS = "fundamentals" # foundational data / baseline facts


class Status(StrEnum):
    """Lifecycle state — meaningful only for `lifecycle == streaming`.

    Topic-style memories (evolving / frozen) ignore this field.
    """

    ACTIVE = "active"
    DISTILLED = "distilled"
    ARCHIVED = "archived"


class Confidence(StrEnum):
    """Required for `role == claim`. Wire format stays emoji.

    PENDING  — claim under investigation; outcome not yet decided.
    VERIFIED — claim supported by evidence and currently held.
    REJECTED — claim was investigated and disconfirmed; kept for history.
    """

    PENDING = "⏳"
    VERIFIED = "✅"
    REJECTED = "❌"


# ---------------------------------------------------------------------------
# Legacy `category` → 4-axis mapping (auto-migration)
# ---------------------------------------------------------------------------


LEGACY_AXES_FROM_CATEGORY: dict[str, dict] = {
    # Events: streaming-instance-auto-observation
    "event": {
        "lifecycle": Lifecycle.STREAMING,
        "scope": Scope.INSTANCE,
        "source": Source.AUTO,
        "role": Role.OBSERVATION,
    },
    # Topic categories — all evolving / class / curated, role varies
    "company": {
        "lifecycle": Lifecycle.EVOLVING,
        "scope": Scope.CLASS,
        "source": Source.CURATED,
        "role": Role.CONCEPT,
    },
    "sector": {
        "lifecycle": Lifecycle.EVOLVING,
        "scope": Scope.CLASS,
        "source": Source.CURATED,
        "role": Role.CONCEPT,
    },
    "concept": {
        "lifecycle": Lifecycle.EVOLVING,
        "scope": Scope.CLASS,
        "source": Source.CURATED,
        "role": Role.CONCEPT,
    },
    "method": {
        "lifecycle": Lifecycle.EVOLVING,
        "scope": Scope.CLASS,
        "source": Source.CURATED,
        "role": Role.METHOD,
    },
    "tool": {
        "lifecycle": Lifecycle.EVOLVING,
        "scope": Scope.CLASS,
        "source": Source.CURATED,
        "role": Role.REFERENCE,
    },
    "profile": {
        "lifecycle": Lifecycle.EVOLVING,
        "scope": Scope.CLASS,
        "source": Source.CURATED,
        "role": Role.PROFILE,
    },
    "thesis": {
        "lifecycle": Lifecycle.EVOLVING,
        "scope": Scope.CLASS,
        "source": Source.CURATED,
        "role": Role.CLAIM,
    },
    "model": {
        "lifecycle": Lifecycle.EVOLVING,
        "scope": Scope.CLASS,
        "source": Source.CURATED,
        "role": Role.CLAIM,
    },
    "questions": {
        "lifecycle": Lifecycle.EVOLVING,
        "scope": Scope.CLASS,
        "source": Source.CURATED,
        "role": Role.QUESTION,
    },
    "fundamentals": {
        "lifecycle": Lifecycle.EVOLVING,
        "scope": Scope.CLASS,
        "source": Source.CURATED,
        "role": Role.FUNDAMENTALS,
    },
    # Materials: frozen-instance-auto-reference (siblings of an event index)
    "material": {
        "lifecycle": Lifecycle.FROZEN,
        "scope": Scope.INSTANCE,
        "source": Source.AUTO,
        "role": Role.REFERENCE,
    },
}


# ---------------------------------------------------------------------------
# Memory — the single typed shape
# ---------------------------------------------------------------------------


class Memory(BaseModel):
    """The typed view of a vault frontmatter.

    `extra="allow"` keeps domain-specific fields (market, ticker, etc.) in
    the parsed object without polluting this base schema. `populate_by_name`
    lets `originSessionId` (legacy camelCase) populate `origin_session_id`.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # -- Identity / common metadata ----------------------------------------

    title: str = Field(default="")
    description: str = Field(default="")
    tags: list[str] = Field(default_factory=list)
    created: date | None = None
    updated: date | None = None

    # -- The four behavioral axes ------------------------------------------

    lifecycle: Lifecycle
    scope: Scope
    source: Source
    role: Role

    # -- Cross-cutting graph fields ----------------------------------------

    topics: list[str] = Field(default_factory=list, description="Outbound wikilinks to class memories.")
    parent: str | None = Field(default=None, description="Wikilink to owning memory (e.g. material → event).")

    # -- Role / lifecycle / source-conditional fields ----------------------

    confidence: Confidence | None = Field(
        default=None,
        description="Required when role == claim. Use ⏳ / ✅ / ❌ as wire form.",
    )
    status: Status | None = Field(
        default=None,
        description="Lifecycle state. Meaningful only when lifecycle == streaming.",
    )
    origin_session_id: str | None = Field(
        default=None,
        alias="originSessionId",
        description="Capture session id. Set when source == auto.",
    )

    # -- Migration: pre-validate hook --------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_category(cls, data):
        """Back-fill 4 axes from old `category` field if present.

        Idempotent: if axes are already populated, do nothing. The legacy
        `category` field is preserved (via `extra="allow"`) so downstream
        code that still reads it keeps working until we delete it.
        """
        if not isinstance(data, dict):
            return data
        if "role" in data and "lifecycle" in data:
            return data
        cat = data.get("category")
        if cat in LEGACY_AXES_FROM_CATEGORY:
            data = {**LEGACY_AXES_FROM_CATEGORY[cat], **data}
        return data

    # -- Field coercion / cleanup ------------------------------------------

    @field_validator("created", "updated", mode="before")
    @classmethod
    def _coerce_date(cls, v):
        if v is None or isinstance(v, date):
            return v
        if isinstance(v, datetime):
            return v.date()
        if isinstance(v, str):
            return datetime.fromisoformat(v).date()
        return v

    @field_validator("tags", "topics", mode="before")
    @classmethod
    def _strip_dedup(cls, v):
        """Strip whitespace + dedup string lists. Preserves order."""
        if not isinstance(v, list):
            return v
        seen: dict[str, None] = {}
        for item in v:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    seen.setdefault(stripped, None)
        return list(seen.keys())

    # -- Role-conditional checks -------------------------------------------

    @model_validator(mode="after")
    def _enforce_role_conditionals(self):
        if self.role is Role.CLAIM and self.confidence is None:
            raise ValueError(
                "role='claim' requires explicit confidence "
                "(one of ⏳ / ✅ / ❌)"
            )
        return self

    # -- Convenience properties --------------------------------------------

    @property
    def is_active(self) -> bool:
        """True for streaming memories that are still mutable."""
        return self.lifecycle is Lifecycle.STREAMING and self.status is Status.ACTIVE

    @property
    def is_terminal(self) -> bool:
        """True for memories no longer accepting writes (distilled / archived / frozen)."""
        if self.lifecycle is Lifecycle.FROZEN:
            return True
        return self.status in (Status.DISTILLED, Status.ARCHIVED)
