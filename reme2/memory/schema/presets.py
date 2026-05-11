"""Memory presets — common axis combinations for hot-write paths.

Hot-write tools (`sync`) and cold-write services (Ingestor R-M-W) start
with a preset, layer the agent's input on top, then validate the merged
dict against `Memory`. Adding a new memory shape = adding one preset
entry here + (optionally) a `LEGACY_AXES_FROM_CATEGORY` row in memory.py
for migration.

Presets carry only the 4 axes + optional default `status` for streaming
memories. Identity fields (title, description, tags, created, updated)
come from the caller.

Values are stored as **plain strings** (not StrEnum members) so they
flow cleanly through `frontmatter.dumps` → `yaml.dump`, which doesn't
know how to represent enum subclasses. Pydantic still coerces them
back into StrEnum members when `Memory.model_validate` runs.
"""

from __future__ import annotations

from .memory import Lifecycle, Role, Scope, Source, Status


EVENT_PRESET: dict = {
    "lifecycle": Lifecycle.STREAMING.value,
    "scope": Scope.INSTANCE.value,
    "source": Source.AUTO.value,
    "role": Role.OBSERVATION.value,
    "status": Status.ACTIVE.value,
    # Legacy compat: the `category` field is preserved by extra="allow",
    # but emit it explicitly so old tooling that still reads `category`
    # (hooks, scripts, downstream consumers) keeps working.
    "category": "event",
}

PROFILE_PRESET: dict = {
    "lifecycle": Lifecycle.EVOLVING.value,
    "scope": Scope.CLASS.value,
    "source": Source.CURATED.value,
    "role": Role.PROFILE.value,
    "category": "profile",
}

CONCEPT_PRESET: dict = {
    "lifecycle": Lifecycle.EVOLVING.value,
    "scope": Scope.CLASS.value,
    "source": Source.CURATED.value,
    "role": Role.CONCEPT.value,
    "category": "concept",
}

THESIS_PRESET: dict = {
    "lifecycle": Lifecycle.EVOLVING.value,
    "scope": Scope.CLASS.value,
    "source": Source.CURATED.value,
    "role": Role.CLAIM.value,
    "category": "thesis",
}

MODEL_PRESET: dict = {
    "lifecycle": Lifecycle.EVOLVING.value,
    "scope": Scope.CLASS.value,
    "source": Source.CURATED.value,
    "role": Role.CLAIM.value,
    "category": "model",
}

QUESTIONS_PRESET: dict = {
    "lifecycle": Lifecycle.EVOLVING.value,
    "scope": Scope.CLASS.value,
    "source": Source.CURATED.value,
    "role": Role.QUESTION.value,
    "category": "questions",
}

METHOD_PRESET: dict = {
    "lifecycle": Lifecycle.EVOLVING.value,
    "scope": Scope.CLASS.value,
    "source": Source.CURATED.value,
    "role": Role.METHOD.value,
    "category": "method",
}

TOOL_PRESET: dict = {
    "lifecycle": Lifecycle.EVOLVING.value,
    "scope": Scope.CLASS.value,
    "source": Source.CURATED.value,
    "role": Role.REFERENCE.value,
    "category": "tool",
}

FUNDAMENTALS_PRESET: dict = {
    "lifecycle": Lifecycle.EVOLVING.value,
    "scope": Scope.CLASS.value,
    "source": Source.CURATED.value,
    "role": Role.FUNDAMENTALS.value,
    "category": "fundamentals",
}

MATERIAL_PRESET: dict = {
    "lifecycle": Lifecycle.FROZEN.value,
    "scope": Scope.INSTANCE.value,
    "source": Source.AUTO.value,
    "role": Role.REFERENCE.value,
    "category": "material",
}


# Old `category` → preset, for migration / lookup use.
PRESETS_BY_CATEGORY: dict[str, dict] = {
    "event": EVENT_PRESET,
    "profile": PROFILE_PRESET,
    "company": CONCEPT_PRESET,
    "sector": CONCEPT_PRESET,
    "concept": CONCEPT_PRESET,
    "thesis": THESIS_PRESET,
    "model": MODEL_PRESET,
    "questions": QUESTIONS_PRESET,
    "method": METHOD_PRESET,
    "tool": TOOL_PRESET,
    "fundamentals": FUNDAMENTALS_PRESET,
    "material": MATERIAL_PRESET,
}


def preset_for_category(category: str) -> dict | None:
    """Look up the preset bound to a legacy category name.

    Returns a fresh dict each call (callers may mutate it). Returns
    None for unknown categories — caller decides whether to refuse or
    fall through to a generic shape.
    """
    p = PRESETS_BY_CATEGORY.get(category)
    return dict(p) if p is not None else None
