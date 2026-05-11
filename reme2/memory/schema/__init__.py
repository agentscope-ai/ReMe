"""MemoryFileNode Schema — vault domain "constitution".

Per `structure.md` §"全局架构蓝图", the Schema layer sits between the
domain-agnostic Core Engine and the three MemoryFileNode Services. Engines store
arbitrary markdown + open frontmatter dicts; Services read this Schema
to know what a "well-formed memory" is and shape the vault accordingly.

Public API:

    MemoryFileNode         — single typed class for every vault frontmatter
    Lifecycle / Scope / Source / Role  — the four behavioral axes (StrEnum)
    Status         — streaming-only state (active / distilled / archived)
    Confidence     — claim-only confidence (PENDING / VERIFIED / REJECTED)

    parse_frontmatter(raw) → (MemoryFileNode | None, errors)   tolerant parser

    *_PRESET dicts — common axis combos for hot-write paths
    preset_for_category(name) — legacy category → preset lookup

    LEGACY_AXES_FROM_CATEGORY — back-fill table; auto-applied by MemoryFileNode's
                                pre-validator so legacy vaults parse free.

Lives in `reme2/memory/` (not `reme2/schema/` and not `reme2/component/`)
because the Schema is owned by the memory services that consume it. The
engine never imports from here.
"""

from .memory import (
    LEGACY_AXES_FROM_CATEGORY,
    Confidence,
    Lifecycle,
    MemoryFileNode,
    Role,
    Scope,
    Source,
    Status,
)
from .parser import parse_frontmatter
from .presets import (
    CONCEPT_PRESET,
    EVENT_PRESET,
    FUNDAMENTALS_PRESET,
    MATERIAL_PRESET,
    METHOD_PRESET,
    MODEL_PRESET,
    PRESETS_BY_CATEGORY,
    PROFILE_PRESET,
    QUESTIONS_PRESET,
    THESIS_PRESET,
    TOOL_PRESET,
    preset_for_category,
)

__all__ = [
    # core
    "MemoryFileNode",
    "Lifecycle",
    "Scope",
    "Source",
    "Role",
    "Status",
    "Confidence",
    "LEGACY_AXES_FROM_CATEGORY",
    # parser
    "parse_frontmatter",
    # presets
    "EVENT_PRESET",
    "PROFILE_PRESET",
    "CONCEPT_PRESET",
    "THESIS_PRESET",
    "MODEL_PRESET",
    "QUESTIONS_PRESET",
    "METHOD_PRESET",
    "TOOL_PRESET",
    "FUNDAMENTALS_PRESET",
    "MATERIAL_PRESET",
    "PRESETS_BY_CATEGORY",
    "preset_for_category",
]
