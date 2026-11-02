"""Auto-dream schemas."""

from typing import Literal

from pydantic import BaseModel, Field

from ..enumeration import DreamBucketEnum

# ProactiveResult moved to reme.schema.proactive; re-exported for import compatibility.
from .proactive import ProactiveResult  # noqa: F401  # pylint: disable=unused-import


class DreamUnit(BaseModel):
    """One cross-file memory unit emitted by global extract."""

    name: str = Field(description="Short kebab-case handle for the abstraction.")
    bucket: DreamBucketEnum = Field(description="Digest bucket; unknown raw values route to wiki before validation.")
    summary: str = Field(description="Grounded abstraction summary with evidence pointers.")
    paths: list[str] = Field(default_factory=list, description="Workspace-relative source paths.")


class DreamExtractOutput(BaseModel):
    """Structured output for ``dream_extract_step``."""

    units: list[DreamUnit] = Field(default_factory=list)


class IntegrateOutcome(BaseModel):
    """Structured output for one unit integration."""

    action: Literal["CREATE", "CORROBORATE", "REFINE", "CORRECT"] = Field(description="Write decision.")
    target_path: str = Field(description="Digest path written or edited.")
    note: str = Field(default="", description="Short summary of what landed.")


class DreamState(BaseModel):
    """Shared state passed across the dream steps."""

    date: str = ""
    dates: list[str] = Field(default_factory=list)
    scan_days: int = 2
    hint: str = ""
    daily_dir: str = ""
    workspace: str = ""
    files_scanned: int = 0
    files_unchanged: int = 0
    files_changed: int = 0
    files_deleted: int = 0
    changed_paths: list[str] = Field(default_factory=list)
    unchanged_paths: list[str] = Field(default_factory=list)
    deleted_paths: list[str] = Field(default_factory=list)
    existing: dict[str, float] = Field(default_factory=dict)
    indexed: dict[str, float] = Field(default_factory=dict)
    units: list[dict] = Field(default_factory=list)
    extract_summary: str = ""
    integrate_results: list[dict] = Field(default_factory=list)
    skipped_units: list[dict] = Field(default_factory=list)
    nodes_created: list[str] = Field(default_factory=list)
    nodes_updated: list[str] = Field(default_factory=list)
    modified_paths: list[str] = Field(
        default_factory=list,
        description="Durable digest files detected as created or changed during this run.",
    )
    failed_units: list[dict] = Field(default_factory=list)
    failed_paths: list[str] = Field(default_factory=list)
    processed_paths: list[str] | None = Field(
        default=None,
        description=(
            "Paths actually sent to the model. ``None`` means the run did not "
            "record it (state written by an older version), and the finish step "
            "then falls back to checkpointing every changed path."
        ),
    )
    checkpoint_paths: list[str] = Field(default_factory=list)
    deferred_paths: list[str] = Field(
        default_factory=list,
        description="Changed paths never sent to the model, left eligible for a later run.",
    )
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    summary: str = ""
