"""Action-item schemas: unresolved work tracked across workspace notes.

The contract is deliberately read-mostly. Extraction and state updates happen
in the memory workflow; consumers (for example a reminder in the host app) only
query the records and apply their own timing, deduplication and delivery.
"""

import hashlib
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ActionItemState(str, Enum):
    """Evidence-backed state of one action item.

    A note being processed by the memory pipeline says nothing about whether
    the work it describes is still open, so every state must be supported by
    evidence. Unknown material maps to ``uncertain`` and never to
    ``unresolved``: a missing field must not become a reminder.
    """

    UNRESOLVED = "unresolved"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    UNCERTAIN = "uncertain"
    CANCELLED = "cancelled"


class ActionItemEvidence(BaseModel):
    """One observation that supports the current state of an item."""

    path: str = Field(description="Workspace-relative path of the note that carries the observation.")
    locator: str = Field(
        default="",
        description="Where in the note the evidence sits, e.g. a heading, a bullet prefix or a line number.",
    )
    excerpt: str = Field(default="", description="Short verbatim excerpt; empty when the locator is enough.")
    observed_at: datetime = Field(description="When the evidence was observed in the source material.")
    source: str = Field(default="", description="Optional provenance label, e.g. the workflow that reported it.")

    @field_validator("path")
    @classmethod
    def _relative_path(cls, value: str) -> str:
        if not value:
            raise ValueError("evidence path must not be empty")
        if value.startswith("/") or "\\" in value or ".." in value.split("/"):
            raise ValueError("evidence path must be a normalized workspace-relative POSIX path")
        return value


class ActionItem(BaseModel):
    """One piece of unfinished work with its current, evidence-backed state."""

    id: str = Field(description="Stable identifier derived from the note and the item text.")
    note: str = Field(description="Workspace-relative path of the note this item was extracted from.")
    description: str = Field(description="Short human-readable description of the work.")
    state: ActionItemState = Field(
        default=ActionItemState.UNCERTAIN,
        description="Current state; unknown material must stay uncertain rather than unresolved.",
    )
    evidence: list[ActionItemEvidence] = Field(
        default_factory=list,
        description="Observations that support the current state, newest last.",
    )
    first_seen: datetime = Field(description="When this item was first recorded.")
    last_updated: datetime = Field(description="When the state or description last changed.")

    @field_validator("note")
    @classmethod
    def _relative_note(cls, value: str) -> str:
        if not value:
            raise ValueError("note must not be empty")
        if value.startswith("/") or "\\" in value or ".." in value.split("/"):
            raise ValueError("note must be a normalized workspace-relative POSIX path")
        return value

    @field_validator("description")
    @classmethod
    def _non_empty_description(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("description must not be empty")
        return text

    @model_validator(mode="after")
    def _state_needs_evidence(self) -> "ActionItem":
        if self.state is not ActionItemState.UNCERTAIN and not self.evidence:
            raise ValueError(f"state '{self.state.value}' requires at least one evidence entry")
        return self

    def sort_key(self) -> tuple[datetime, str]:
        """Order items deterministically, newest update first."""
        return (self.last_updated, self.id)


class ActionItemQuery(BaseModel):
    """Read-only filter for action items."""

    states: list[ActionItemState] | None = Field(
        default=None,
        description="Only return items in these states; None means every state.",
    )
    updated_since: datetime | None = Field(
        default=None,
        description="Only return items updated at or after this moment; None means no lower bound.",
    )
    limit: int = Field(default=100, ge=1, description="Maximum number of items to return.")

    @field_validator("states")
    @classmethod
    def _known_states(cls, value: list[ActionItemState] | None) -> list[ActionItemState] | None:
        if value is None:
            return None
        return list(dict.fromkeys(value))

    def matches(self, item: ActionItem) -> bool:
        """Return whether ``item`` satisfies this filter."""
        if self.states is not None and item.state not in self.states:
            return False
        if self.updated_since is not None and item.last_updated < self.updated_since:
            return False
        return True


def action_item_id(note: str, description: str) -> str:
    """Derive a stable id so later evidence can refer to the same item.

    The id depends only on the note path and a normalized form of the
    description, both of which survive restarts and index rebuilds.
    """
    normalized = " ".join(description.split()).casefold()
    digest = hashlib.sha1(f"{note}\x00{normalized}".encode("utf-8")).hexdigest()
    return f"ai_{digest[:16]}"


def merge_evidence(
    existing: list[ActionItemEvidence],
    incoming: list[ActionItemEvidence],
) -> list[ActionItemEvidence]:
    """Union evidence lists, dropping exact duplicates and keeping time order."""
    seen = {(e.path, e.locator, e.observed_at, e.excerpt) for e in existing}
    merged = list(existing)
    for evidence in incoming:
        key = (evidence.path, evidence.locator, evidence.observed_at, evidence.excerpt)
        if key in seen:
            continue
        seen.add(key)
        merged.append(evidence)
    merged.sort(key=lambda e: e.observed_at)
    return merged


def apply_update(existing: ActionItem, incoming: ActionItem) -> ActionItem:
    """Merge a later observation into an existing item.

    Later evidence updates the state; evidence processed late but observed
    earlier must not overwrite a newer resolution. In that case the state and
    description stay as they are and only the evidence list grows.
    """
    evidence = merge_evidence(existing.evidence, incoming.evidence)
    if incoming.last_updated < existing.last_updated:
        return existing.model_copy(update={"evidence": evidence})
    return existing.model_copy(
        update={
            "description": incoming.description,
            "state": incoming.state,
            "evidence": evidence,
            "last_updated": incoming.last_updated,
        },
    )


def as_record(item: ActionItem) -> dict[str, Any]:
    """Serialize an item for file storage."""
    return item.model_dump(mode="json")
