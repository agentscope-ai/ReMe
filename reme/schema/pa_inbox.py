"""Read-only desktop projections; user reading and agent consumption stay separate."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

InboxKind = Literal["task_result", "instruction"]
InboxStatus = Literal["pending", "received", "superseded", "removed", "unknown"]


class PaInboxItem(BaseModel):
    """One input addressed to Proactive, without its full document body."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: InboxKind
    title: str
    preview: str
    status: InboxStatus
    created_at: str | None = None
    task_id: str | None = None
    path: str | None = None
    user_read_at: str | None = None
    source_modified: bool = False
    source_present: bool | None = True
    receipt_status: str | None = None


class PaInboxPage(BaseModel):
    """Global counts remain independent of the current page and filters."""

    items: list[PaInboxItem]
    total: int
    offset: int
    limit: int
    task_ids: list[str]
    pending: int
    complete: bool = True
    warnings: list[str] = Field(default_factory=list)


class PaInboxDetail(BaseModel):
    """Actual source text together with its original request receipt."""

    item: PaInboxItem
    content: str
    received_at: str | None = None
    warnings: list[str] = Field(default_factory=list)
