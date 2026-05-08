"""Common frontmatter schema shared by all vault files."""

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class VaultBaseFrontmatter(BaseModel):
    """Fields present on every vault file."""

    model_config = ConfigDict(extra="allow")

    title: str = Field(...)
    description: str = Field(default="")
    tags: list[str] = Field(default_factory=list)
    category: str = Field(...)
    created: date | None = None
    updated: date | None = None

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


def parse_frontmatter(raw: dict[str, Any]) -> VaultBaseFrontmatter:
    """Tolerant parse — never raises; missing required fields fall back to defaults."""
    safe = dict(raw or {})
    safe.setdefault("title", safe.get("name") or "")
    safe.setdefault("category", "unknown")
    return VaultBaseFrontmatter.model_validate(safe)
