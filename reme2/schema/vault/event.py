"""Event schema — task process records under events/{date}/{name}/."""

from typing import Literal

from .frontmatter import VaultBaseFrontmatter

EventStatus = Literal["active", "distilled", "archived"]


class Event(VaultBaseFrontmatter):
    """events/{YYYY-MM-DD}/{name}/{name}.md."""

    category: Literal["event"] = "event"  # type: ignore[assignment]
    status: EventStatus = "active"
    topics: list[str] = []
    originSessionId: str | None = None
