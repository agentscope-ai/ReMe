"""Topic schema — every .md under topics/ is a Topic.

No satellite/tentacle/folder note subclassing. Folder topic = Topic whose
filename equals its parent directory name (path convention; not a schema field).
"""

from typing import Literal

from pydantic import model_validator

from .frontmatter import VaultBaseFrontmatter

# Topic categories grouped by usage:
#   - Index categories: usually appear as folder topics; may have ticker/market
#   - Content categories: cluster siblings; thesis/model/questions need confidence
INDEX_CATEGORIES = {"company", "sector", "concept", "method", "tool", "profile"}
JUDGMENT_CATEGORIES = {"thesis", "model", "questions"}
CONTENT_CATEGORIES = JUDGMENT_CATEGORIES | {"fundamentals"}

TopicCategory = Literal[
    "company", "sector", "concept", "method", "tool", "profile",
    "thesis", "model", "questions", "fundamentals",
]
Confidence = Literal["⏳", "✅", "❌"]


class Topic(VaultBaseFrontmatter):
    """topics/{folder}/{name}.md — long-lived cognitive memory node.

    Folder topic is identified by path convention: filename stem == parent
    folder name. Not represented in this schema directly; checked at runtime
    via `Path(p).stem == Path(p).parent.name`.
    """

    category: TopicCategory  # type: ignore[assignment]
    market: str | None = None
    ticker: str | None = None
    confidence: Confidence | None = None

    @model_validator(mode="after")
    def _confidence_required_for_judgments(self):
        if self.category in JUDGMENT_CATEGORIES and self.confidence is None:
            raise ValueError(
                f"category={self.category} requires explicit confidence "
                f"(one of ⏳ / ✅ / ❌)"
            )
        return self
