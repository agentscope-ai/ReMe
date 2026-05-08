"""Vault business schemas — domain models for the markdown vault.

These are the *business objects* the user stores in the vault (Topic,
Event), distinct from the engine schemas in `reme2/schema/` (FileMetadata,
FileChunk, ChunkFilter — internal data types).

Single Topic class covers all categories under topics/ (no satellite /
tentacle subclassing); folder-topic is identified by path convention.

Lives under `reme2/schema/vault/` (not in `reme2/mcp/`) so memory
services (Maintainer.lint, Ingestor) can validate frontmatter without
importing the MCP transport layer — that's what was creating the
mcp ↔ memory dependency cycle.
"""

from .event import Event, EventStatus
from .frontmatter import VaultBaseFrontmatter, parse_frontmatter
from .registry import ALL_KNOWN_CATEGORIES, schema_for
from .topic import (
    INDEX_CATEGORIES,
    JUDGMENT_CATEGORIES,
    CONTENT_CATEGORIES,
    Confidence,
    Topic,
    TopicCategory,
)

__all__ = [
    "ALL_KNOWN_CATEGORIES",
    "CONTENT_CATEGORIES",
    "Confidence",
    "Event",
    "EventStatus",
    "INDEX_CATEGORIES",
    "JUDGMENT_CATEGORIES",
    "Topic",
    "TopicCategory",
    "VaultBaseFrontmatter",
    "parse_frontmatter",
    "schema_for",
]
