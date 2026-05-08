"""RegexEdgeExtractor — fast route per `structure.md` §"双态抽取".

Pulls edges from inline syntax + frontmatter walk in O(text length).
Synchronous and dependency-free; runs inside the watcher's hot path
without blocking on any external service.

Sources of edges:
    - body inline syntax:  `[[X]]`, `![[X#Section|Alias]]`,
                           `[predicate:: [[X]]]`, `(predicate:: [[X]])`
    - frontmatter walk:    `author: "[[John]]"` → predicate=author,
                           `related: ["[[X]]", "[[Y]]"]` → two edges

Dedup key: (target, predicate, anchor, alias, embed). When the same
edge appears in both body and frontmatter, the inline (regex) variant
wins — it's the more direct authoring intent.
"""

from __future__ import annotations

from .base_edge_extractor import BaseEdgeExtractor
from ..component_registry import R
from ...schema import FileEdge
from ...utils.wikilink import parse_wikilinks, parse_wikilinks_from_metadata


@R.register("regex")
class RegexEdgeExtractor(BaseEdgeExtractor):
    """Inline-syntax + frontmatter typed-edge extractor."""

    async def extract(
        self,
        text: str,
        metadata: dict | None = None,
        path: str | None = None,
    ) -> list[FileEdge]:
        body_edges = parse_wikilinks(text or "")
        meta_edges = parse_wikilinks_from_metadata(metadata or {})

        seen: set[tuple] = set()
        out: list[FileEdge] = []
        for edge in (*body_edges, *meta_edges):
            key = (edge.target, edge.predicate, edge.anchor, edge.alias, edge.embed)
            if key in seen:
                continue
            seen.add(key)
            out.append(edge)
        return out
