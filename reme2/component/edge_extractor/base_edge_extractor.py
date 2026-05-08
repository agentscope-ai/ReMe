"""Abstract base class for typed-edge extractors.

A `BaseEdgeExtractor` ingests a markdown file's body text + frontmatter
dict and returns a flat list of `FileEdge` objects. The concrete
implementations realize the two routes described in `structure.md`
§"图谱关系的双态抽取":

    Fast route (regex)   — RegexEdgeExtractor
    Slow route (LLM IE)  — LLMEdgeExtractor

Edges are returned with the `source` field already populated so
downstream consumers (file_store indexing, retrievers, maintainer) can
reason about provenance.

Resolution to absolute paths and dedup against already-indexed edges
are NOT this layer's responsibility — `target` stays raw, mirroring
how it appears in source.
"""

from __future__ import annotations

from abc import abstractmethod

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ...schema import FileEdge


class BaseEdgeExtractor(BaseComponent):
    """Pluggable typed-edge extractor (`ComponentEnum.EDGE_EXTRACTOR`)."""

    component_type = ComponentEnum.EDGE_EXTRACTOR

    @abstractmethod
    async def extract(
        self,
        text: str,
        metadata: dict | None = None,
        path: str | None = None,
    ) -> list[FileEdge]:
        """Return the edges discovered in `text` and `metadata`.

        Args:
            text: The body text of the source file (frontmatter stripped).
            metadata: Parsed YAML frontmatter dict, or None.
            path: Optional absolute path of the source file. Slow-route
                extractors use this for logging / source-text caching;
                fast-route ignores it.
        """
