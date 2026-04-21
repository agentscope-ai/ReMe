"""Memory search step for semantic search in memory files."""

import json

from ..component import R
from ..component.base_step import BaseStep
from ..enumeration import ComponentEnum
from ..schema import SearchFilter


@R.register("memory_search")
class MemorySearch(BaseStep):
    """Semantically search MEMORY.md and memory files."""

    component_type = ComponentEnum.STEP

    def __init__(self, vector_weight: float = 0.7, candidate_multiplier: float = 3.0, **kwargs):
        """Initialize memory search step.

        Args:
            vector_weight: Weight for vector search vs keyword search.
            candidate_multiplier: Multiplier for candidate count before filtering.
            **kwargs: Additional arguments passed to BaseStep.
        """
        super().__init__(**kwargs)
        self.vector_weight = vector_weight
        self.candidate_multiplier = candidate_multiplier

    async def execute(self):
        """Execute the memory search operation."""
        assert self.context is not None, "Context is not set"
        query: str = self.context.get("query", "").strip()
        min_score: float = self.context.get("min_score", 0.1)
        max_results: int = self.context.get("max_results", 5)

        assert query, "Query cannot be empty"
        assert (
            isinstance(min_score, float | int) and 0.0 <= min_score <= 1.0
        ), f"min_score must be between 0 and 1, got {min_score}"
        assert (
            isinstance(max_results, int) and max_results > 0
        ), f"max_results must be a positive integer, got {max_results}"

        filter_paths: list[str] | None = self.context.get("paths") or None
        filter_tags: list[str] | None = self.context.get("tags") or None
        exclude_paths: list[str] | None = self.context.get("exclude_paths") or None
        search_filter = None
        if filter_paths or filter_tags or exclude_paths:
            search_filter = SearchFilter(paths=filter_paths, tags=filter_tags, exclude_paths=exclude_paths)

        results = await self.file_store.hybrid_search(
            query=query,
            limit=max_results,
            vector_weight=self.vector_weight,
            candidate_multiplier=self.candidate_multiplier,
            search_filter=search_filter,
        )

        # Filter by min_score
        results = [r for r in results if r.score >= min_score]

        return json.dumps([result.model_dump(exclude_none=True) for result in results], indent=2, ensure_ascii=False)
