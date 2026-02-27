"""Memory search tool for semantic search in memory files."""

import json

from loguru import logger

from ....core.enumeration import MemorySource
from ....core.op import BaseTool
from ....core.runtime_context import RuntimeContext
from ....core.schema import ToolCall


class MemorySearch(BaseTool):
    """Semantically search MEMORY.md and memory files."""

    def __init__(
        self,
        sources: list[MemorySource] | None = None,
        min_score: float = 0.1,
        max_results: int = 5,
        vector_weight: float = 0.7,
        candidate_multiplier: float = 3.0,
        **kwargs,
    ):
        """Initialize memory search tool."""
        assert 0.0 <= vector_weight <= 1.0, f"vector_weight must be between 0 and 1, got {vector_weight}"
        kwargs.setdefault("max_retries", 1)
        kwargs.setdefault("raise_exception", False)
        super().__init__(**kwargs)
        self.sources = sources or [MemorySource.MEMORY]
        self.min_score = min_score
        self.max_results = max_results
        self.vector_weight = vector_weight
        self.candidate_multiplier = candidate_multiplier

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": (
                    "Mandatory recall step: semantically search MEMORY.md + memory/*.md "
                    "(and optional session transcripts) before answering questions about "
                    "prior work, decisions, dates, people, preferences, or todos; returns "
                    "top snippets with path + lines."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The semantic search query to find relevant memory snippets",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of search results to return (optional), default 5",
                        },
                        "min_score": {
                            "type": "number",
                            "description": "Minimum similarity score threshold for results (optional), default 0.1",
                        },
                    },
                    "required": ["query"],
                },
            },
        )

    async def execute(self) -> str:
        """Execute the memory search operation."""
        query: str = self.context.query.strip()
        min_score: float = self.context.get("min_score", self.min_score)
        max_results: int = self.context.get("max_results", self.max_results)

        assert query, "Query cannot be empty"
        assert (
            isinstance(min_score, float) and 0.0 <= min_score <= 1.0
        ), f"min_score must be between 0 and 1, got {min_score}"
        assert (
            isinstance(max_results, int) and max_results > 0
        ), f"max_results must be a positive integer, got {max_results}"

        # Use hybrid_search from file_store
        results = await self.file_store.hybrid_search(
            query=query,
            limit=max_results,
            sources=self.sources,
            vector_weight=self.vector_weight,
            candidate_multiplier=self.candidate_multiplier,
        )

        # Filter by min_score
        results = [r for r in results if r.score >= min_score]

        return json.dumps([result.model_dump(exclude_none=True) for result in results], indent=2, ensure_ascii=False)

    async def call(self, context: RuntimeContext = None, **kwargs):
        """Execute the tool with unified error handling.

        This method catches all exceptions and returns error messages
        to the LLM instead of raising them.
        """
        self.context = RuntimeContext.from_context(context, **kwargs)

        try:
            await self.before_execute()
            response = await self.execute()
            response = await self.after_execute(response)
            return response

        except Exception as e:
            # Return error message to LLM instead of raising
            error_msg = f"{self.__class__.__name__} failed: {str(e)}"
            logger.error(error_msg)
            return await self.after_execute(error_msg)
