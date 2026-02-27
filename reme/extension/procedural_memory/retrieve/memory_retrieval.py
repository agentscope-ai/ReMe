"""Operation for recalling memories from the vector store based on a query."""

from typing import List

from loguru import logger

from ....core.op import BaseOp
from ....core.schema.memory_node import MemoryNode
from ....core.schema.vector_node import VectorNode


class MemoryRetrieval(BaseOp):
    """Operation that retrieves relevant memories from the vector store.

    This operation performs a semantic search on the vector store to find
    memories relevant to a given query. It supports optional score filtering
    and deduplication based on memory content.
    """

    async def execute(self):
        """Execute the memory recall operation.

        Performs a semantic search in the vector store using the provided query,
        retrieves the top-k most relevant memories, and optionally filters them
        by a score threshold. Duplicate memories (based on content) are removed.

        Expected context attributes:
            query: The search query string.
            top_k: Number of top results to retrieve (default: 3).

        Expected context attributes (optional):
            threshold_score: Optional minimum score threshold for filtering.

        Sets response.metadata:
            memory_list: List of retrieved MemoryNode objects.
        """
        top_k: int = self.context.get("top_k", 5)

        query: str = self.context.get("query", "")
        assert query, "query should be not empty!"

        # Perform semantic search
        nodes: List[VectorNode] = await self.vector_store.search(
            query=query,
            limit=top_k,
            filters=None,
        )

        # Convert VectorNodes to MemoryNodes and deduplicate by content
        memory_list: List[MemoryNode] = []
        memory_content_set: set[str] = set()  # for deduplication
        for node in nodes:
            try:
                memory = MemoryNode.from_vector_node(node)
                if memory.content not in memory_content_set:
                    memory_list.append(memory)
                    memory_content_set.add(memory.content)
            except Exception as e:
                logger.warning(f"Failed to convert VectorNode to MemoryNode: {e}")
                continue
        logger.info(f"Retrieved memory.size={len(memory_list)}")

        threshold_score: float | None = self.context.get("threshold_score", None)
        if threshold_score is not None:
            memory_list = [mem for mem in memory_list if mem.score >= threshold_score]
            logger.info(f"After threshold filter: {len(memory_list)} memories retained")

        self.context.response.metadata["memory_list"] = memory_list
