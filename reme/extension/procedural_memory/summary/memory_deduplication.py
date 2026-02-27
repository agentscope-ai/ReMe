"""Memory deduplication operation for task memory management.

This module provides operations to remove duplicate or highly similar task
memories by comparing embeddings and calculating similarity scores.
"""

from typing import List

from loguru import logger

from ....core.op import BaseOp
from ....core.schema.memory_node import MemoryNode


class MemoryDeduplication(BaseOp):
    """Remove duplicate task memories using embedding similarity.

    This operation identifies and removes duplicate or highly similar task
    memories by comparing their embeddings against both existing memories
    in the vector store and other memories in the current batch.
    """

    async def execute(self):
        """Remove duplicate task memories"""
        # Get task memories to deduplicate
        task_memories: List[MemoryNode] = self.context.response.metadata.get("memory_list", [])

        if not task_memories:
            logger.info("No task memories found for deduplication")
            return

        logger.info(f"Starting deduplication for {len(task_memories)} task memories")

        # Perform deduplication
        deduplicated_task_memories = await self._deduplicate_task_memories(task_memories)

        logger.info(
            f"Deduplication complete: {len(deduplicated_task_memories)} deduplicated "
            f"task memories out of {len(task_memories)}",
        )

        # Update context
        self.context.response.metadata["memory_list"] = deduplicated_task_memories

    async def _deduplicate_task_memories(self, task_memories: List[MemoryNode]) -> List[MemoryNode]:
        """Remove duplicate task memories"""
        if not task_memories:
            return task_memories

        similarity_threshold = self.context.get("similarity_threshold", 0.5)

        unique_task_memories = []

        # Get existing task memory embeddings
        existing_embeddings = await self._get_existing_task_memory_embeddings()

        for task_memory in task_memories:
            # Generate embedding for current task memory
            current_embedding = await self._get_task_memory_embedding(task_memory)

            if current_embedding is None:
                logger.warning(f"Failed to generate embedding for task memory: {str(task_memory.when_to_use)[:50]}...")
                continue

            # Check similarity with existing task memories
            if self._is_similar_to_existing_task_memories(current_embedding, existing_embeddings, similarity_threshold):
                logger.debug(f"Skipping similar task memory: {str(task_memory.when_to_use)[:50]}...")
                continue

            # Check similarity with current batch task memories
            if await self._is_similar_to_current_task_memories(
                current_embedding,
                unique_task_memories,
                similarity_threshold,
            ):
                logger.debug(f"Skipping duplicate in current batch: {str(task_memory.when_to_use)[:50]}...")
                continue

            # Add to unique task memories list
            unique_task_memories.append(task_memory)
            logger.debug(f"Added unique task memory: {str(task_memory.when_to_use)[:50]}...")

        return unique_task_memories

    async def _get_existing_task_memory_embeddings(self) -> List[List[float]]:
        """Get embeddings of existing task memories"""
        try:
            if not hasattr(self, "vector_store") or not self.vector_store:
                return []

            # List all existing task memory nodes
            existing_nodes = await self.vector_store.list(
                filters=None,  # No filters to get all nodes
                limit=self.context.get("max_existing_task_memories", 1000),
            )

            # Extract embeddings
            existing_embeddings = []
            for node in existing_nodes:
                if node.vector:
                    existing_embeddings.append(node.vector)

            logger.debug(
                f"Retrieved {len(existing_embeddings)} existing task memory embeddings",
            )
            return existing_embeddings

        except Exception as e:
            logger.warning(f"Failed to retrieve existing task memory embeddings: {e}")
            return []

    async def _get_task_memory_embedding(self, task_memory: MemoryNode) -> List[float] | None:
        """Generate embedding for task memory"""
        try:

            # Combine task memory description and content for embedding
            text_for_embedding = f"{task_memory.when_to_use} {task_memory.content}"
            embeddings = await self.vector_store.get_embeddings([text_for_embedding])

            if embeddings and len(embeddings) > 0:
                return embeddings[0]
            else:
                logger.warning("Empty embedding generated for task memory")
                return None

        except Exception as e:
            logger.error(f"Error generating embedding for task memory: {e}")
            return None

    def _is_similar_to_existing_task_memories(
        self,
        current_embedding: List[float],
        existing_embeddings: List[List[float]],
        threshold: float,
    ) -> bool:
        """Check if current embedding is similar to existing embeddings"""
        for existing_embedding in existing_embeddings:
            similarity = self._calculate_cosine_similarity(current_embedding, existing_embedding)
            if similarity > threshold:
                logger.debug(f"Found similar existing task memory with similarity: {similarity:.3f}")
                return True
        return False

    async def _is_similar_to_current_task_memories(
        self,
        current_embedding: List[float],
        current_task_memories: List[MemoryNode],
        threshold: float,
    ) -> bool:
        """Check if current embedding is similar to other memories in current batch."""
        for existing_task_memory in current_task_memories:
            existing_embedding = await self._get_task_memory_embedding(existing_task_memory)
            if existing_embedding is None:
                continue

            similarity = self._calculate_cosine_similarity(current_embedding, existing_embedding)
            if similarity > threshold:
                logger.debug(f"Found similar task memory in current batch with similarity: {similarity:.3f}")
                return True
        return False

    @staticmethod
    def _calculate_cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity"""
        try:
            import numpy as np

            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
