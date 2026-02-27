"""Operation for deleting memories from the vector store."""

import json
from typing import List

from loguru import logger

from ....core.op import BaseOp
from ....core.schema.vector_node import VectorNode


class MemoryDeletion(BaseOp):
    """Operation that deletes memories from the vector store.

    This operation identifies memories to delete based on frequency and utility
    thresholds, then deletes them. Memories with frequency >= freq_threshold
    and utility/frequency ratio < utility_threshold are deleted.
    """

    async def execute(self):
        """Execute the memory deletion operation.

        Identifies and deletes memories from the vector store:
        1. Lists all nodes from the vector store
        2. Identifies memories that meet deletion criteria based on thresholds
        3. Deletes identified memories from the vector store
        4. Stores deletion count in response.metadata["result"]

        The deletion criteria:
        - Memory frequency must be >= freq_threshold
        - Memory utility/frequency ratio must be < utility_threshold

        Expected context attributes:
            freq_threshold: Minimum frequency threshold for consideration.
            utility_threshold: Maximum utility/frequency ratio threshold.
        """

        # Step 1: Identify memories to delete based on thresholds
        freq_threshold: int = self.context.freq_threshold
        utility_threshold: float = self.context.utility_threshold
        nodes: List[VectorNode] = await self.vector_store.list()

        deleted_memory_ids = []
        for node in nodes:
            freq = node.metadata.get("freq", 0)
            utility = node.metadata.get("utility", 0)
            if freq >= freq_threshold:
                if freq > 0 and utility * 1.0 / freq < utility_threshold:
                    deleted_memory_ids.append(node.vector_id)

        # Step 2: Execute deletion if there are any IDs to delete
        if deleted_memory_ids:
            await self.vector_store.delete(vector_ids=deleted_memory_ids)
            logger.info(f"Deleted {len(deleted_memory_ids)} memories: {json.dumps(deleted_memory_ids, indent=2)}")
