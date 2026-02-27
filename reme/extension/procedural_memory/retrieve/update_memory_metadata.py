"""Operation for updating memory metadata (frequency and utility).

This module provides a unified operation to update frequency counters and
optionally utility scores for recalled memories, directly updating the
vector store.
"""

from typing import List

from loguru import logger

from ....core.op import BaseOp
from ....core.schema.memory_node import MemoryNode
from ....core.schema.vector_node import VectorNode


class UpdateMemoryMetadata(BaseOp):
    """Update memory metadata: frequency and optionally utility.

    This operation (1) increments each memory's frequency counter;
    (2) optionally increments utility when update_utility is True;
    (3) directly updates the VectorNode in the vector store using the update method.

    Expected context attributes:
        memory_list: List of MemoryNode objects to update (already loaded from
            previous operations like rerank_memory).
        update_utility: Boolean flag. If True, also increment utility for each memory.
    """

    async def execute(self):
        """Run frequency update, optional utility update, and directly update vector store."""
        memory_list: List[MemoryNode] = [MemoryNode(**node) for node in self.context.memory_list]
        update_utility = self.context.update_utility

        if not memory_list:
            logger.info("No memories to update metadata")
            return

        updated_nodes: List[VectorNode] = []
        for memory in memory_list:
            meta = memory.metadata
            meta["freq"] = meta.get("freq", 0) + 1
            if update_utility:
                meta["utility"] = meta.get("utility", 0) + 1
            memory.metadata = meta
            vector_node = memory.to_vector_node()
            updated_nodes.append(vector_node)

        if updated_nodes:
            await self.vector_store.update(nodes=updated_nodes)
            logger.info(f"Updated metadata for {len(updated_nodes)} memories in vector store")
