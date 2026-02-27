"""Operation for adding memories to the vector store."""

from typing import List
from loguru import logger
from ....core.op import BaseOp
from ....core.schema.vector_node import VectorNode
from ....core.schema.memory_node import MemoryNode


class MemoryAddition(BaseOp):
    """Operation that adds new or updated memories to the vector store.

    This operation inserts memories into the vector store. It reads the list
    of memories to insert from response.metadata and performs the actual
    database insertion operations.
    """

    async def execute(self):
        """Execute the memory insertion operation.

        Inserts new or updated memories into the vector store:
        1. Reads memory_list from context (can be dicts or MemoryNode)
        2. Converts raw items to MemoryNode objects
        3. Converts MemoryNode objects to VectorNode objects
        4. Inserts them into the vector store
        """
        raw_memory_list = self.context.memory_list
        insert_memory_list: List[MemoryNode] = [MemoryNode(**x) if isinstance(x, dict) else x for x in raw_memory_list]
        if insert_memory_list:
            insert_nodes: List[VectorNode] = [x.to_vector_node() for x in insert_memory_list]
            await self.vector_store.insert(nodes=insert_nodes)
            logger.info(f"insert insert_node.size={len(insert_nodes)}")
