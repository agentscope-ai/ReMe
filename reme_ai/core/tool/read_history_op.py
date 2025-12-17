"""Read history operation for retrieving history messages by their IDs from the vector store.

This module provides the ReadHistoryOp class for reading history messages
by their unique IDs from the vector store.
"""

from typing import List

from flowllm.core.schema import VectorNode

from .base_memory_tool_op import BaseMemoryToolOp
from .. import C
from ..enumeration import MemoryType
from ..schema import MemoryNode


@C.register_op()
class ReadHistoryOp(BaseMemoryToolOp):
    """Operation for reading history messages from the vector store by their IDs.

    This operation supports both single and multiple ID modes,
    controlled by the `enable_multiple` parameter inherited from BaseMemoryToolOp.
    """

    def build_input_schema(self) -> dict:
        """Build input schema for single history message reading mode.

        Returns:
            dict: Input schema for reading a single history message by ID.
        """
        return {
            "memory_id": {
                "type": "string",
                "description": self.get_prompt("memory_id"),
                "required": True,
            }
        }

    def build_multiple_input_schema(self) -> dict:
        """Build input schema for multiple history messages reading mode.

        Returns:
            dict: Input schema for reading multiple history messages by IDs.
        """
        return {
            "memory_ids": {
                "type": "array",
                "description": self.get_prompt("memory_ids"),
                "required": True,
                "items": {"type": "string"},
            }
        }

    async def async_execute(self):
        """Execute the read history operation.

        Reads one or more history messages from the vector store based on their IDs.
        The operation handles both single ID (string) and multiple IDs (list) inputs.

        Returns the memory_value for each found history message.

        Raises:
            ValueError: If no valid history IDs are provided.
        """
        workspace_id: str = self.workspace_id

        # Get memory_ids based on mode
        if self.enable_multiple:
            memory_ids: List[str] = self.input_dict.get("memory_ids", [])
        else:
            memory_ids: List[str] = [self.input_dict.get("memory_id", "")]

        # Filter out empty strings
        memory_ids = [memory_id for memory_id in memory_ids if memory_id]

        if not memory_ids:
            self.set_output("No valid history IDs provided for reading.")
            return

        # Perform search by IDs using filter
        nodes: List[VectorNode] = await self.vector_store.async_search(
            query="",
            workspace_id=workspace_id,
            top_k=len(memory_ids),
            filter_dict={"unique_id": memory_ids}
        )

        if not nodes:
            self.set_output(f"No history messages found with the provided IDs in workspace={workspace_id}.")
            return

        # Convert nodes to memories
        memories: List[MemoryNode] = [MemoryNode.from_vector_node(n) for n in nodes]

        # Format output with memory_value
        output_lines = []
        for memory in memories:
            assert memory.memory_type is MemoryType.HISTORY
            output_lines.append(memory.memory_value)

        self.set_output("\n".join(output_lines))
