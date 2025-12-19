"""Delete memory operation for removing memories from the vector store.

This module provides the DeleteMemoryOp class for deleting memories
by their IDs from the vector store.
"""

from typing import List

from ..base_memory_tool_op import BaseMemoryToolOp
from ... import C


@C.register_op()
class DeleteMemoryOp(BaseMemoryToolOp):
    """Operation for deleting memories from the vector store.

    This operation supports both single and multiple memory deletion modes,
    controlled by the `enable_multiple` parameter inherited from BaseMemoryToolOp.
    """

    def build_input_schema(self) -> dict:
        """Build input schema for single memory deletion mode.

        Returns:
            dict: Input schema for deleting a single memory by ID.
        """
        return {
            "memory_id": {
                "type": "string",
                "description": self.get_prompt("memory_id"),
                "required": True,
            }
        }

    def build_multiple_input_schema(self) -> dict:
        """Build input schema for multiple memory deletion mode.

        Returns:
            dict: Input schema for deleting multiple memories by IDs.
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
        """Execute the delete memory operation.

        Deletes one or more memories from the vector store based on their IDs.
        The operation handles both single ID (string) and multiple IDs (list) inputs.

        Raises:
            ValueError: If no valid memory IDs are provided.
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
            self.set_output("No valid memory IDs provided for deletion.")
            return

        # Perform deletion
        await self.vector_store.async_delete(node_ids=memory_ids, workspace_id=workspace_id)

        # Format output message
        if len(memory_ids) == 1:
            output_msg = f"Successfully deleted memory (id={memory_ids[0]}) from workspace={workspace_id}."
        else:
            output_msg = f"Successfully deleted {len(memory_ids)} memories from workspace={workspace_id}."

        self.set_output(output_msg)
