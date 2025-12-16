"""Add memory operation for inserting memories into the vector store.

This module provides the AddMemoryOp class for adding memories
to the vector store with support for both single and multiple memory modes.
"""

from typing import Any, Dict, List

from .base_memory_tool_op import BaseMemoryToolOp
from .. import C
from ..schema import MemoryNode


@C.register_op()
class AddMemoryOp(BaseMemoryToolOp):
    """Operation for adding memories to the vector store.

    This operation supports both single and multiple memory addition modes,
    controlled by the `enable_multiple` parameter inherited from BaseMemoryToolOp.
    """

    def build_input_schema(self) -> dict:
        """Build input schema for single memory addition mode.

        Returns:
            dict: Input schema for adding a single memory.
        """
        return {
            "query": {
                "type": "string",
                "description": self.get_prompt("query"),
                "required": True,
            },
            "metadata": {
                "type": "object",
                "description": self.get_prompt("metadata"),
                "required": False,
            }
        }

    def build_multiple_input_schema(self) -> dict:
        """Build input schema for multiple memory addition mode.

        Returns:
            dict: Input schema for adding multiple memories.
        """
        return {
            "memories": {
                "type": "array",
                "description": self.get_prompt("memories"),
                "required": True,
                "items": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": self.get_prompt("query"),
                        },
                        "metadata": {
                            "type": "object",
                            "description": self.get_prompt("metadata"),
                        }
                    },
                    "required": ["query"]
                },
            }
        }

    def _build_memory_node(
            self,
            query: str,
            metadata: Dict[str, Any],
            workspace_id: str,
    ) -> MemoryNode:
        """Build a MemoryNode from query and metadata.

        Args:
            query: The memory content.
            metadata: Additional metadata for the memory.
            workspace_id: The workspace ID.

        Returns:
            MemoryNode: The constructed memory node.
        """
        return MemoryNode(
            workspace_id=workspace_id,
            memory_type=self.memory_type,
            memory_target=self.memory_target,
            when_to_use="",
            content=query,
            ref_memory_id=self.ref_memory_id,
            author=self.author,
            metadata=metadata,
        )

    async def async_execute(self):
        """Execute the add memory operation.

        Adds one or more memories to the vector store. The operation handles both
        single memory (query + metadata) and multiple memories (list) inputs.
        For each memory, it first deletes any existing memory with the same ID,
        then inserts the new memory.

        Raises:
            ValueError: If no valid memories are provided.
        """
        workspace_id: str = self.workspace_id

        # Build memory nodes based on mode
        memory_nodes: List[MemoryNode] = []
        if self.enable_multiple:
            memories: List[Dict[str, Any]] = self.input_dict.get("memories", [])
            for mem in memories:
                query = mem.get("query", "")
                if not query:
                    continue
                metadata = mem.get("metadata", {}) or {}
                memory_nodes.append(self._build_memory_node(query, metadata, workspace_id))
        else:
            query = self.input_dict.get("query", "")
            if query:
                metadata = self.input_dict.get("metadata", {}) or {}
                memory_nodes.append(self._build_memory_node(query, metadata, workspace_id))

        if not memory_nodes:
            self.set_output("No valid memories provided for addition.")
            return

        # Delete existing memories with the same IDs (upsert behavior)
        memory_ids: List[str] = [node.memory_id for node in memory_nodes]
        await self.vector_store.async_delete(node_ids=memory_ids, workspace_id=workspace_id)

        # Convert to VectorNodes and insert
        vector_nodes = [node.to_vector_node() for node in memory_nodes]
        await self.vector_store.async_insert(nodes=vector_nodes, workspace_id=workspace_id)

        # Format output message
        if len(memory_nodes) == 1:
            output_msg = f"Successfully added memory (id={memory_nodes[0].memory_id}) to workspace={workspace_id}."
        else:
            ids_str = ", ".join([node.memory_id for node in memory_nodes])
            output_msg = f"Successfully added {len(memory_nodes)} memories (ids={ids_str}) to workspace={workspace_id}."

        self.set_output(output_msg)
