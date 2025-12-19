"""Update memory operation for updating memories in the vector store.

This module provides the UpdateMemoryOp class for updating memories
by deleting the old memory and inserting a new one with updated content.
"""

from typing import Any, Dict, List

from .base_memory_tool_op import BaseMemoryToolOp
from .. import C
from ..schema import MemoryNode


@C.register_op()
class UpdateMemoryOp(BaseMemoryToolOp):
    """Operation for updating memories in the vector store.

    This operation supports both single and multiple memory update modes,
    controlled by the `enable_multiple` parameter inherited from BaseMemoryToolOp.
    Update is performed by deleting the old memory and inserting a new one.
    """

    def build_input_schema(self) -> dict:
        """Build input schema for single memory update mode.

        Returns:
            dict: Input schema for updating a single memory.
        """
        return {
            "memory_id": {
                "type": "string",
                "description": self.get_prompt("memory_id"),
                "required": True,
            },
            "memory_content": {
                "type": "string",
                "description": self.get_prompt("memory_content"),
                "required": True,
            },
            "metadata": {
                "type": "object",
                "description": self.get_prompt("metadata"),
                "required": False,
            }
        }

    def build_multiple_input_schema(self) -> dict:
        """Build input schema for multiple memory update mode.

        Returns:
            dict: Input schema for updating multiple memories.
        """
        return {
            "memories": {
                "type": "array",
                "description": self.get_prompt("memories"),
                "required": True,
                "items": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": self.get_prompt("memory_id"),
                        },
                        "memory_content": {
                            "type": "string",
                            "description": self.get_prompt("memory_content"),
                        },
                        "metadata": {
                            "type": "object",
                            "description": self.get_prompt("metadata"),
                        }
                    },
                    "required": ["memory_id", "memory_content"]
                },
            }
        }

    def _build_memory_node(
            self,
            memory_content: str,
            metadata: Dict[str, Any],
            workspace_id: str,
    ) -> MemoryNode:
        """Build a MemoryNode from memory_content and metadata.

        Args:
            memory_content: The memory content.
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
            content=memory_content,
            ref_memory_id=self.ref_memory_id,
            author=self.author,
            metadata=metadata,
        )

    async def async_execute(self):
        """Execute the update memory operation.

        Updates one or more memories in the vector store. The operation handles both
        single memory and multiple memories inputs. For each update, it first deletes
        the old memory by memory_id, then inserts the new memory with updated content.

        Raises:
            ValueError: If no valid memories are provided.
        """
        workspace_id: str = self.workspace_id

        # Collect old memory_ids to delete and new memory nodes to insert
        old_memory_ids: List[str] = []
        new_memory_nodes: List[MemoryNode] = []

        if self.enable_multiple:
            memories: List[Dict[str, Any]] = self.input_dict.get("memories", [])
            for mem in memories:
                memory_id = mem.get("memory_id", "")
                memory_content = mem.get("memory_content", "")
                if not memory_id or not memory_content:
                    continue
                metadata = mem.get("metadata", {}) or {}
                old_memory_ids.append(memory_id)
                new_memory_nodes.append(self._build_memory_node(memory_content, metadata, workspace_id))
        else:
            memory_id = self.input_dict.get("memory_id", "")
            memory_content = self.input_dict.get("memory_content", "")
            if memory_id and memory_content:
                metadata = self.input_dict.get("metadata", {}) or {}
                old_memory_ids.append(memory_id)
                new_memory_nodes.append(self._build_memory_node(memory_content, metadata, workspace_id))

        if not old_memory_ids or not new_memory_nodes:
            self.set_output("No valid memories provided for update.")
            return

        # Delete old memories and any existing memories with the same new IDs (upsert behavior)
        new_memory_ids: List[str] = [node.memory_id for node in new_memory_nodes]
        all_ids_to_delete: List[str] = list(set(old_memory_ids + new_memory_ids))
        await self.vector_store.async_delete(node_ids=all_ids_to_delete, workspace_id=workspace_id)

        # Convert to VectorNodes and insert
        vector_nodes = [node.to_vector_node() for node in new_memory_nodes]
        await self.vector_store.async_insert(nodes=vector_nodes, workspace_id=workspace_id)

        # Format output message
        if len(new_memory_nodes) == 1:
            output_msg = (
                f"Successfully updated memory: deleted old (id={old_memory_ids[0]}), "
                f"added new (id={new_memory_nodes[0].memory_id}) in workspace={workspace_id}."
            )
        else:
            old_ids_str = ", ".join(old_memory_ids)
            new_ids_str = ", ".join([node.memory_id for node in new_memory_nodes])
            output_msg = (
                f"Successfully updated {len(new_memory_nodes)} memories: "
                f"deleted old (ids={old_ids_str}), added new (ids={new_ids_str}) in workspace={workspace_id}."
            )

        self.set_output(output_msg)

