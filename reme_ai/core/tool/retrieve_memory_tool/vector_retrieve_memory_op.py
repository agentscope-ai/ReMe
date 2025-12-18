"""Vector retrieve memory operation for searching memories from the memory store.

This module provides the VectorRetrieveMemoryOp class for retrieving memories
by query texts using vector similarity search.
"""

from typing import List

from flowllm.core.schema import VectorNode

from ..base_memory_tool_op import BaseMemoryToolOp
from ... import C
from ...enumeration import MemoryType
from ...schema import MemoryNode


@C.register_op()
class VectorRetrieveMemoryOp(BaseMemoryToolOp):
    """Operation for retrieving memories from the memory store.

    This operation supports both single and multiple query modes,
    controlled by the `enable_multiple` parameter inherited from BaseMemoryToolOp.
    """

    def __init__(self, enable_summary_memory: bool = True, top_k: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.enable_summary_memory: bool = enable_summary_memory
        self.top_k: int = self.service_config_metadata.get("top_k", top_k)

    def build_input_schema(self) -> dict:
        """Build input schema for single query mode.

        Returns:
            dict: Input schema for retrieving memories with memory_type, memory_target, and query.
        """
        return {
            "memory_type": {
                "type": "string",
                "description": self.get_prompt("memory_type"),
                "enum": [mt.value for mt in MemoryType],
                "required": True,
            },
            "memory_target": {
                "type": "string",
                "description": self.get_prompt("memory_target"),
                "required": True,
            },
            "query": {
                "type": "string",
                "description": self.get_prompt("query"),
                "required": True,
            },
        }

    def build_multiple_input_schema(self) -> dict:
        """Build input schema for multiple query mode.

        Returns:
            dict: Input schema for retrieving memories with list of query items.
        """
        return {
            "query_items": {
                "type": "array",
                "description": self.get_prompt("query_items"),
                "required": True,
                "items": {
                    "type": "object",
                    "properties": {
                        "memory_type": {
                            "type": "string",
                            "description": self.get_prompt("memory_type"),
                            "enum": [mt.value for mt in MemoryType],
                        },
                        "memory_target": {
                            "type": "string",
                            "description": self.get_prompt("memory_target"),
                        },
                        "query": {
                            "type": "string",
                            "description": self.get_prompt("query"),
                        },
                    },
                    "required": ["memory_type", "memory_target", "query"],
                },
            }
        }

    async def retrieve_by_query(
        self, query: str, memory_type: str, memory_target: str, workspace_id: str
    ) -> List[MemoryNode]:
        """Retrieve memories by query using vector similarity search.

        Args:
            query: The query string for similarity search.
            memory_type: The type of memory to search for.
            memory_target: The target of memory to search for.
            workspace_id: The workspace ID to search in.

        Returns:
            List[MemoryNode]: List of matching memories.
        """
        memory_type_list = [MemoryType(memory_type)]
        if self.enable_summary_memory:
            memory_type_list.append(MemoryType.SUMMARY)

        nodes: List[VectorNode] = await self.vector_store.async_search(
            query=query,
            workspace_id=workspace_id,
            top_k=self.top_k,
            filter_dict={
                "metadata.memory_type": memory_type_list,
                "metadata.memory_target": [memory_target],
            },
        )
        return [MemoryNode.from_vector_node(n) for n in nodes]

    async def async_execute(self):
        """Execute the retrieve memory operation.

        Retrieves memories from the memory store based on query texts.
        The operation handles both single query and multiple queries inputs.

        Raises:
            ValueError: If no valid query texts are provided.
        """
        workspace_id: str = self.workspace_id

        # Get query items based on mode
        # Each item contains: memory_type, memory_target, query
        if self.enable_multiple:
            query_items: List[dict] = self.input_dict.get("query_items", [])
        else:
            query_items: List[dict] = [
                {
                    "memory_type": self.input_dict.get("memory_type", ""),
                    "memory_target": self.input_dict.get("memory_target", ""),
                    "query": self.input_dict.get("query", ""),
                }
            ]

        # Filter out items with empty query
        query_items = [item for item in query_items if item.get("query")]

        if not query_items:
            self.set_output("No valid query texts provided for retrieval.")
            return

        # Perform retrieval
        memories: List[MemoryNode] = []
        for item in query_items:
            memories.extend(
                await self.retrieve_by_query(
                    query=item["query"],
                    memory_type=item["memory_type"],
                    memory_target=item["memory_target"],
                    workspace_id=workspace_id,
                )
            )

        # Deduplicate by memory_id
        seen_ids: set = set()
        unique_memories: List[MemoryNode] = []
        for m in memories:
            if m.memory_id not in seen_ids:
                seen_ids.add(m.memory_id)
                unique_memories.append(m)
        memories = unique_memories

        # Format output message
        if not memories:
            self.set_output(f"No memories found in workspace={workspace_id}.")
        else:
            self.set_output("\n".join([m.format_memory() for m in memories]))
