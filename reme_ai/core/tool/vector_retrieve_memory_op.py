"""Vector retrieve memory operation for searching memories from the memory store.

This module provides the VectorRetrieveMemoryOp class for retrieving memories
by query texts using vector similarity search.
"""

from typing import Dict, List

from flowllm.core.schema import VectorNode

from .base_memory_tool_op import BaseMemoryToolOp
from .. import C
from ..enumeration import MemoryType
from ..schema import MemoryNode


@C.register_op()
class VectorRetrieveMemoryOp(BaseMemoryToolOp):
    """Operation for retrieving memories from the memory store.

    This operation supports both single and multiple query modes,
    controlled by the `enable_multiple` parameter inherited from BaseMemoryToolOp.

    When `add_memory_type_target` is False, memory_type and memory_target are not
    included in the tool call schema, and will be retrieved from context instead.
    """

    def __init__(
            self,
            enable_summary_memory: bool = False,
            add_memory_type_target: bool = False,
            top_k: int = 20,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.enable_summary_memory: bool = enable_summary_memory
        self.add_memory_type_target: bool = add_memory_type_target
        self.top_k: int = C.service_config.metadata.get("top_k", top_k)

    def build_input_schema(self) -> dict:
        """Build input schema for single query mode.

        Returns:
            dict: Input schema for retrieving memories. When add_memory_type_target is True,
                  includes memory_type, memory_target, and query. Otherwise only query.
        """
        schema = {
            "query": {
                "type": "string",
                "description": self.get_prompt("query"),
                "required": True,
            },
        }

        if self.add_memory_type_target:
            memory_type_target_schema = {
                "memory_type": {
                    "type": "string",
                    "description": self.get_prompt("memory_type"),
                    "enum": [
                        MemoryType.IDENTITY.value,
                        MemoryType.PERSONAL.value,
                        MemoryType.PROCEDURAL.value,
                    ],
                    "required": True,
                },
                "memory_target": {
                    "type": "string",
                    "description": self.get_prompt("memory_target"),
                    "required": True,
                },
            }
            schema = {**memory_type_target_schema, **schema}

        return schema

    def build_multiple_input_schema(self) -> dict:
        """Build input schema for multiple query mode.

        Returns:
            dict: Input schema for retrieving memories with list of query items.
                  When add_memory_type_target is True, each item includes memory_type,
                  memory_target, and query. Otherwise only query.
        """
        # Build item properties by combining schemas
        item_properties = {}
        item_required = []

        if self.add_memory_type_target:
            # Add memory_type and memory_target schemas
            item_properties["memory_type"] = {
                "type": "string",
                "description": self.get_prompt("memory_type"),
                "enum": [
                    MemoryType.IDENTITY.value,
                    MemoryType.PERSONAL.value,
                    MemoryType.PROCEDURAL.value,
                ],
            }
            item_properties["memory_target"] = {
                "type": "string",
                "description": self.get_prompt("memory_target"),
            }
            item_required.extend(["memory_type", "memory_target"])

        # Add query schema
        item_properties["query"] = {
            "type": "string",
            "description": self.get_prompt("query"),
        }
        item_required.append("query")

        return {
            "query_items": {
                "type": "array",
                "description": self.get_prompt("query_items"),
                "required": True,
                "items": {
                    "type": "object",
                    "properties": item_properties,
                    "required": item_required,
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
        # Build memory type list, including summary if enabled
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
        memory_nodes: List[MemoryNode] = [MemoryNode.from_vector_node(n) for n in nodes]
        filtered_memory_nodes = []
        for memory_node in memory_nodes:
            # For TOOL type memories, only keep those where when_to_use matches the query (tool name)
            if memory_node.memory_type == MemoryType.TOOL and memory_node.when_to_use != query:
                continue
            filtered_memory_nodes.append(memory_node)

        return filtered_memory_nodes

    @staticmethod
    def _deduplicate_memories(memories: List[MemoryNode]) -> List[MemoryNode]:
        """Remove duplicate memories based on memory_id while preserving order.

        Args:
            memories: List of memory nodes that may contain duplicates.

        Returns:
            List[MemoryNode]: Deduplicated list of memory nodes.
        """
        seen_memories: Dict[str, MemoryNode] = {}
        for memory in memories:
            if memory.memory_id not in seen_memories:
                seen_memories[memory.memory_id] = memory
        return list(seen_memories.values())

    async def async_execute(self):
        """Execute the retrieve memory operation.

        Retrieves memories from the memory store based on query texts.
        The operation handles both single query and multiple queries inputs.

        When add_memory_type_target is False, memory_type and memory_target
        are retrieved from self.context instead of from each query item.
        """
        workspace_id: str = self.workspace_id

        # Get default memory_type and memory_target from input_dict
        default_memory_type: str = self.context.get("memory_type", "")
        default_memory_target: str = self.context.get("memory_target", "")

        # Get query items based on mode
        if self.enable_multiple:
            query_items: List[dict] = self.context.get("query_items", [])
        else:
            query_items: List[dict] = [
                {
                    "memory_type": default_memory_type,
                    "memory_target": default_memory_target,
                    "query": self.context.get("query", ""),
                }
            ]

        # Filter out items with empty query
        query_items = [item for item in query_items if item.get("query")]

        if not query_items:
            self.set_output("No valid query texts provided for retrieval.")
            return

        # Perform retrieval for all query items
        memories: List[MemoryNode] = []
        for item in query_items:
            # Use item's memory_type/memory_target if available,
            # otherwise fall back to default values from input_dict
            memory_type = item.get("memory_type") or default_memory_type
            memory_target = item.get("memory_target") or default_memory_target

            retrieved = await self.retrieve_by_query(
                query=item["query"],
                memory_type=memory_type,
                memory_target=memory_target,
                workspace_id=workspace_id,
            )
            memories.extend(retrieved)

        # Deduplicate and format output
        memories = self._deduplicate_memories(memories)

        if not memories:
            output = f"No memories found in workspace={workspace_id}."
        else:
            output = "\n".join([m.format_memory() for m in memories])
        self.set_output(output)
