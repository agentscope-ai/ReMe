"""Vector retrieve memory operation for searching memories from the vector store.

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
    """Operation for retrieving memories from the vector store using vector similarity search.

    This operation supports both single and multiple query modes,
    controlled by the `enable_multiple` parameter inherited from BaseMemoryToolOp.
    """

    def __init__(self, enable_summary_memory: bool = False, top_k: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.enable_summary_memory: bool = enable_summary_memory
        self.top_k: int = self.service_config_metadata.get("top_k", top_k)

    def build_input_schema(self) -> dict:
        """Build input schema for single query mode.

        Returns:
            dict: Input schema for retrieving memories with a single query text.
        """
        return {
            "query": {
                "type": "string",
                "description": self.get_prompt("query"),
                "required": True,
            }
        }

    def build_multiple_input_schema(self) -> dict:
        """Build input schema for multiple query mode.

        Returns:
            dict: Input schema for retrieving memories with multiple query texts.
        """
        return {
            "queries": {
                "type": "array",
                "description": self.get_prompt("queries"),
                "required": True,
                "items": {"type": "string"},
            }
        }

    async def retrieve_by_query(self, query: str, workspace_id: str) -> List[MemoryNode]:
        """Retrieve memories by query using vector similarity search.

        Args:
            query: The query string for similarity search.
            workspace_id: The workspace ID to search in.

        Returns:
            List[MemoryNode]: List of matching memories.
        """
        memory_type_list = [self.memory_type]
        if self.enable_summary_memory:
            memory_type_list.append(MemoryType.SUMMARY)

        nodes: List[VectorNode] = await self.vector_store.async_search(
            query=query,
            workspace_id=workspace_id,
            top_k=self.top_k,
            filter_dict={
                "metadata.memory_type": memory_type_list,
                "metadata.memory_target": [self.memory_target],
            },
        )
        return [MemoryNode.from_vector_node(n) for n in nodes]

    async def async_execute(self):
        """Execute the vector retrieve memory operation.

        Retrieves memories from the vector store based on query texts using
        vector similarity search. The operation handles both single query (string)
        and multiple queries (list) inputs.

        Raises:
            ValueError: If no valid query texts are provided.
        """
        workspace_id: str = self.workspace_id

        # Get query texts based on mode
        if self.enable_multiple:
            query_list: List[str] = self.input_dict.get("queries", [])
        else:
            query_list: List[str] = [self.input_dict.get("query", "")]

        # Filter out empty strings
        query_list = [q for q in query_list if q]

        if not query_list:
            self.set_output("No valid query texts provided for retrieval.")
            return

        # Perform retrieval
        memories: List[MemoryNode] = []
        for query in query_list:
            memories.extend(await self.retrieve_by_query(query, workspace_id))

        # Format output message
        if not memories:
            self.set_output(f"No memories found in workspace={workspace_id}.")
        else:
            self.set_output("\n".join([m.format_memory() for m in memories]))
