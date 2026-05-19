"""Retrieve relevant history chunks from vector store."""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ....core.schema import MemoryNode, ToolCall


class RetrieveHistory(BaseMemoryTool):
    """Retrieve history chunks by semantic similarity."""

    def __init__(self, top_k: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k

    def _build_query_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query used to retrieve relevant history chunks.",
                },
                "history_id": {
                    "type": "string",
                    "description": "Optional history_id. If provided, search only within this history group.",
                },
            },
            "required": ["query"],
        }

    def _build_tool_call(self) -> ToolCall:
        """Build and return the tool call schema."""
        return ToolCall(
            **{
                "description": "Retrieve relevant history chunks. Optionally restrict search by history_id.",
                "parameters": self._build_query_parameters(),
            },
        )

    def _build_multiple_tool_call(self) -> ToolCall:
        """Build and return the tool call schema for multiple retrieval queries."""
        return ToolCall(
            **{
                "description": "Retrieve relevant history chunks for multiple queries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_items": {
                            "type": "array",
                            "description": "List of query items.",
                            "items": self._build_query_parameters(),
                        },
                    },
                    "required": ["query_items"],
                },
            },
        )

    async def execute(self):
        """Execute semantic retrieval over stored history chunks."""
        query_items = self.context.get("query_items", []) if self.enable_multiple else [self.context]
        if not query_items:
            output = "No history retrieval queries provided."
            logger.warning(output)
            return output

        memory_nodes: list[MemoryNode] = []
        for item in query_items:
            query = item["query"]
            filters = {"node_kind": "history_chunk"}
            history_id = item.get("history_id")
            if history_id:
                filters["history_id"] = history_id

            vector_nodes = await self.vector_store.search(
                query=query,
                limit=self.top_k,
                filters=filters,
            )
            memory_nodes.extend(MemoryNode.from_vector_node(node) for node in vector_nodes)

        retrieved_ids = {node.memory_id for node in self.retrieved_nodes if node.memory_id}
        new_nodes = []
        for node in memory_nodes:
            if node.memory_id not in retrieved_ids:
                retrieved_ids.add(node.memory_id)
                new_nodes.append(node)

        self.retrieved_nodes.extend(new_nodes)

        if not new_nodes:
            output = "No relevant history found."
        else:
            output = "\n\n".join(self._format_history_chunk(node) for node in new_nodes)

        logger.info(f"Retrieved {len(memory_nodes)} history chunks, {len(new_nodes)} new after deduplication")
        return output

    @staticmethod
    def _format_history_chunk(node: MemoryNode) -> str:
        history_id = node.metadata.get("history_id") or node.ref_memory_id
        chunk_index = node.metadata.get("chunk_index")
        score = node.metadata.get("score") or node.score

        header = f"Historical Dialogue Chunk[{node.memory_id}]"
        if history_id:
            header += f" history_id={history_id}"
        if chunk_index is not None and chunk_index != "":
            header += f" chunk_index={chunk_index}"
        if score:
            header += f" score={float(score):.4f}"

        return f"{header}\n{node.content}"
