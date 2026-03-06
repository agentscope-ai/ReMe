"""Add draft memory and retrieve similar memories from vector store"""

from loguru import logger

from .memory_handler import MemoryHandler
from ..base_memory_tool import BaseMemoryTool
from ....core.schema import ToolCall, MemoryNode
from ....core.utils import deduplicate_memories


class AddAndRetrieveSimilarMemory(BaseMemoryTool):
    """Tool to add draft memory and retrieve similar memories"""

    def __init__(
        self,
        top_k: int = 20,
        enable_memory_target: bool = False,
        enable_when_to_use: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k: int = top_k
        self.enable_memory_target: bool = enable_memory_target
        self.enable_when_to_use: bool = enable_when_to_use

    def _build_query_parameters(self) -> dict:
        """Build the query parameters schema"""
        properties = {
            "message_time": {
                "type": "string",
                "description": "message time, e.g. '2020-01-01 00:00:00'",
            },
            "memory_content": {
                "type": "string",
                "description": "content of the memory.",
            },
        }
        required = ["message_time", "memory_content"]

        if self.enable_when_to_use:
            properties["when_to_use"] = {
                "type": "string",
                "description": "description of when to use this memory.",
            }
            required.append("when_to_use")

        if self.enable_memory_target:
            properties["memory_target"] = {
                "type": "string",
                "description": "target memory type for this memory.",
            }
            required.append("memory_target")

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": "Add memory and retrieve similar memories from the vector store.",
                "parameters": self._build_query_parameters(),
            },
        )

    def _build_multiple_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": "Add memory and retrieve similar memories from the vector store.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "description": "items",
                            "items": self._build_query_parameters(),
                        },
                    },
                    "required": ["items"],
                },
            },
        )

    async def execute(self):
        if self.enable_multiple:
            draft_items = self.context.get("items", [])
        else:
            draft_items = [self.context]

        queries_by_target: dict[str, list[dict]] = {}
        for item in draft_items:
            if self.enable_memory_target:
                target = item["memory_target"]
            else:
                target = self.memory_target
            if target not in queries_by_target:
                queries_by_target[target] = []

            queries_by_target[target].append(
                {
                    "query": item["memory_content"],
                    "limit": self.top_k,
                    "filters": {},
                },
            )

        # Execute batch searches for each target
        memory_nodes: list[MemoryNode] = []
        for target, searches in queries_by_target.items():
            handler = MemoryHandler(target, self.service_context)
            nodes = await handler.batch_search(searches)
            memory_nodes.extend(nodes)

        memory_nodes = deduplicate_memories(memory_nodes)
        retrieved_ids = {n.memory_id for n in self.retrieved_nodes if n.memory_id}
        new_nodes = [n for n in memory_nodes if n.memory_id not in retrieved_ids]
        self.retrieved_nodes.extend(new_nodes)

        if not new_nodes:
            output = "No similar memories found."
        else:
            output = "\n".join([n.format(ref_memory_id_key="history_id") for n in new_nodes])

        logger.info(f"Retrieved {len(memory_nodes)} similar memories, {len(new_nodes)} new after deduplication")
        return output
