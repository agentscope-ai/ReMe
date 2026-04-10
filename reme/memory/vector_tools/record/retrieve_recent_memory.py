"""Retrieve most recent memories from vector store"""

from loguru import logger

from .memory_handler import MemoryHandler
from ..base_memory_tool import BaseMemoryTool
from ....core.schema import ToolCall, MemoryNode
from ....core.utils import deduplicate_memories


class RetrieveRecentMemory(BaseMemoryTool):
    """Tool to retrieve most recent memories sorted by time"""

    def __init__(self, top_k: int = 20, **kwargs):
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)
        self.top_k: int = top_k

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": "retrieve the most recent memories sorted by message time (newest first).",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        )

    async def execute(self):
        handler = MemoryHandler(self.memory_target, self.service_context)

        memory_nodes: list[MemoryNode] = await handler.list(
            limit=self.top_k,
            sort_key="message_time",
            reverse=True,
        )
        memory_nodes = deduplicate_memories(memory_nodes)

        retrieved_ids = {n.memory_id for n in self.retrieved_nodes if n.memory_id}
        new_nodes = [n for n in memory_nodes if n.memory_id not in retrieved_ids]
        self.retrieved_nodes.extend(new_nodes)
        self.memory_nodes.extend(new_nodes)

        if not new_nodes:
            output = "No new memories found."
        else:
            output = "\n".join([n.format(ref_memory_id_key="history_id") for n in new_nodes])

        logger.info(f"Retrieved {len(memory_nodes)} memories, {len(new_nodes)} new after deduplication")
        return output
