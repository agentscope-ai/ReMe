"""Delete memory from vector store"""

from loguru import logger

from .memory_handler import MemoryHandler
from ..base_memory_tool import BaseMemoryTool
from ....core.schema import ToolCall


class DeleteMemory(BaseMemoryTool):
    """Tool to delete memories from vector store"""

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": "delete a memory from vector store using its unique ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "memory_id of the memory to delete.",
                        },
                    },
                    "required": ["memory_id"],
                },
            },
        )

    def _build_multiple_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": "delete multiple memories from vector store using their unique IDs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_ids": {
                            "type": "array",
                            "description": "memory_ids of memories to delete.",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["memory_ids"],
                },
            },
        )

    async def execute(self):
        memory_ids = self.context.get("memory_ids") or []
        if not memory_ids:
            memory_ids = [self.context.get("memory_id", "")]

        handler = MemoryHandler(self.memory_target, self.service_context)
        await handler.delete(memory_ids)
        self.memory_nodes.extend(memory_ids)

        output = f"Successfully deleted {len(memory_ids)} memories."
        logger.info(output)
        return output
