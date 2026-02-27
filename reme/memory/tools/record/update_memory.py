"""Update memory in vector store"""

from loguru import logger

from .memory_handler import MemoryHandler
from ..base_memory_tool import BaseMemoryTool
from ....core.schema import ToolCall


class UpdateMemory(BaseMemoryTool):
    """Tool to update memories in vector store"""

    def __init__(
        self,
        enable_memory_target: bool = False,
        enable_when_to_use: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.enable_memory_target: bool = enable_memory_target
        self.enable_when_to_use: bool = enable_when_to_use

    def _build_update_parameters(self) -> dict:
        """Build the update parameters schema based on enabled features."""
        properties = {
            "memory_id": {
                "type": "string",
                "description": "unique identifier of memory to update.",
            },
            "message_time": {
                "type": "string",
                "description": "message time, e.g. '2020-01-01 00:00:00'",
            },
            "memory_content": {
                "type": "string",
                "description": "new content of the memory.",
            },
        }
        required = ["memory_id", "message_time", "memory_content"]

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
                "description": "update a memory in vector store by replacing old memory with new content.",
                "parameters": self._build_update_parameters(),
            },
        )

    def _build_multiple_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": "update multiple memories in vector store by replacing old memories with new content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memories": {
                            "type": "array",
                            "description": "list of memory update objects.",
                            "items": self._build_update_parameters(),
                        },
                    },
                    "required": ["memories"],
                },
            },
        )

    async def execute(self):
        if self.enable_multiple:
            memories = self.context.get("memories", [])
        else:
            memories = [self.context]

        # Group memories by memory_target if enabled
        if self.enable_memory_target:
            memories_by_target = {}
            for mem in memories:
                target = mem["memory_target"]
                if target not in memories_by_target:
                    memories_by_target[target] = []
                memories_by_target[target].append(mem)
        else:
            memories_by_target = {self.memory_target: memories}

        # Process each memory_target group
        all_memory_nodes = []
        for target, target_memories in memories_by_target.items():
            # Parse and prepare update data
            update_dicts = []
            for mem in target_memories:
                memory_content = mem.get("memory_content", "")
                message_time = mem.get("message_time", "")
                when_to_use = mem.get("when_to_use", "") if self.enable_when_to_use else ""
                metadata = {}
                try:
                    metadata["time_int"] = int(message_time.split(" ")[0].replace("-", ""))
                except Exception:
                    logger.warning(f"Invalid message time format: {message_time}")

                update_dicts.append(
                    {
                        "memory_id": mem.get("memory_id", ""),
                        "content": memory_content,
                        "when_to_use": when_to_use,
                        "message_time": message_time,
                        "author": self.author,
                        "metadata": metadata,
                    },
                )

            if update_dicts:
                handler = MemoryHandler(target, self.service_context)
                memory_nodes = await handler.update_batch(update_dicts)
                all_memory_nodes.extend(memory_nodes)

        if not all_memory_nodes:
            return "No valid memories provided."

        self.memory_nodes.extend(all_memory_nodes)
        output = f"Successfully updated {len(all_memory_nodes)} memories."
        logger.info(output)
        return output
