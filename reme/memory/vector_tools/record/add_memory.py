"""Add memory to vector store"""

from loguru import logger

from .memory_handler import MemoryHandler
from ..base_memory_tool import BaseMemoryTool
from ....core.schema import ToolCall


class AddMemory(BaseMemoryTool):
    """Tool to add memories to vector store"""

    def __init__(
        self,
        enable_memory_target: bool = False,
        enable_when_to_use: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.enable_memory_target: bool = enable_memory_target
        self.enable_when_to_use: bool = enable_when_to_use

    def _build_memory_parameters(self) -> dict:
        """Build the memory parameters schema based on enabled features."""
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
                "description": "add a memory to vector store for future retrieval.",
                "parameters": self._build_memory_parameters(),
            },
        )

    def _build_multiple_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": "add multiple memories to vector store for future retrieval.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memories": {
                            "type": "array",
                            "description": "list of memories to store.",
                            "items": self._build_memory_parameters(),
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
            # Parse and prepare memory data
            memory_dicts = []
            for mem in target_memories:
                memory_content = mem.get("memory_content", "")
                message_time = mem.get("message_time", "")
                when_to_use = mem.get("when_to_use", "") if self.enable_when_to_use else ""
                metadata = {}
                try:
                    metadata["time_int"] = int(message_time.split(" ")[0].replace("-", ""))
                except Exception:
                    logger.warning(f"Invalid message time format: {message_time}")

                memory_dicts.append(
                    {
                        "content": memory_content,
                        "when_to_use": when_to_use,
                        "message_time": message_time,
                        "ref_memory_id": self.history_id,
                        "author": self.author,
                        "metadata": metadata,
                    },
                )

            if memory_dicts:
                handler = MemoryHandler(target, self.service_context)
                memory_nodes = await handler.add_batch(memory_dicts)
                all_memory_nodes.extend(memory_nodes)

        if not all_memory_nodes:
            return "No valid memories provided."

        self.memory_nodes.extend(all_memory_nodes)
        output = f"Successfully added {len(all_memory_nodes)} memories."
        logger.info(output)
        return output
