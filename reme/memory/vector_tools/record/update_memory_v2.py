"""Update memory in vector store"""

from loguru import logger

from .memory_handler import MemoryHandler
from ..base_memory_tool import BaseMemoryTool
from ....core.schema import ToolCall


class UpdateMemoryV2(BaseMemoryTool):
    """Tool to update memories in vector store by deleting and adding memory entries"""

    def __init__(
        self,
        name="update_memory",
        enable_memory_target: bool = False,
        enable_when_to_use: bool = False,
        **kwargs,
    ):
        kwargs["enable_multiple"] = True
        super().__init__(name=name, **kwargs)
        self.enable_memory_target: bool = enable_memory_target
        self.enable_when_to_use: bool = enable_when_to_use

    def _build_add_memory_parameters(self) -> dict:
        """Build the add memory parameters schema based on enabled features."""
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

    def _build_multiple_tool_call(self) -> ToolCall:
        """Build and return the multiple tool call schema"""
        return ToolCall(
            **{
                "description": "update memories by removing and adding memory entries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_ids_to_delete": {
                            "type": "array",
                            "description": "List of memory IDs to delete",
                            "items": {
                                "type": "string",
                            },
                        },
                        "memories_to_add": {
                            "type": "array",
                            "description": "List of memories to add",
                            "items": self._build_add_memory_parameters(),
                        },
                    },
                    "required": ["memory_ids_to_delete", "memories_to_add"],
                },
            },
        )

    async def execute(self):
        # Get parameters
        memory_ids_to_delete = self.context.get("memory_ids_to_delete", [])
        memory_ids_to_delete = sorted({mid for mid in memory_ids_to_delete if mid})
        memories_to_add = self.context.get("memories_to_add", [])

        if not memory_ids_to_delete and not memories_to_add:
            return "No memories to remove or add, operation completed."

        # Group memories by memory_target if enabled
        if self.enable_memory_target:
            memories_by_target = {}
            for mem in memories_to_add:
                target = mem.get("memory_target", self.memory_target)
                if target not in memories_by_target:
                    memories_by_target[target] = []
                memories_by_target[target].append(mem)
        else:
            memories_by_target = {self.memory_target: memories_to_add}

        # Delete memories (all at once, regardless of target)
        removed_count = 0
        if memory_ids_to_delete:
            # Use the default memory_target handler for deletion
            handler = MemoryHandler(self.memory_target, self.service_context)
            await handler.delete(memory_ids_to_delete)
            removed_count = len(memory_ids_to_delete)

        # Add new memories by target
        added_count = 0
        all_memory_nodes = []
        for target, target_memories in memories_by_target.items():
            # Parse and prepare add data
            add_dicts = []
            for mem in target_memories:
                memory_content = mem.get("memory_content", "")
                message_time = mem.get("message_time", "")
                when_to_use = mem.get("when_to_use", "") if self.enable_when_to_use else ""
                metadata = {}
                try:
                    metadata["time_int"] = int(message_time.split(" ")[0].replace("-", ""))
                except Exception:
                    logger.warning(f"Invalid message time format: {message_time}")

                add_dicts.append(
                    {
                        "content": memory_content,
                        "when_to_use": when_to_use,
                        "message_time": message_time,
                        "ref_memory_id": self.history_id,
                        "author": self.author,
                        "metadata": metadata,
                    },
                )

            if add_dicts:
                handler = MemoryHandler(target, self.service_context)
                memory_nodes = await handler.add_batch(add_dicts)
                all_memory_nodes.extend(memory_nodes)
                added_count += len(memory_nodes)

        # Extend memory_nodes for tracking
        self.memory_nodes.extend(all_memory_nodes)

        # Build output message
        operations = []
        if removed_count > 0:
            operations.append(f"removed {removed_count} old memories.")
        if added_count > 0:
            operations.append(f"added {added_count} new memories.")
        operations.append("Operation completed.")
        logger.info("\n".join(operations))
        return "\n".join(operations)
