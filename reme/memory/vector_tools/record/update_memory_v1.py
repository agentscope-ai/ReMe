"""Update memory in vector store"""

from loguru import logger

from .memory_handler import MemoryHandler
from ..base_memory_tool import BaseMemoryTool
from ....core.schema import ToolCall


class UpdateMemoryV1(BaseMemoryTool):
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

    def _build_memory_parameters(self, include_memory_id: bool = False) -> dict:
        """Build the memory parameters schema based on enabled features.

        Args:
            include_memory_id: If True, include memory_id field (for updates)
        """
        properties = {}
        required = []

        if include_memory_id:
            properties["memory_id"] = {
                "type": "string",
                "description": "ID of the memory to update",
            }
            required.append("memory_id")

        properties.update(
            {
                "message_time": {
                    "type": "string",
                    "description": "message time, e.g. '2020-01-01 00:00:00'",
                },
                "memory_content": {
                    "type": "string",
                    "description": "content of the memory.",
                },
            },
        )
        required.extend(["message_time", "memory_content"])

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
                "description": "update memories by updating existing memories and adding new memory entries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memories_to_update": {
                            "type": "array",
                            "description": "List of memories to update",
                            "items": self._build_memory_parameters(include_memory_id=True),
                        },
                        "memories_to_add": {
                            "type": "array",
                            "description": "List of memories to add",
                            "items": self._build_memory_parameters(include_memory_id=False),
                        },
                    },
                    "required": ["memories_to_update", "memories_to_add"],
                },
            },
        )

    async def execute(self):
        # Get parameters
        memories_to_update = self.context.get("memories_to_update", [])
        memories_to_add = self.context.get("memories_to_add", [])

        if not memories_to_update and not memories_to_add:
            return "No memories to update or add, operation completed."

        # Step 1: Collect and delete all old memories that need to be updated
        if memories_to_update:
            # Group deletion IDs by memory_target if enabled
            if self.enable_memory_target:
                delete_by_target = {}
                for mem in memories_to_update:
                    target = mem.get("memory_target", self.memory_target)
                    memory_id = mem.get("memory_id")
                    if memory_id:
                        if target not in delete_by_target:
                            delete_by_target[target] = []
                        delete_by_target[target].append(memory_id)
            else:
                delete_by_target = {
                    self.memory_target: [mem.get("memory_id") for mem in memories_to_update if mem.get("memory_id")],
                }

            # Delete old memories for each target
            for target, memory_ids in delete_by_target.items():
                if memory_ids:
                    handler = MemoryHandler(target, self.service_context)
                    await handler.delete(memory_ids)

        # Step 2: Prepare all memories to add (both updated and new)
        all_memories_to_add = []

        # Add memories from updates
        if memories_to_update:
            for mem in memories_to_update:
                target = (
                    mem.get(
                        "memory_target",
                        self.memory_target,
                    )
                    if self.enable_memory_target
                    else self.memory_target
                )
                all_memories_to_add.append((target, mem))

        # Add new memories
        if memories_to_add:
            for mem in memories_to_add:
                target = (
                    mem.get(
                        "memory_target",
                        self.memory_target,
                    )
                    if self.enable_memory_target
                    else self.memory_target
                )
                all_memories_to_add.append((target, mem))

        # Step 3: Group all memories by target and add them in batch
        memories_by_target = {}
        for target, mem in all_memories_to_add:
            if target not in memories_by_target:
                memories_by_target[target] = []
            memories_by_target[target].append(mem)

        # Process each target and add memories
        all_memory_nodes = []
        updated_count = len(memories_to_update)
        added_count = len(memories_to_add)

        for target, target_memories in memories_by_target.items():
            # Prepare memory data for batch add
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

            # Batch add all memories for this target
            if memory_dicts:
                handler = MemoryHandler(target, self.service_context)
                memory_nodes = await handler.add_batch(memory_dicts)
                all_memory_nodes.extend(memory_nodes)

        # Extend memory_nodes for tracking
        self.memory_nodes.extend(all_memory_nodes)

        # Build output message
        operations = []
        if updated_count > 0:
            operations.append(f"updated {updated_count} memories.")
        if added_count > 0:
            operations.append(f"added {added_count} new memories.")
        operations.append("Operation completed.")
        logger.info("\n".join(operations))
        return "\n".join(operations)
