"""Read meta memory operation for retrieving memory metadata.

This module provides the ReadMetaMemoryOp class for reading memory metadata
(memory_type and memory_target) from file storage.
"""

from typing import List, Dict

from .base_memory_tool_op import BaseMemoryToolOp
from .. import C
from ..enumeration.memory_type import MemoryType


@C.register_op()
class ReadMetaMemoryOp(BaseMemoryToolOp):
    """Operation for reading memory metadata from file storage."""

    def __init__(
        self,
        enable_tool_memory: bool = True,
        enable_identity_memory: bool = False,
        **kwargs
    ):
        """Initialize the ReadMetaMemoryOp.

        Args:
            enable_tool_memory: Whether to include TOOL type meta memory. Defaults to True.
            enable_identity_memory: Whether to include IDENTITY type meta memory. Defaults to False.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)
        self.enable_tool_memory = enable_tool_memory
        self.enable_identity_memory = enable_identity_memory

    def _load_meta_memories(self, workspace_id: str) -> List[Dict[str, str]]:
        """Load meta memories from file.

        Args:
            workspace_id: The workspace ID.

        Returns:
            List[Dict[str, str]]: List of memory metadata entries.
        """
        metadata_handler = self.get_metadata_handler(workspace_id)
        result = self.load_metadata_value(metadata_handler, "meta_memories")
        all_memories = result if result is not None else []

        # Filter to only include PERSONAL and PROCEDURAL by default
        filtered_memories = []
        for m in all_memories:
            memory_type = MemoryType(m.get("memory_type"))
            memory_target = m.get("memory_target")
            
            if memory_type in (MemoryType.PERSONAL, MemoryType.PROCEDURAL):
                filtered_memories.append(m)
        
        if self.enable_tool_memory:
            filtered_memories.append({"memory_type": MemoryType.TOOL.value, "memory_target": "tool_guidelines"})
        
        if self.enable_identity_memory:
            filtered_memories.append({"memory_type": MemoryType.IDENTITY.value, "memory_target": "self"})

        return filtered_memories

    def _format_memory_metadata(self, memories: List[Dict[str, str]]) -> str:
        """Format memory metadata into a readable string.

        Args:
            memories: List of memory metadata entries.

        Returns:
            str: Formatted memory metadata string.
        """
        if not memories:
            return ""

        lines = []
        for memory in memories:
            memory_type = memory["memory_type"]
            memory_target = memory["memory_target"]
            description = self.get_prompt(f"type_{memory_type}")
            lines.append(f"- {memory_type}({memory_target}): {description}")

        return "\n".join(lines)

    async def async_execute(self):
        """Execute the read meta memory operation.

        Reads memory metadata from file storage and formats output.
        """
        workspace_id: str = self.workspace_id
        memories = self._load_meta_memories(workspace_id)

        if memories:
            formatted = self._format_memory_metadata(memories)
            self.set_output(formatted)
        else:
            self.set_output(f"No memory metadata found in workspace={workspace_id}.")
