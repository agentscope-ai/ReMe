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

    def __init__(self, include_identity: bool = False, **kwargs):
        """Initialize the ReadMetaMemoryOp.

        Args:
            include_identity: Whether to include IDENTITY type meta memory. Defaults to True.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)
        self.include_identity = include_identity

    def _load_meta_memories(self, workspace_id: str) -> List[Dict[str, str]]:
        """Load meta memories from file.

        Args:
            workspace_id: The workspace ID.

        Returns:
            List[Dict[str, str]]: List of memory metadata entries.
        """
        metadata_handler = self.get_metadata_handler(workspace_id)
        result = self.load_metadata_value(metadata_handler, "meta_memories")
        memories = result if result is not None else []

        if not self.include_identity:
            memories = [m for m in memories if MemoryType(m.get("memory_type")) is not MemoryType.IDENTITY]

        return memories

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
