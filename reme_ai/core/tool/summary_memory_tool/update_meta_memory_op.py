"""Update meta memory operation for updating memory metadata.

This module provides the UpdateMetaMemoryOp class for updating memory metadata
(memory_type and memory_target) using file-based storage with CacheHandler.
"""

import json
from typing import List

from ..base_memory_tool_op import BaseMemoryToolOp
from ... import C
from ...enumeration import MemoryType


@C.register_op()
class UpdateMetaMemoryOp(BaseMemoryToolOp):
    """Operation for updating memory metadata using file-based storage."""

    def __init__(self, **kwargs):
        kwargs["enable_multiple"] = True
        super().__init__(**kwargs)

    def build_multiple_input_schema(self) -> dict:
        """Build input schema for meta memory update.

        Returns:
            dict: Input schema for updating memory metadata.
        """
        return {
            "meta_memories": {
                "type": "array",
                "description": self.get_prompt("meta_memories"),
                "required": True,
                "items": {
                    "type": "object",
                    "properties": {
                        "memory_type": {
                            "type": "string",
                            "description": self.get_prompt("memory_type"),
                            "enum": [mt.value for mt in MemoryType],
                        },
                        "memory_target": {
                            "type": "string",
                            "description": self.get_prompt("memory_target"),
                        },
                    },
                    "required": ["memory_type", "memory_target"],
                },
            },
        }

    def _save_meta_memories(self, workspace_id: str, memories: List[dict]) -> bool:
        """Save meta memories to file.

        Args:
            workspace_id: The workspace ID.
            memories: List of memory metadata entries.

        Returns:
            bool: Whether the save was successful.
        """
        metadata_handler = self.get_metadata_handler(f"{workspace_id}/meta")
        return self.save_metadata_value(metadata_handler, "meta_memories", memories)

    async def async_execute(self):
        """Execute the update meta memory operation.

        Updates memory metadata to file storage based on memory_type and memory_target.
        """

        workspace_id: str = self.workspace_id
        meta_memories: List[dict] = self.input_dict.get("meta_memories", [])
        self._save_meta_memories(workspace_id, meta_memories)
        self.set_output(json.dumps(meta_memories, ensure_ascii=False))
