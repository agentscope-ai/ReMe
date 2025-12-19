"""Add meta memory operation for adding memory metadata.

This module provides the AddMetaMemoryOp class for adding memory metadata
(memory_type and memory_target) using file-based storage with CacheHandler.
"""

import json
from typing import List

from .base_memory_tool_op import BaseMemoryToolOp
from .. import C
from ..enumeration import MemoryType


@C.register_op()
class AddMetaMemoryOp(BaseMemoryToolOp):
    """Operation for adding memory metadata using file-based storage."""

    def build_input_schema(self) -> dict:
        """Build input schema for single meta memory addition.

        Returns:
            dict: Input schema for adding a single memory metadata entry.
        """
        return {
            "memory_type": {
                "type": "string",
                "description": self.get_prompt("memory_type"),
                "enum": [mt.value for mt in MemoryType],
                "required": True,
            },
            "memory_target": {
                "type": "string",
                "description": self.get_prompt("memory_target"),
                "required": True,
            },
        }

    def build_multiple_input_schema(self) -> dict:
        """Build input schema for multiple meta memory addition.

        Returns:
            dict: Input schema for adding multiple memory metadata entries.
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

    def _load_meta_memories(self, workspace_id: str) -> List[dict]:
        """Load existing meta memories from file.

        Args:
            workspace_id: The workspace ID.

        Returns:
            List[dict]: List of existing memory metadata entries.
        """
        metadata_handler = self.get_metadata_handler(workspace_id)
        result = self.load_metadata_value(metadata_handler, "meta_memories")
        return result if result is not None else []

    def _save_meta_memories(self, workspace_id: str, memories: List[dict]) -> bool:
        """Save meta memories to file.

        Args:
            workspace_id: The workspace ID.
            memories: List of memory metadata entries.

        Returns:
            bool: Whether the save was successful.
        """
        metadata_handler = self.get_metadata_handler(workspace_id)
        return self.save_metadata_value(metadata_handler, "meta_memories", memories)

    async def async_execute(self):
        """Execute the add meta memory operation.

        Adds memory metadata to file storage. Supports both single and multiple modes.
        Duplicates (same memory_type and memory_target) are skipped.
        """
        workspace_id: str = self.workspace_id

        # Load existing memories
        existing_memories: List[dict] = self._load_meta_memories(workspace_id)
        existing_set = {
            (m["memory_type"], m["memory_target"]) for m in existing_memories
        }

        # Build new memories to add based on mode
        new_memories: List[dict] = []
        if self.enable_multiple:
            meta_memories: List[dict] = self.input_dict.get("meta_memories", [])
            for mem in meta_memories:
                memory_type = mem.get("memory_type", "")
                memory_target = mem.get("memory_target", "")
                if memory_type and (memory_type, memory_target) not in existing_set:
                    new_memories.append({
                        "memory_type": memory_type,
                        "memory_target": memory_target,
                    })
                    existing_set.add((memory_type, memory_target))
        else:
            memory_type = self.input_dict.get("memory_type", "")
            memory_target = self.input_dict.get("memory_target", "")
            if memory_type and (memory_type, memory_target) not in existing_set:
                new_memories.append({
                    "memory_type": memory_type,
                    "memory_target": memory_target,
                })

        if not new_memories:
            self.set_output("No new meta memories to add (all entries already exist or invalid).")
            return

        # Merge and save
        all_memories = existing_memories + new_memories
        self._save_meta_memories(workspace_id, all_memories)

        # Format output
        added_str = json.dumps(new_memories, ensure_ascii=False)
        self.set_output(f"Successfully added {len(new_memories)} meta memory entries: {added_str}")

