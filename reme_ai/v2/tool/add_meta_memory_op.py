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
                "enum": [MemoryType.PERSONAL.value, MemoryType.PROCEDURAL.value],
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
                            "enum": [MemoryType.PERSONAL.value, MemoryType.PROCEDURAL.value],
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

    def _load_meta_memories(self) -> List[dict]:
        result = self.metadata_handler.load("meta_memories")
        return result if result is not None else []

    def _save_meta_memories(self, memories: List[dict]) -> bool:
        return self.metadata_handler.save("meta_memories", memories)

    async def async_execute(self):
        """Execute the add meta memory operation.

        Adds memory metadata to file storage. Supports both single and multiple modes.
        Duplicates (same memory_type and memory_target) are skipped.
        """
        existing_memories: List[dict] = self._load_meta_memories()
        existing_set = {(m["memory_type"], m["memory_target"]) for m in existing_memories}

        # Build new memories to add based on mode
        new_memories: List[dict] = []
        if self.enable_multiple:
            meta_memories: List[dict] = self.context.get("meta_memories", [])
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
            memory_type = self.context.get("memory_type", "")
            memory_target = self.context.get("memory_target", "")
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
        self._save_meta_memories(all_memories)

        # Format output
        added_str = json.dumps(new_memories, ensure_ascii=False)
        self.set_output(f"Successfully added {len(new_memories)} meta memory entries: {added_str}")
