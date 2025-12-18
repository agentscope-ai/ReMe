"""Update identity memory operation for saving agent self-cognition memories.

This module provides the UpdateIdentityMemoryOp class for updating identity memories
using file-based storage with CacheHandler.
"""

from ..base_memory_tool_op import BaseMemoryToolOp
from ... import C


@C.register_op()
class UpdateIdentityMemoryOp(BaseMemoryToolOp):
    """Operation for updating identity memories using file-based storage."""

    def __init__(self, **kwargs):
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)

    def build_input_schema(self) -> dict:
        """Build input schema for identity memory update.

        Returns:
            dict: Input schema for updating an identity memory.
        """
        return {
            "identity_memory": {
                "type": "string",
                "description": self.get_prompt("identity_memory"),
                "required": True,
            },
        }

    def _save_identity_memory(self, identity_memory: str, workspace_id: str) -> bool:
        """Save identity memory to file.

        Args:
            identity_memory: The memory content for identity.
            workspace_id: The workspace ID.

        Returns:
            bool: Whether the save was successful.
        """
        # Get metadata handler for the workspace identity directory
        metadata_handler = self.get_metadata_handler(f"{workspace_id}/identity")
        return self.save_metadata_value(metadata_handler, "identity_memory", identity_memory)

    async def async_execute(self):
        """Execute the update identity memory operation.

        Updates identity memory to file storage.
        """
        workspace_id: str = self.workspace_id

        identity_memory = self.input_dict.get("identity_memory", "")

        if not identity_memory:
            self.set_output("No valid identity memory provided for update.")
            return

        self._save_identity_memory(identity_memory, workspace_id)
        self.set_output(f"Successfully updated identity memory in workspace={workspace_id}.")

