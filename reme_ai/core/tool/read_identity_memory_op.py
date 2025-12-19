"""Read identity memory operation for retrieving agent self-cognition memories.

This module provides the ReadIdentityMemoryOp class for reading identity memories
using file-based storage with CacheHandler.
"""

from .base_memory_tool_op import BaseMemoryToolOp
from .. import C


@C.register_op()
class ReadIdentityMemoryOp(BaseMemoryToolOp):
    """Operation for reading identity memories using file-based storage."""

    def __init__(self, **kwargs):
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)

    def _load_identity_memory(self, workspace_id: str) -> str:
        """Load identity memory from file.

        Args:
            workspace_id: The workspace ID.

        Returns:
            str: The identity memory content, or empty string if not found.
        """
        metadata_handler = self.get_metadata_handler(workspace_id)
        result = self.load_metadata_value(metadata_handler, "identity_memory")
        return result if result is not None else ""

    async def async_execute(self):
        """Execute the read identity memory operation.

        Reads identity memory from file storage.
        """
        workspace_id: str = self.workspace_id

        identity_memory = self._load_identity_memory(workspace_id)
        if identity_memory:
            self.set_output(f"Identity memory:\n{identity_memory}")
        else:
            self.set_output(f"No identity memory found in workspace={workspace_id}.")
