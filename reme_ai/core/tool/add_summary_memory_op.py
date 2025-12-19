"""Add summary memory operation for inserting summarized memories into the vector store.

This module provides the AddSummaryMemoryOp class for adding summary memories
to the vector store. The LLM summarizes the context and calls this tool with
the summarized content.
"""

from .add_memory_op import AddMemoryOp
from .. import C


@C.register_op()
class AddSummaryMemoryOp(AddMemoryOp):
    """Operation for adding summary memories to the vector store.

    This operation only supports single memory addition mode (enable_multiple=False).
    The external LLM summarizes the provided context and calls this tool with
    the summary_memory parameter to store the summarized content.
    """

    def __init__(self, **kwargs):
        # Force enable_multiple to False
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)

    def build_input_schema(self) -> dict:
        """Build input schema for summary memory addition.

        Returns:
            dict: Input schema for adding a summary memory.
        """
        schema = {
            "summary_memory": {
                "type": "string",
                "description": self.get_prompt("summary_memory"),
                "required": True,
            },
        }
        if self.add_metadata:
            schema["metadata"] = {
                "type": "object",
                "description": self.get_prompt("metadata"),
                "required": False,
            }
        return schema

    async def async_execute(self):
        """Execute the add summary memory operation.

        Adds a summary memory to the vector store. The external LLM should
        summarize the context before calling this tool with the summary_memory parameter.
        """
        # Map summary_memory to memory_content for parent class
        summary_memory = self.input_dict.get("summary_memory", "")
        self.input_dict["memory_content"] = summary_memory

        # Call parent's async_execute
        await super().async_execute()
