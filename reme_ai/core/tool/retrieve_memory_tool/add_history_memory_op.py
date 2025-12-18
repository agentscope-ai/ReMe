"""Add history memory operation for inserting history memory into the vector store.

This module provides the AddHistoryMemoryOp class for adding history memory
from message lists to the vector store.
"""

from typing import Any, Dict, List

from ..base_memory_tool_op import BaseMemoryToolOp
from ... import C
from ...enumeration import MemoryType
from ...schema import Message, MemoryNode


@C.register_op()
class AddHistoryMemoryOp(BaseMemoryToolOp):
    """Operation for adding history memory to the vector store.

    This operation takes a list of messages, formats them using Message.format_message,
    and stores them as history memory in the vector store.
    """

    def __init__(self, **kwargs):
        kwargs["enable_multiple"] = True
        super().__init__(**kwargs)

    def build_multiple_input_schema(self) -> dict:
        """Build input schema for history memory addition.

        Returns:
            dict: Input schema for adding history memory from messages.
        """
        return {
            "messages": {
                "type": "array",
                "description": self.get_prompt("messages"),
                "required": True,
                "items": {
                    "type": "object",
                    "description": "Message object with role, content, and optional fields",
                },
            }
        }

    @staticmethod
    def _format_messages(messages: List[Dict[str, Any]]) -> str:
        """Format messages into a single text string.

        Args:
            messages: List of message dictionaries.

        Returns:
            str: Formatted text combining all messages.
        """
        formatted_lines: List[str] = []
        for i, msg_dict in enumerate(messages):
            message = Message(**msg_dict)
            formatted_line = message.format_message(
                i=i,
                add_time_created=True,
                use_name_first=True,
                add_reasoning_content=True,
                add_tool_calls=True,
            )
            formatted_lines.append(formatted_line)
        return "\n\n".join(formatted_lines)

    async def async_execute(self):
        """Execute the add history memory operation.

        Takes messages from input, formats them using Message.format_message,
        creates a MemoryNode with memory_type=HISTORY, and stores it in the vector store.

        Returns the created memory.
        """
        workspace_id: str = self.workspace_id
        messages: List[Dict[str, Any]] = self.input_dict.get("messages", [])

        if not messages:
            self.set_output("No messages provided for history memory addition.")
            return

        # Format messages into memory_value
        memory_value: str = self._format_messages(messages)

        if not memory_value:
            self.set_output("Formatted message content is empty.")
            return

        memory_node = MemoryNode(
            workspace_id=workspace_id,
            memory_type=MemoryType.HISTORY,
            content=memory_value,
            author=self.author,
        )

        await self.vector_store.async_delete(node_ids=[memory_node.memory_id], workspace_id=workspace_id)

        vector_node = memory_node.to_vector_node()
        await self.vector_store.async_insert(nodes=[vector_node], workspace_id=workspace_id)

        self.set_output(memory_node)
