"""Add history tool"""

import json

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ....core.enumeration import MemoryType
from ....core.schema import ToolCall, MemoryNode, Message
from ....core.utils import format_messages


class AddHistory(BaseMemoryTool):
    """Tool to add historical dialogue to vector store"""

    def __init__(self, **kwargs):
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)

    def _build_tool_call(self) -> ToolCall:
        """Build and return the tool call schema"""
        return ToolCall(
            **{
                "description": "Add original history dialogue.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        )

    async def execute(self):
        """Execute the add history operation"""
        self.context.messages = [Message(**m) if isinstance(m, dict) else m for m in self.context.messages]
        history_content: str = self.context.description + "\n" + format_messages(self.context.messages)
        history_content = history_content.strip()
        history_node = MemoryNode(
            memory_type=MemoryType.HISTORY,
            when_to_use=history_content[:1024],
            content=history_content,
            author=self.author,
            metadata={
                "messages": json.dumps(
                    [m.model_dump(exclude_none=True) for m in self.context.messages],
                    ensure_ascii=False,
                ),
            },
        )
        self.context.history_node = history_node
        logger.info(f"Adding history node: {history_node.model_dump_json(indent=2)}")

        vector_node = history_node.to_vector_node()
        await self.vector_store.delete(vector_node.vector_id)
        await self.vector_store.insert([vector_node])

        return f"Successfully added history: {history_node.memory_id}"
