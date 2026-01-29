"""Return relevant content tool"""

from loguru import logger

from .base_memory_tool import BaseMemoryTool
from ...core.schema import ToolCall


class ReturnRelevantContent(BaseMemoryTool):
    """Tool to format and return relevant content with message times"""

    def __init__(self, **kwargs):
        kwargs["enable_multiple"] = True
        super().__init__(**kwargs)

    def _build_multiple_tool_call(self) -> ToolCall:
        """Build and return the multiple tool call schema"""
        return ToolCall(
            **{
                "description": "Return formatted relevant content with message times.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "description": "List of relevant content items with message times",
                            "items": {
                                "type": "object",
                                "description": "A content item with message time",
                                "properties": {
                                    "message_time": {
                                        "type": "string",
                                        "description": "The timestamp or time identifier for the message",
                                    },
                                    "relevant_content": {
                                        "type": "string",
                                        "description": "The relevant content for this message",
                                    },
                                },
                                "required": ["message_time", "relevant_content"],
                            },
                        },
                    },
                    "required": ["items"],
                },
            },
        )

    async def execute(self):
        """Execute the tool to format and return relevant content"""
        items = self.context.get("items", [])
        
        if not items:
            output = "No items provided."
            logger.warning(output)
            return output
        
        # Join with newlines
        output = "\n".join([f"[{item['message_time']}] {item['relevant_content']}" for item in items])
        
        logger.info(f"Successfully formatted {len(items)} item(s)")
        return output
