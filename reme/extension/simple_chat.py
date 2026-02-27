"""Simple chat for test."""

from loguru import logger

from ..core.enumeration import Role
from ..core.op import BaseTool
from ..core.schema import Message, ToolCall


class SimpleChat(BaseTool):
    """Simple chat agent that handles non-streaming conversations."""

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": "simple chat agent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "query",
                        },
                        "messages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {
                                        "type": "string",
                                        "description": "role",
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "content",
                                    },
                                },
                                "required": ["role", "content"],
                            },
                        },
                    },
                    "required": [],
                },
            },
        )

    async def execute(self):
        if "query" in self.context:
            messages = [
                Message(role=Role.SYSTEM, content="You are a helpful assistant."),
                Message(role=Role.USER, content=self.context.query),
            ]
        elif "messages" in self.context:
            messages = [Message(**m) if isinstance(m, dict) else m for m in self.context.messages if m]
        else:
            raise ValueError("query or messages must be provided!")
        logger.info(f"messages={messages}")
        assistant_message = await self.llm.chat(messages=messages)
        logger.info(f"assistant_message={assistant_message.simple_dump()}")
        return assistant_message.content
