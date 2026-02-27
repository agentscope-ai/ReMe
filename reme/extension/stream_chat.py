"""Streaming chat for test."""

from loguru import logger

from ..core.enumeration import Role, ChunkEnum
from ..core.op import BaseTool
from ..core.schema import Message, ToolCall


class StreamChat(BaseTool):
    """Streaming chat agent that handles real-time conversation streaming."""

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
        """Execute streaming chat operation with query or messages."""
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

        async for stream_chunk in self.llm.stream_chat(messages):
            if stream_chunk.chunk_type in [ChunkEnum.ANSWER, ChunkEnum.THINK, ChunkEnum.ERROR, ChunkEnum.TOOL]:
                await self.context.add_stream_chunk(stream_chunk)
