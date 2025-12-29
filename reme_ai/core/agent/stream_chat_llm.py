from loguru import logger

from ..context import C
from ..enumeration import ChunkEnum, Role
from ..op import BaseOp
from ..schema import Message, ToolCall


@C.register_op()
class StreamChatLLM(BaseOp):

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": "chat with llm",
                "input_schema": {
                    "query": {
                        "type": "string",
                        "description": "search query",
                        "required": False,
                    },
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                        },
                    },
                },
            }
        )

    async def execute(self):
        if self.context.get("query", ""):
            messages = [
                Message(role=Role.SYSTEM, content="You are a helpful assistant."),
                Message(role=Role.USER, content=self.context.query),
            ]

        elif self.context.get("messages", []):
            messages = self.context.messages
            messages = [Message(**m) for m in messages if isinstance(m, dict)]

        else:
            raise NotImplementedError

        logger.info(f"messages={messages}")

        async for stream_chunk in self.llm.stream_chat(messages):
            if stream_chunk.chunk_type in [ChunkEnum.ANSWER, ChunkEnum.THINK, ChunkEnum.ERROR, ChunkEnum.TOOL]:
                await self.context.add_stream_chunk(stream_chunk)
