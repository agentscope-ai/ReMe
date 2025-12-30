from loguru import logger

from ..context import C
from ..enumeration import Role
from ..op import BaseOp
from ..schema import Message, ToolCall


@C.register_op()
class ChatLLM(BaseOp):

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
            },
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
        assistant_message = await self.llm.chat(messages=messages)
        logger.info(f"assistant_message={assistant_message.content}")
        self.output = assistant_message.content.strip()
