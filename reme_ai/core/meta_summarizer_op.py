from typing import List

from loguru import logger

from . import BaseAsyncToolOp, C
from .schema import ToolCall, Message
from .utils import format_messages


@C.register_op()
class BaseSummarizerOp(BaseAsyncToolOp):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": self.get_prompt("tool_description"),
                "input_schema": {
                    "messages": {
                        "type": "array",
                        "description": "messages",
                        "required": False,
                    },
                },
            }
        )

    async def async_execute(self):
        logger.info("start")

        messages: List[Message] = [Message(**x) for x in self.input_dict["messages"]]

        print(format_messages(messages))
