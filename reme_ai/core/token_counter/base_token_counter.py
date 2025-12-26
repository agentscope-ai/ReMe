import math
from typing import List

from loguru import logger

from ..context import C
from ..schema import Message, ToolCall


@C.register_token_counter("base")
class BaseTokenCounter:

    def __init__(self, model_name: str, **kwargs):
        self.model_name: str = model_name
        self.kwargs: dict = kwargs

    def count_token(
            self,
            messages: List[Message],
            tools: List[ToolCall] | None = None,
            **_kwargs,
    ) -> int:
        total_chars = 0
        logger.info("token count: using rule")

        for message in messages:
            content = message.content
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="ignore")
            total_chars += len(content)

            if message.reasoning_content:
                total_chars += len(message.reasoning_content)

        if tools:
            for tool in tools:
                total_chars += len(tool.name)
                total_chars += len(tool.description)
                total_chars += len(tool.arguments)

        return math.ceil(total_chars / 4) if total_chars else 0
