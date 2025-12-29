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

    @staticmethod
    def _is_chinese_char(char: str) -> bool:
        """Check if a character is a Chinese character."""
        return '\u4e00' <= char <= '\u9fff'

    @staticmethod
    def _count_chars_by_language(text: str) -> tuple[int, int]:
        """Count Chinese and non-Chinese characters separately.
        
        Returns:
            tuple: (chinese_chars, other_chars)
        """
        chinese_chars = 0
        other_chars = 0
        
        for char in text:
            if BaseTokenCounter._is_chinese_char(char):
                chinese_chars += 1
            else:
                other_chars += 1
        
        return chinese_chars, other_chars

    def count_token(
            self,
            messages: List[Message],
            tools: List[ToolCall] | None = None,
            **_kwargs,
    ) -> int:
        chinese_chars = 0
        other_chars = 0
        logger.info("token count: using rule")

        for message in messages:
            content = message.content
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="ignore")
            
            if content:
                msg_cn, msg_other = self._count_chars_by_language(content)
                chinese_chars += msg_cn
                other_chars += msg_other

            if message.reasoning_content:
                reasoning_cn, reasoning_other = self._count_chars_by_language(message.reasoning_content)
                chinese_chars += reasoning_cn
                other_chars += reasoning_other

        if tools:
            for tool in tools:
                if tool.name:
                    tool_name_cn, tool_name_other = self._count_chars_by_language(tool.name)
                    chinese_chars += tool_name_cn
                    other_chars += tool_name_other
                if tool.description:
                    tool_desc_cn, tool_desc_other = self._count_chars_by_language(tool.description)
                    chinese_chars += tool_desc_cn
                    other_chars += tool_desc_other
                if tool.arguments:
                    tool_args_cn, tool_args_other = self._count_chars_by_language(tool.arguments)
                    chinese_chars += tool_args_cn
                    other_chars += tool_args_other

        # Chinese: 1/2 (2 chars = 1 token), Others: 1/4 (4 chars = 1 token)
        total_tokens = math.ceil(chinese_chars / 2) + math.ceil(other_chars / 4)
        return total_tokens
