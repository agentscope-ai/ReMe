"""Token counting implementation for OpenAI-compatible models."""

import json
from typing import List

from loguru import logger

from .base_token_counter import BaseTokenCounter
from ..context import C
from ..schema import Message, ToolCall


@C.register_token_counter("openai")
class OpenAITokenCounter(BaseTokenCounter):
    """The OpenAI token counting class."""

    def count_token(
        self,
        messages: List[Message],
        tools: List[ToolCall] | None = None,
        **_kwargs,
    ) -> int:
        """Estimate token usage for messages and tool payloads using tiktoken."""
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
            logger.info(f"token count: using model={self.model_name}")

        except KeyError:
            encoding = tiktoken.get_encoding("o200k_base")
            logger.info("token count: using model=o200k_base")

        num_tokens = 0
        # <|im_start|>system\n...<|im_end|>
        for message in messages:
            msg_tokens = 4
            if message.content:
                msg_tokens += len(encoding.encode(message.content))
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_call_content = json.dumps(tool_call.simple_output_dump(), ensure_ascii=False)
                    msg_tokens += len(encoding.encode(tool_call_content))

            num_tokens += msg_tokens

        if tools:
            for tool in tools:
                tool_content = json.dumps(tool.simple_input_dump(), ensure_ascii=False)
                num_tokens += len(encoding.encode(tool_content))
        return num_tokens
