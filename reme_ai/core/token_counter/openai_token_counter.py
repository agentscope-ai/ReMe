"""Token counting implementation for OpenAI-compatible models."""

import json
from loguru import logger
from .base_token_counter import BaseTokenCounter
from ..context import C
from ..schema import Message, ToolCall


@C.register_token_counter("openai")
class OpenAITokenCounter(BaseTokenCounter):
    """Token counter for OpenAI models using tiktoken."""

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self._encoding = None

    @property
    def encoding(self):
        """Get or initialize the tiktoken encoding for the specified model."""
        if self._encoding is None:
            import tiktoken

            try:
                self._encoding = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                logger.warning(f"Model {self.model_name} not found; falling back to o200k_base.")
                self._encoding = tiktoken.get_encoding("o200k_base")
        return self._encoding

    def count_token(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        **_kwargs,
    ) -> int:
        """Calculate total tokens for a request including messages and tool definitions."""
        enc = self.encoding
        total_tokens = 0

        for msg in messages:
            # Every message has <|start|>{role/name}\n{content}<|end|>\n
            total_tokens += 3  # Base overhead per message
            if msg.content:
                total_tokens += len(enc.encode(msg.content))

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    dump = json.dumps(tc.simple_output_dump(), ensure_ascii=False)
                    total_tokens += len(enc.encode(dump))

        if tools:
            # Account for tool/function definitions if provided
            tool_json = json.dumps([t.simple_input_dump() for t in tools], ensure_ascii=False)
            total_tokens += len(enc.encode(tool_json))

        total_tokens += 3  # Every reply is primed with <|start|>assistant<|message|>
        return total_tokens
