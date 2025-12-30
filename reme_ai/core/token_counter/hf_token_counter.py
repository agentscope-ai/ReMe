"""Token counting utilities for HuggingFace chat models."""

import os
from typing import Any, List

from loguru import logger

from .base_token_counter import BaseTokenCounter
from ..context import C
from ..schema import Message, ToolCall


@C.register_token_counter("hf")
class HFTokenCounter(BaseTokenCounter):
    """Estimate token usage with `transformers.AutoTokenizer.apply_chat_template`."""

    def __init__(
        self,
        model_name: str,
        *,
        use_fast: bool = False,
        trust_remote_code: bool = False,
        use_mirror: bool = True,
        **kwargs,
    ):
        """Initialize HuggingFace token counter.

        Args:
            model_name: Name or path of the HuggingFace model.
            use_fast: Pass-through flag for ``AutoTokenizer.from_pretrained``.
            trust_remote_code: Whether to trust remote code when loading tokenizers.
            use_mirror: If ``True``, set ``HF_ENDPOINT`` to the HuggingFace mirror.
            **kwargs: Extra keyword arguments forwarded to ``BaseTokenCounter``.
        """
        super().__init__(model_name=model_name, **kwargs)

        self.use_fast: bool = use_fast
        self.trust_remote_code: bool = trust_remote_code
        self.use_mirror: bool = use_mirror

        # Lazily initialized tokenizer instance
        self._tokenizer: Any | None = None

    def _ensure_tokenizer(self) -> Any:
        """Lazily create and cache the tokenizer instance."""
        if self._tokenizer is not None:
            return self._tokenizer

        if self.use_mirror:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        from transformers import AutoTokenizer

        logger.info(f"token count: using model={self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=self.use_fast,
            trust_remote_code=self.trust_remote_code,
            **self.kwargs,
        )

        if tokenizer.chat_template is None:
            raise ValueError(f"The tokenizer for model {self.model_name} is missing a chat template.")

        self._tokenizer = tokenizer
        return tokenizer

    @staticmethod
    def _serialize_messages(messages: List[Message]) -> list[dict]:
        """Convert internal `Message` objects to simple dictionaries."""
        return [msg.simple_dump() for msg in messages]

    @staticmethod
    def _serialize_tools(tools: List[ToolCall] | None) -> list[dict] | None:
        """Convert tool schemas into JSON-serializable dictionaries."""
        if not tools:
            return None
        return [tool.simple_input_dump() for tool in tools]

    def count_token(
        self,
        messages: List[Message],
        tools: List[ToolCall] | None = None,
        **kwargs,
    ) -> int:
        """Return the token length computed by the HuggingFace tokenizer."""
        tokenizer = self._ensure_tokenizer()
        serialized_messages = self._serialize_messages(messages)
        serialized_tools = self._serialize_tools(tools)

        tokenized = tokenizer.apply_chat_template(
            serialized_messages,
            add_generation_prompt=kwargs.pop("add_generation_prompt", False),
            tokenize=True,
            return_tensors="np",
            tools=serialized_tools,
            **kwargs,
        )[0]
        return int(tokenized.size if hasattr(tokenized, "size") else len(tokenized))
