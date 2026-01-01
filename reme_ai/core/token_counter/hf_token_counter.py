"""HuggingFace token counting utilities."""

import os

from loguru import logger

from .base_token_counter import BaseTokenCounter
from ..context import C
from ..schema import Message, ToolCall


@C.register_token_counter("hf")
class HFTokenCounter(BaseTokenCounter):
    """Token counter using transformers.AutoTokenizer.apply_chat_template."""

    def __init__(
        self,
        model_name: str,
        use_fast: bool = False,
        trust_remote_code: bool = False,
        use_mirror: bool = True,
        **kwargs,
    ):
        """Initialize the counter with model config and lazy tokenizer loading."""
        super().__init__(model_name=model_name, **kwargs)
        self.use_fast = use_fast
        self.trust_remote_code = trust_remote_code
        self.use_mirror = use_mirror
        self._tokenizer = None

    def _ensure_tokenizer(self):
        """Initialize and cache the HuggingFace tokenizer safely."""
        if self._tokenizer:
            return self._tokenizer

        if self.use_mirror:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        try:
            from transformers import AutoTokenizer

            logger.info("Initializing HuggingFace tokenizer for {}", self.model_name)

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=self.use_fast,
                trust_remote_code=self.trust_remote_code,
                **self.kwargs,
            )

            if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
                raise ValueError(f"Model {self.model_name} lacks a chat template.")

            self._tokenizer = tokenizer
            return tokenizer
        except Exception as e:
            logger.error("Failed to load tokenizer {}: {}", self.model_name, e)
            raise

    def count_token(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        **kwargs,
    ) -> int:
        """Calculate total tokens for messages and tools using the chat template."""
        tokenizer = self._ensure_tokenizer()

        # Serialize inputs for the template
        formatted_msgs = [m.simple_dump() for m in messages]
        formatted_tools = [t.simple_input_dump() for t in tools] if tools else None

        # Setting tokenize=True and leaving return_tensors=None returns a List[int]
        tokens = tokenizer.apply_chat_template(
            formatted_msgs,
            tools=formatted_tools,
            add_generation_prompt=kwargs.pop("add_generation_prompt", False),
            tokenize=True,
            **kwargs,
        )

        return len(tokens)
