"""Test utilities for copaw tests."""

import os
from pathlib import Path
from typing import Any

from loguru import logger

_token_counter = None


def get_token_counter():
    """Get or initialize the global token counter instance.

    Returns:
        TokenCounterBase: The token counter instance for Qwen models.

    Raises:
        RuntimeError: If token counter initialization fails.
    """
    global _token_counter
    if _token_counter is None:
        from agentscope.token import HuggingFaceTokenCounter

        # Use Qwen tokenizer for DashScope models
        # Qwen3 series uses the same tokenizer as Qwen2.5

        # Try local tokenizer first, fall back to online if not found
        local_tokenizer_path = Path(__file__).parent.parent.parent / "tokenizer"

        if local_tokenizer_path.exists() and (local_tokenizer_path / "tokenizer.json").exists():
            tokenizer_path = str(local_tokenizer_path)
            logger.info(f"Using local Qwen tokenizer from {tokenizer_path}")
        else:
            tokenizer_path = "Qwen/Qwen2.5-7B-Instruct"
            logger.info(
                "Local tokenizer not found, downloading from HuggingFace",
            )

        _token_counter = HuggingFaceTokenCounter(
            pretrained_model_name_or_path=tokenizer_path,
            use_mirror=True,  # Use HF mirror for users in China
            use_fast=True,
            trust_remote_code=True,
        )
        logger.debug("Token counter initialized with Qwen tokenizer")
    return _token_counter


def get_dash_chat_model(model_name: str = "qwen3.5-plus"):
    """Get DashScope chat model instance."""
    from agentscope.model import OpenAIChatModel
    from reme.core.utils import load_env

    load_env()
    return OpenAIChatModel(
        api_key=os.environ["REME_LLM_API_KEY"],
        client_kwargs={"base_url": os.environ["REME_LLM_BASE_URL"]},
        model_name=model_name,
    )


def get_formatter():
    """Get formatter instance."""
    from agentscope.formatter import OpenAIChatFormatter
    from agentscope.token import HuggingFaceTokenCounter
    from reme.memory.file_based_copaw.utils import _extract_text_from_messages

    class ReMeChatFormatter(OpenAIChatFormatter):
        """ReMe chat formatter class."""

        async def _count(self, msgs: list[dict[str, Any]]) -> int | None:
            """Count the number of tokens in the input messages. If token counter
            is not provided, `None` will be returned.

            Args:
                msgs (`list[Msg]`):
                    The input messages to count tokens for.
            """
            if self.token_counter is None:
                return None

            assert isinstance(self.token_counter, HuggingFaceTokenCounter)
            text = _extract_text_from_messages(msgs)
            token_ids = self.token_counter.tokenizer.encode(text)
            token_count = len(token_ids)
            return token_count

    return ReMeChatFormatter(token_counter=get_token_counter())
