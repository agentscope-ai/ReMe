"""ReMe chat formatter."""

from typing import Any

from agentscope.formatter import OpenAIChatFormatter
from agentscope.token import HuggingFaceTokenCounter

from .utils import _extract_text_from_messages


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
