"""Token counting utility based on character-type rules."""

import math
import re
from loguru import logger

from ..context import C
from ..schema import Message, ToolCall


@C.register_token_counter("base")
class BaseTokenCounter:
    """A rule-based token counter for Chinese and non-Chinese text."""

    def __init__(self, model_name: str, **kwargs):
        """Initialize with model name and additional parameters."""
        self.model_name = model_name
        self.kwargs = kwargs
        # Matches Chinese characters including extensions
        self._cn_regex = re.compile(r"[\u4e00-\u9fff]")

    def _count_chars(self, text: str) -> tuple[int, int]:
        """Count Chinese and other characters in a string."""
        if not text:
            return 0, 0
        cn_count = len(self._cn_regex.findall(text))
        return cn_count, len(text) - cn_count

    def count_token(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        **_kwargs,
    ) -> int:
        """Calculate total tokens using the 1:2 (CN) and 1:4 (Other) rule."""
        cn_total = 0
        ot_total = 0
        logger.info("Calculating tokens using rule-based estimation.")

        # Extract text from messages
        segments = []
        for msg in messages:
            content = msg.content
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="ignore")
            segments.extend([content, msg.reasoning_content])

        # Extract text from tools
        if tools:
            for tool in tools:
                segments.extend([tool.name, tool.description, tool.arguments])

        # Process all segments
        for text in filter(None, segments):
            cn_chars, ot_chars = self._count_chars(text)
            cn_total += cn_chars
            ot_total += ot_chars

        return math.ceil(cn_total / 2) + math.ceil(ot_total / 4)
