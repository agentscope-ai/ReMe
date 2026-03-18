"""Rule-based token counter for fast estimation without loading tokenizer."""

from typing import Any

from agentscope.token import HuggingFaceTokenCounter


class RuleTokenCounter(HuggingFaceTokenCounter):
    """Lightweight token counter using rule-based estimation only.

    This class provides fast token estimation without loading any tokenizer,
    useful when exact token counts are not critical or for quick approximations.

    Attributes:
        token_count_estimate_divisor: Divisor for token estimation.
    """

    def __init__(
        self,
        token_count_estimate_divisor: float = 3.75,
        **_kwargs,
    ):
        """Initialize the rule-based token counter.

        Args:
            token_count_estimate_divisor: Divisor for estimating tokens.
                Defaults to 3.75 (approximately 4 characters per token).
            **kwargs: Additional keyword arguments (ignored).
        """
        self.token_count_estimate_divisor = token_count_estimate_divisor
        # Skip tokenizer initialization from parent
        self._tokenizer_available = False

    async def count(
        self,
        messages: list[dict],
        _tools: list[dict] | None = None,
        text: str | None = None,
        **_kwargs: Any,
    ) -> int:
        """Count tokens using rule-based estimation.

        Args:
            messages: List of message dictionaries in chat format.
            _tools: Optional list of tool definitions (ignored).
            text: Optional text string to count tokens directly.
            **_kwargs: Additional keyword arguments (ignored).

        Returns:
            The estimated number of tokens.
        """
        if text:
            return self.estimate_tokens(text)

        # Estimate from messages
        total_text = ""
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_text += content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        total_text += part["text"]
        return self.estimate_tokens(total_text)

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text string.

        Uses character-based estimation with the configured divisor.

        Args:
            text: The text string to estimate tokens for.

        Returns:
            The estimated number of tokens in the text string.
        """
        return int(len(text.encode("utf-8")) / self.token_count_estimate_divisor + 0.5)
