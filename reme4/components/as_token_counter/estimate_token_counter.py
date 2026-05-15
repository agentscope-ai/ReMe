"""Character-based token-count estimator."""

from agentscope.token import TokenCounterBase


class EstimatedTokenCounter(TokenCounterBase):
    """Approximate token count as ``encoded_byte_len / divisor``.

    Cheap proxy when exact counts aren't needed; use the model's real
    tokenizer for accuracy.
    """

    def __init__(self, estimate_divisor: float = 4, encoding: str = "utf-8"):
        if estimate_divisor <= 0:
            raise ValueError("estimate_divisor must be positive")
        self.estimate_divisor: float = estimate_divisor
        self.encoding: str = encoding

    async def count(self, text: str, **kwargs) -> int:
        """Estimated token count for ``text``."""
        return int(len(text.encode(self.encoding)) / self.estimate_divisor + 0.5)
