from typing import Any

from agentscope.token import TokenCounterBase


class EstimatedTokenCounter(TokenCounterBase):

    def __init__(self, estimate_divisor: float = 4):
        if estimate_divisor == 0:
            raise ValueError("estimate_divisor cannot be zero")
        self.estimate_divisor: float = estimate_divisor

    async def count(
            self,
            messages: list[dict],
            text: str | None = None,
            **kwargs: Any,
    ) -> int:
        if not text:
            return 0
        else:
            return int(len(text.encode("utf-8")) / self.estimate_divisor + 0.5)
