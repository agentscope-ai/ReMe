"""Exact model-token budgets, independent of BM25 tokenization."""

from collections.abc import Callable

from reme.utils.qwen_token_tools import DEFAULT_TOKENIZER, QwenTokenTools


class TokenBudget:
    """Reject oversized fields; split original text without loss or truncation."""

    def __init__(self, tokenizer_source: str = DEFAULT_TOKENIZER, counter: Callable | None = None, **limits):
        self.source = tokenizer_source
        self.tools = QwenTokenTools(tokenizer_source)
        self.counter = counter
        self.limits = {
            "segment": 98304,
            "summary": 8192,
            "description": 128,
            "when-to-use": 512,
            "content": 2048,
            "context": 262144,
            "reserve": 4096,
            **limits,
        }
        if any(type(value) is not int or value < 1 for value in self.limits.values()):
            raise ValueError("Token budgets must be positive integers")

    def count(self, text: str) -> int:
        """Require exact tokenization rather than silently using the core estimate fallback."""
        if self.counter is not None:
            return self.counter(text)
        if self.tools.tokenizer is None:
            raise RuntimeError(
                "Exact tokenizer unavailable; install reme-ai[qwen-tokenizer] and configure tokenizer_source",
            )
        return self.tools.count(text)

    def check(self, field: str, text: str) -> None:
        """Allow 10% summary/experience slack internally; keep advertised targets unchanged."""
        count = self.count(text)
        limit = self.limits[field]
        exceeds = (
            count * 10 > limit * 11 if field in {"summary", "description", "when-to-use", "content"} else count > limit
        )
        if exceeds:
            raise ValueError(f"{field} has {count} tokens; maximum is {self.limits[field]}. Rewrite it more concisely.")

    def prefix(self, text: str, limit: int, *, start: int = 0) -> int:
        """Find a bounded prefix at start, probing progressively larger windows."""
        available = len(text) - start
        if not available:
            return 0
        low, high = 0, min(available, limit)
        while True:
            count = self.count(text[start : start + high])
            if count > limit:
                high -= 1
                break
            low = high
            if high == available:
                return high
            # An exact fit commonly occurs in logs. One extra character can
            # establish the boundary without another full-sized search window.
            if count == limit and self.count(text[start : start + high + 1]) > limit:
                return high
            high = min(available, high * 2)
        while low < high:
            middle = (low + high + 1) // 2
            if self.count(text[start : start + middle]) <= limit:
                low = middle
            else:
                high = middle - 1
        if not low:
            raise ValueError("Token budget cannot hold one character")
        return low
