"""Jieba tokenizer for Chinese text segmentation."""

from typing import Callable

from .base_tokenizer import BaseTokenizer
from ..component_registry import R


@R.register("jieba")
class JiebaTokenizer(BaseTokenizer):
    """Tokenizer backed by jieba for Chinese word segmentation.

    `backend` selects the underlying implementation:
      - "rjieba": Rust binding of jieba-rs, ~10-30x faster than pure Python (default).
      - "jieba":  Original pure-Python jieba, slowest but the reference.
    """

    SUPPORTED_BACKENDS = ("rjieba", "jieba")

    def __init__(self, backend: str = "rjieba", **kwargs):
        super().__init__(**kwargs)
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown jieba backend {backend!r}; expected one of {self.SUPPORTED_BACKENDS}",
            )
        self.backend = backend
        self._cut: Callable[[str], list[str]] | None = None

    async def _start(self) -> None:
        await super()._start()
        # Resolve the backend once at startup so per-call overhead is just one attribute lookup.
        if self.backend == "rjieba":
            import rjieba

            self._cut = rjieba.cut
        else:
            import jieba

            self._cut = jieba.cut
        self.logger.info(f"JiebaTokenizer using backend: {self.backend}")

    def _tokenize_one(self, text: str, **kwargs) -> list[str]:
        return list(self._cut(text))
