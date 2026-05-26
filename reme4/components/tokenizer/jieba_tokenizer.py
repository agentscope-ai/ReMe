"""Jieba tokenizer for Chinese text segmentation."""

from .base_tokenizer import BaseTokenizer
from ..component_registry import R


@R.register("jieba")
class JiebaTokenizer(BaseTokenizer):
    """Tokenizer backed by jieba for Chinese word segmentation."""

    def _tokenize_one(self, text: str, **kwargs) -> list[str]:
        # Lazy import: jieba startup cost is non-trivial and only paid when used.
        import jieba

        return list(jieba.cut(text))
