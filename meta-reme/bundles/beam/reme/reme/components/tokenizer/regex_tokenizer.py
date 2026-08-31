"""Regex tokenizer with Chinese character splitting."""

import re

from .base_tokenizer import BaseTokenizer
from ..component_registry import R


@R.register("regex")
class RegexTokenizer(BaseTokenizer):
    """Regex tokenizer: each CJK char is its own token, non-CJK uses word boundaries.

    Treating CJK characters as individual tokens avoids needing a Chinese
    segmenter while still giving BM25-style indexes useful unigrams.
    """

    WORD_PATTERN = re.compile(r"(?u)\b\w\w+\b")  # non-CJK words, 2+ chars
    CHINESE_PATTERN = re.compile(r"[一-鿿]")

    def _tokenize_one(self, text: str, **kwargs) -> list[str]:
        # Pull CJK chars first, then strip them out so the word regex only sees the rest.
        tokens = self.CHINESE_PATTERN.findall(text)
        tokens.extend(self.WORD_PATTERN.findall(self.CHINESE_PATTERN.sub(" ", text)))
        return tokens
