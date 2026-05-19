"""Regex tokenizer with Chinese character splitting."""

import re
from .base_tokenizer import BaseTokenizer
from ..component_registry import R


@R.register("regex")
class RegexTokenizer(BaseTokenizer):
    """Tokenizer using regex: splits Chinese chars individually, extracts non-Chinese words."""

    WORD_PATTERN = re.compile(r"(?u)\b\w\w+\b")  # 2+ char words
    CHINESE_PATTERN = re.compile(r"[一-鿿]")  # single Chinese char

    def __init__(self, filter_stopwords: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.filter_stopwords = filter_stopwords

    def tokenize(self, texts: list[str], lower: bool = True, **kwargs) -> list[list[str]]:
        """Tokenize texts. Extracts Chinese chars, then non-Chinese words from remaining text."""
        result = []
        for text in texts:
            # Extract Chinese chars individually, then non-Chinese words
            tokens = self.CHINESE_PATTERN.findall(text)
            tokens.extend(self.WORD_PATTERN.findall(self.CHINESE_PATTERN.sub(" ", text)))
            if lower:
                tokens = [t.lower() for t in tokens]
            if self.filter_stopwords and self._stopwords:
                tokens = [t for t in tokens if t not in self._stopwords]
            result.append(tokens)
        return result
