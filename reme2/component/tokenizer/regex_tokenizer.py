"""Regex tokenizer implementation."""

import re

from .base_tokenizer import BaseTokenizer
from ..component_registry import R


@R.register("regex")
class RegexTokenizer(BaseTokenizer):
    """Tokenizer using regex for word segmentation, with Chinese character splitting."""

    # Match words with word boundaries (2+ characters)
    WORD_PATTERN = re.compile(r"(?u)\b\w\w+\b")
    # Match single Chinese character
    CHINESE_PATTERN = re.compile(r"[一-鿿]")

    def __init__(self, filter_stopwords: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.filter_stopwords = filter_stopwords

    def tokenize(self, texts: list[str], lower: bool = True, **kwargs) -> list[list[str]]:
        """Tokenize texts using regex pattern.

        Strategy:
        1. Extract all Chinese characters (split by character)
        2. Replace Chinese with spaces in original text
        3. Extract non-Chinese words with word boundaries

        Args:
            texts: List of texts to tokenize.
            lower: Whether to lowercase tokens.

        Returns:
            List of token lists. Note: tokens are unordered (Chinese chars first, then words).
        """
        result = []
        for text in texts:
            tokens = []

            # Extract all Chinese characters
            tokens.extend(self.CHINESE_PATTERN.findall(text))

            # Replace Chinese with spaces, then extract words
            text_without_chinese = self.CHINESE_PATTERN.sub(" ", text)
            tokens.extend(self.WORD_PATTERN.findall(text_without_chinese))

            if lower:
                tokens = [t.lower() for t in tokens]

            if self.filter_stopwords and self._stopwords:
                tokens = [t for t in tokens if t not in self._stopwords]

            result.append(tokens)
        return result
