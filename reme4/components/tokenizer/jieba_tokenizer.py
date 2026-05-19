"""Jieba tokenizer for Chinese text segmentation."""

from .base_tokenizer import BaseTokenizer
from ..component_registry import R


@R.register("jieba")
class JiebaTokenizer(BaseTokenizer):
    """Tokenizer using jieba for Chinese text segmentation."""

    def __init__(self, filter_stopwords: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.filter_stopwords = filter_stopwords

    def tokenize(self, texts: list[str], lower: bool = True, **kwargs) -> list[list[str]]:
        """Tokenize texts using jieba."""
        import jieba

        result = []
        for text in texts:
            tokens = jieba.cut(text)
            if lower:
                tokens = [x.lower() for x in tokens]
            if self.filter_stopwords and self._stopwords:
                tokens = [t for t in tokens if t not in self._stopwords]
            result.append(tokens)
        return result
