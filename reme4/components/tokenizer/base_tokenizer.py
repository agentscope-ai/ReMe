"""Abstract base class for tokenizers."""

from abc import abstractmethod
from pathlib import Path

import aiofiles

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum


class BaseTokenizer(BaseComponent):
    """Tokenizer base class with shared stopword loading and post-processing.

    Subclasses implement raw tokenization via `_tokenize_one`; lowercasing and
    stopword filtering are handled here so every backend behaves consistently.
    """

    component_type = ComponentEnum.TOKENIZER
    DEFAULT_STOPWORDS_PATH = Path(__file__).parent / "stopwords"

    def __init__(
        self,
        stopwords_path: str | Path | None = None,
        filter_stopwords: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.stopwords_path = Path(stopwords_path) if stopwords_path else self.DEFAULT_STOPWORDS_PATH
        self.filter_stopwords = filter_stopwords
        self._stopwords: set[str] = set()

    async def _start(self) -> None:
        # A missing file is non-fatal: tokenizers still work, just without filtering.
        if not self.stopwords_path.exists():
            self.logger.warning(f"Stopwords file not found: {self.stopwords_path}")
            return
        async with aiofiles.open(self.stopwords_path, encoding="utf-8") as f:
            content = await f.read()
        self._stopwords = {line.strip().lower() for line in content.splitlines() if line.strip()}
        self.logger.info(f"Loaded {len(self._stopwords)} stopwords from {self.stopwords_path}")

    async def _close(self) -> None:
        self._stopwords.clear()

    @property
    def stopwords(self) -> set[str]:
        """Loaded stopwords (empty set if none were loaded)."""
        return self._stopwords

    def tokenize(self, texts: list[str], lower: bool = True, **kwargs) -> list[list[str]]:
        """Tokenize each text and apply shared post-processing."""
        return [self._postprocess(self._tokenize_one(t, **kwargs), lower) for t in texts]

    def _postprocess(self, tokens: list[str], lower: bool) -> list[str]:
        if lower:
            tokens = [t.lower() for t in tokens]
        if self.filter_stopwords and self._stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]
        return tokens

    @abstractmethod
    def _tokenize_one(self, text: str, **kwargs) -> list[str]:
        """Return raw tokens for one text; lowercasing/filtering happen upstream."""
