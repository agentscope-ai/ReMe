"""Abstract base class for tokenizers."""

import aiofiles
from abc import abstractmethod
from pathlib import Path

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum


class BaseTokenizer(BaseComponent):
    """Abstract base class for tokenizers.

    Subclasses must implement the `tokenize` method.
    The `_start` method loads stopwords, and `_close` clears them.
    """

    component_type = ComponentEnum.TOKENIZER

    DEFAULT_STOPWORDS_PATH = Path(__file__).parent / "stopwords"

    def __init__(self, stopwords_path: str | Path | None = None, **kwargs):
        super().__init__(**kwargs)
        self.stopwords_path = Path(stopwords_path) if stopwords_path else self.DEFAULT_STOPWORDS_PATH
        self._stopwords: set[str] = set()

    async def _start(self) -> None:
        """Load stopwords from file."""
        if self.stopwords_path.exists():
            async with aiofiles.open(self.stopwords_path, encoding="utf-8") as f:
                content = await f.read()
            self._stopwords = set(line.strip().lower() for line in content.splitlines() if line.strip())
            self.logger.info(f"Loaded {len(self._stopwords)} stopwords from {self.stopwords_path}")
        else:
            self.logger.warning(f"Stopwords file not found: {self.stopwords_path}")

    async def _close(self) -> None:
        """Clear stopwords."""
        self._stopwords.clear()
        self.logger.info("Cleared stopwords")

    @property
    def stopwords(self) -> set[str]:
        """Get the loaded stopwords."""
        return self._stopwords

    @abstractmethod
    def tokenize(self, texts: list[str], **kwargs) -> list[list[str]]:
        """Tokenize a list of texts."""
