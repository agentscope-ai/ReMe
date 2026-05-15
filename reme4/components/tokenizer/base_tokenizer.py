"""Abstract base class for tokenizers."""

import aiofiles
from abc import abstractmethod
from pathlib import Path

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum


class BaseTokenizer(BaseComponent):
    """Base tokenizer. Subclasses must implement `tokenize`. Loads stopwords on start."""

    component_type = ComponentEnum.TOKENIZER
    DEFAULT_STOPWORDS_PATH = Path(__file__).parent / "stopwords"

    def __init__(self, stopwords_path: str | Path | None = None, **kwargs):
        super().__init__(**kwargs)
        self.stopwords_path = Path(stopwords_path) if stopwords_path else self.DEFAULT_STOPWORDS_PATH
        self._stopwords: set[str] = set()

    async def _start(self) -> None:
        """Load stopwords from file."""
        if not self.stopwords_path.exists():
            self.logger.warning(f"Stopwords file not found: {self.stopwords_path}")
            return
        async with aiofiles.open(self.stopwords_path, encoding="utf-8") as f:
            content = await f.read()
        self._stopwords = {line.strip().lower() for line in content.splitlines() if line.strip()}
        self.logger.info(f"Loaded {len(self._stopwords)} stopwords from {self.stopwords_path}")

    async def _close(self) -> None:
        """Clear stopwords."""
        self._stopwords.clear()

    @property
    def stopwords(self) -> set[str]:
        """Get the loaded stopwords."""
        return self._stopwords

    @abstractmethod
    def tokenize(self, texts: list[str], **kwargs) -> list[list[str]]:
        """Tokenize a list of texts."""
