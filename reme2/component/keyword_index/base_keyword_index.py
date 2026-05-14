"""Abstract base class for keyword index implementations."""

from abc import abstractmethod
from pathlib import Path

from ..base_component import BaseComponent
from ..tokenizer import BaseTokenizer
from ...enumeration import ComponentEnum


class BaseKeywordIndex(BaseComponent):
    """Abstract base class for keyword index implementations."""

    component_type = ComponentEnum.KEYWORD_INDEX

    def __init__(self, tokenizer: str = "default", **kwargs):
        super().__init__(**kwargs)
        from ..tokenizer import RegexTokenizer

        self.tokenizer = self.bind(
            tokenizer,
            BaseTokenizer,
            default_factory=lambda: RegexTokenizer(filter_stopwords=False),
        )
        self.index_path = self.working_path / self.component_type.value
        self.index_path.mkdir(parents=True, exist_ok=True)

    async def _start(self) -> None:
        """Load existing index if available. Tokenizer is injected and started by the owner lifecycle."""
        if self.index_file.exists():
            await self.load()
            self.logger.info(f"Loaded index from {self.index_path}")

    async def _close(self) -> None:
        """Save index and cleanup tokenizer on shutdown."""
        await self.dump()
        self.logger.info(f"Saved index to {self.index_path}")

    @property
    def index_file(self) -> Path:
        """Path to the index pickle file based on tokenizer name."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call start() first.")
        name = type(self.tokenizer).__name__.replace("Tokenizer", "").lower()
        return self.index_path / f"bm25_{name}.pkl"

    def _tokenize(self, text: str) -> list[str]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call start() first.")
        return self.tokenizer.tokenize([text])[0]

    @abstractmethod
    async def add_docs(self, docs_dict: dict[str, str]) -> None:
        """Index or update multiple documents.

        Args:
            docs_dict: Mapping of document ID to document content.
        """

    @abstractmethod
    async def delete_docs(self, doc_ids: list[str]) -> None:
        """Remove documents from the index.

        Args:
            doc_ids: List of document IDs to remove.
        """

    @abstractmethod
    async def retrieve(self, query: str, limit: int = 3) -> dict[str, float]:
        """Search for documents matching the query.

        Args:
            query: Search query string.
            limit: Maximum number of results to return.

        Returns:
            Dict mapping document IDs to scores, sorted by score descending.
        """

    @abstractmethod
    async def dump(self) -> None:
        """Persist index to disk."""

    @abstractmethod
    async def load(self) -> None:
        """Load index from disk."""

    @abstractmethod
    async def clear(self) -> None:
        """Reset the index to empty state."""

    async def reset_index(self, docs_dict: dict[str, str]) -> None:
        """Reset the index and re-add documents."""
        await self.clear()
        await self.add_docs(docs_dict)
        await self.dump()

    async def optimize_index(self) -> None:
        """Optimize index for performance."""
