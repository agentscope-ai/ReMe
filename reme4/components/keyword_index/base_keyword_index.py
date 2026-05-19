"""Abstract base class for keyword index implementations."""

from abc import abstractmethod
from pathlib import Path

from ..base_component import BaseComponent
from ..tokenizer import BaseTokenizer
from ...enumeration import ComponentEnum


class BaseKeywordIndex(BaseComponent):
    """Abstract base class for keyword index implementations."""

    component_type = ComponentEnum.KEYWORD_INDEX

    def __init__(self, tokenizer: str = "default", index_version: str = "v1", **kwargs):
        super().__init__(**kwargs)
        from ..tokenizer import RegexTokenizer

        self.tokenizer = self.bind(tokenizer, BaseTokenizer, default_factory=RegexTokenizer)
        self.index_version = index_version
        self.index_path = self.working_metadata_path / self.component_type.value
        self.index_path.mkdir(parents=True, exist_ok=True)

    async def _start(self) -> None:
        """Load existing index from disk if available."""
        await self.load()

    async def _close(self) -> None:
        """Save index to disk on shutdown."""
        await self.dump()

    @property
    def index_file(self) -> Path:
        """Return the pickle file path derived from tokenizer name."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call start() first.")
        name = type(self.tokenizer).__name__.replace("Tokenizer", "").lower()
        return self.index_path / f"bm25_{name}_{self.index_version}.pkl"

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize a text string into tokens."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call start() first.")
        return self.tokenizer.tokenize([text])[0]

    @abstractmethod
    async def add_docs(self, docs_dict: dict[str, str]) -> None:
        """Index or update documents. Mapping of doc_id to content."""

    @abstractmethod
    async def delete_docs(self, doc_ids: list[str]) -> None:
        """Remove documents by their IDs."""

    @abstractmethod
    async def retrieve(self, query: str, limit: int = 3) -> dict[str, float]:
        """Search documents. Returns {doc_id: score} sorted descending."""

    @abstractmethod
    async def clear(self) -> None:
        """Reset index to empty state."""

    async def reset_index(self, docs_dict: dict[str, str]) -> None:
        """Clear index, re-add all documents, and persist."""
        await self.clear()
        await self.add_docs(docs_dict)
        await self.dump()

    async def optimize_index(self) -> None:
        """Optimize index for performance. Override in subclass if needed."""
