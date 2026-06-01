"""Abstract base class for keyword indexes (BM25 and other lexical backends)."""

from abc import abstractmethod

from ..base_component import BaseComponent
from ..tokenizer import BaseTokenizer
from ...enumeration import ComponentEnum


class BaseKeywordIndex(BaseComponent):
    """Common interface for keyword indexes (add / delete / retrieve / clear)."""

    component_type = ComponentEnum.KEYWORD_INDEX

    def __init__(self, tokenizer: str = "default", **kwargs):
        super().__init__(**kwargs)
        from ..tokenizer import RegexTokenizer

        self.tokenizer = self.bind(tokenizer, BaseTokenizer, default_factory=RegexTokenizer)

    async def _start(self) -> None:
        self.component_metadata_path.mkdir(parents=True, exist_ok=True)
        await self.load()

    async def _close(self) -> None:
        await self.dump()

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize a single text into a list of tokens."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call start() first.")
        return self.tokenizer.tokenize([text])[0]

    @abstractmethod
    async def add_docs(self, docs_dict: dict[str, str]) -> None:
        """Add or replace documents keyed by id."""

    @abstractmethod
    async def delete_docs(self, doc_ids: list[str]) -> None:
        """Delete documents by id; missing ids are skipped."""

    @abstractmethod
    async def retrieve(self, query: str, limit: int = 3) -> dict[str, float]:
        """Return top-`limit` doc_id → score for the given query."""

    @abstractmethod
    async def clear(self) -> None:
        """Wipe in-memory state and remove any persisted artifacts."""

    async def reset_index(self, docs_dict: dict[str, str]) -> None:
        """Wipe the index, rebuild it from `docs_dict`, and persist the result."""
        await self.clear()
        await self.add_docs(docs_dict)
        await self.dump()

    async def optimize_index(self) -> None:
        """Compact or rebuild the index. No-op by default; override as needed."""
