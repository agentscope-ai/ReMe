"""Abstract base class for keyword indexes (BM25 and other lexical backends)."""

from abc import abstractmethod
from collections.abc import Collection, Set

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
        if self.is_started:
            await self.dump()

    @property
    def document_ids(self) -> Set[str]:
        """Return live document IDs without materializing per-document metadata."""
        raise NotImplementedError(f"{type(self).__name__} does not expose live document IDs")

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize a single text into a list of tokens."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call start() first.")
        return self.tokenizer.tokenize([text])[0]

    def is_indexable(self, text: str) -> bool:
        """Return whether this backend can represent *text* as a document."""
        return bool(text)

    @abstractmethod
    async def add_docs(self, docs_dict: dict[str, str]) -> None:
        """Add or replace documents keyed by id."""

    @abstractmethod
    async def delete_docs(self, doc_ids: list[str]) -> None:
        """Delete documents by id; missing ids are skipped."""

    @abstractmethod
    async def retrieve(self, query: str, limit: int = 3) -> dict[str, float]:
        """Return top-`limit` doc_id → score for the given query."""

    async def score_documents(self, query: str, document_ids: Collection[str]) -> dict[str, float]:
        """Score selected documents using the index's normal corpus statistics.

        This compatibility implementation uses the existing public retrieval
        contract. Backends can override it to avoid ranking the whole corpus.
        """
        allowed = set(document_ids)
        if not allowed:
            return {}
        results = await self.retrieve(query, limit=len(self.document_ids))
        return {doc_id: score for doc_id, score in results.items() if doc_id in allowed}

    async def retrieve_filtered(
        self,
        query: str,
        limit: int,
        document_ids: Collection[str],
    ) -> dict[str, float]:
        """Return top results restricted to document IDs, preserving old APIs."""
        if limit <= 0:
            return {}
        scores = await self.score_documents(query, document_ids)
        return dict(list(scores.items())[:limit])

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
