"""BM25 search engine with persistent index support.

Implements Okapi BM25 ranking with an inverted index for efficient
document lookup, incremental updates, and pickle-based persistence.
"""

import math
import pickle
from collections import Counter
from typing import TypedDict

from .base_keyword_index import BaseKeywordIndex
from ..component_registry import R


class DocMeta(TypedDict):
    """Per-document metadata: token count and unique token ID set."""

    len: int
    token_ids: set[int]


@R.register("bm25")
class BM25Index(BaseKeywordIndex):
    """BM25 search engine with file-based persistence.

    Args:
        k1: Term frequency saturation parameter (default 1.5).
        b: Document length normalization parameter (default 0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, **kwargs):
        super().__init__(**kwargs)
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}  # token -> token_id
        self.inverted_index: dict[int, dict[str, int]] = {}  # token_id -> {doc_id: tf}
        self.doc_meta: dict[str, DocMeta] = {}  # doc_id -> metadata
        self.total_len: int = 0
        self._idf_cache: dict[int, float] = {}

    # -- Properties -----------------------------------------------------------

    @property
    def n_docs(self) -> int:
        """Number of indexed documents."""
        return len(self.doc_meta)

    @property
    def avg_len(self) -> float:
        """Average document length in tokens."""
        return self.total_len / self.n_docs if self.n_docs > 0 else 0.0

    # -- Internal helpers -----------------------------------------------------

    def _tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """Map tokens to integer IDs, assigning new IDs on first encounter."""
        ids = []
        for token in tokens:
            token = token.strip()
            if token:
                ids.append(self.vocab.setdefault(token, len(self.vocab)))
        return ids

    def _remove_doc(self, doc_id: str) -> None:
        """Remove a single document from all internal structures."""
        if doc_id not in self.doc_meta:
            return
        meta = self.doc_meta[doc_id]
        self.total_len -= meta["len"]
        for tid in meta["token_ids"]:
            if tid in self.inverted_index:
                self.inverted_index[tid].pop(doc_id, None)
                if not self.inverted_index[tid]:
                    del self.inverted_index[tid]
        del self.doc_meta[doc_id]

    def _get_idf(self, token_id: int) -> float:
        """Compute and cache IDF for a token ID."""
        if token_id in self._idf_cache:
            return self._idf_cache[token_id]
        df = len(self.inverted_index.get(token_id, {}))
        self._idf_cache[token_id] = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5)) if df else 0.0
        return self._idf_cache[token_id]

    # -- Public API -----------------------------------------------------------

    async def add_docs(self, docs_dict: dict[str, str]) -> None:
        """Index or update multiple documents. Mapping of doc_id to content."""
        for doc_id, content in docs_dict.items():
            if doc_id in self.doc_meta:
                self._remove_doc(doc_id)
            tokens = self._tokenize(content)
            if not tokens:
                continue
            token_ids = self._tokens_to_ids(tokens)
            token_counts = Counter(token_ids)
            for tid, tf in token_counts.items():
                self.inverted_index.setdefault(tid, {})[doc_id] = tf
            self.doc_meta[doc_id] = {"len": len(token_ids), "token_ids": set(token_counts)}
            self.total_len += len(token_ids)
        self._idf_cache = {}

    async def delete_docs(self, doc_ids: list[str]) -> None:
        """Remove documents by their IDs."""
        for doc_id in doc_ids:
            self._remove_doc(doc_id)
        self._idf_cache = {}

    async def retrieve(self, query: str, limit: int = 3) -> dict[str, float]:
        """Search documents. Returns {doc_id: score} sorted descending."""
        query_ids = [self.vocab[t] for t in self._tokenize(query) if t in self.vocab]
        if not query_ids or self.n_docs == 0:
            return {}

        scores: dict[str, float] = {}
        avg_len = self.avg_len
        for tid in query_ids:
            if tid not in self.inverted_index:
                continue
            idf = self._get_idf(tid)
            for doc_id, tf in self.inverted_index[tid].items():
                doc_len = self.doc_meta[doc_id]["len"]
                tf_score = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * doc_len / avg_len))
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf_score

        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]) if scores else {}

    async def dump(self) -> None:
        """Persist index to disk via pickle."""
        with open(self.index_file, "wb") as f:
            pickle.dump(
                {
                    "vocab": self.vocab,
                    "inverted_index": self.inverted_index,
                    "doc_meta": self.doc_meta,
                    "total_len": self.total_len,
                    "k1": self.k1,
                    "b": self.b,
                },
                f,
            )

    async def load(self) -> None:
        """Load index from disk. Clears index on failure."""
        try:
            with open(self.index_file, "rb") as f:
                data = pickle.load(f)
            self.vocab = data["vocab"]
            self.inverted_index = data["inverted_index"]
            self.doc_meta = data["doc_meta"]
            self.total_len = data.get("total_len", 0)
            self.k1 = data.get("k1", 1.5)
            self.b = data.get("b", 0.75)
            self._idf_cache = {}
        except Exception as e:
            self.logger.exception(f"Failed to load index: {e}")
            self.index_file.unlink(missing_ok=True)
            await self.clear()

    async def clear(self) -> None:
        """Reset index to empty state."""
        self.vocab = {}
        self.inverted_index = {}
        self.doc_meta = {}
        self.total_len = 0
        self._idf_cache = {}

    async def optimize_index(self) -> None:
        """Rebuild vocab to remove unused tokens and compact token IDs."""
        used_token_ids: set[int] = set()
        for tid in self.inverted_index:
            used_token_ids.add(tid)
        if not used_token_ids:
            await self.clear()
            return

        # Build compact ID mapping
        old_to_new: dict[int, int] = {}
        new_vocab: dict[str, int] = {}
        for token, old_tid in self.vocab.items():
            if old_tid in used_token_ids:
                new_tid = len(new_vocab)
                new_vocab[token] = new_tid
                old_to_new[old_tid] = new_tid

        # Rebuild inverted index and doc_meta with new IDs
        new_inverted_index: dict[int, dict[str, int]] = {}
        for old_tid, postings in self.inverted_index.items():
            new_inverted_index[old_to_new[old_tid]] = postings
        for meta in self.doc_meta.values():
            meta["token_ids"] = {old_to_new[t] for t in meta["token_ids"] if t in old_to_new}

        self.vocab = new_vocab
        self.inverted_index = new_inverted_index
        self._idf_cache = {}
