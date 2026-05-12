"""Lightweight BM25 search engine with persistent index support.

BM25Lite implements the Okapi BM25 ranking algorithm for text retrieval.
It uses an inverted index for efficient document lookup and supports
incremental updates, persistence via pickle, and automatic vocab compaction.
"""

import math
import pickle
from collections import Counter
from pathlib import Path
from typing import TypedDict

from ..base_component import BaseComponent
from ..tokenizer import BaseTokenizer
from ...enumeration import ComponentEnum


class DocMeta(TypedDict):
    """Document metadata stored in the index.

    Attributes:
        len: Number of tokens in the document.
        token_ids: Set of unique token IDs present in the document.
    """
    len: int
    token_ids: set[int]


class BM25Lite(BaseComponent):
    """Lightweight BM25 search engine with file-based persistence.

    BM25 (Best Matching 25) is a probabilistic ranking function that scores
    documents based on term frequency and document length normalization.

    Attributes:
        k1: Term frequency saturation parameter (default: 1.5).
        b: Document length normalization parameter (default: 0.75).
        vocab: Token to token ID mapping.
        inverted_index: Token ID to {doc_id: term_frequency} mapping.
        doc_meta: Document ID to metadata mapping.

    Args:
        index_dir: Directory to store index files.
        k1: BM25 term frequency saturation parameter.
        b: BM25 document length normalization parameter.
        tokenizer: Name of tokenizer component to use.
    """

    def __init__(
            self,
            index_dir: str | Path,
            k1: float = 1.5,
            b: float = 0.75,
            tokenizer: str = "default",
            **kwargs):
        super().__init__(**kwargs)
        self.k1 = k1
        self.b = b
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vocab: dict[str, int] = {}
        self.inverted_index: dict[int, dict[str, int]] = {}
        self.doc_meta: dict[str, DocMeta] = {}
        self.total_len = 0

        self._idf_cache: dict[int, float] = {}
        self.tokenizer_name = tokenizer
        self._tokenizer: BaseTokenizer | None = None

    def clear(self):
        """Reset the index to empty state."""
        self.vocab = {}
        self.inverted_index = {}
        self.doc_meta = {}
        self.total_len = 0
        self._idf_cache = {}

    @property
    def n_docs(self) -> int:
        """Number of indexed documents."""
        return len(self.doc_meta)

    @property
    def avg_len(self) -> float:
        """Average document length in tokens."""
        return self.total_len / self.n_docs if self.n_docs > 0 else 0.0

    @property
    def index_file(self) -> Path:
        """Path to the index pickle file based on tokenizer name."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call start() first.")
        name = type(self._tokenizer).__name__.replace("Tokenizer", "").lower()
        return self.index_dir / f"index_{name}.pkl"

    async def _start(self) -> None:
        """Initialize tokenizer and load existing index if available."""
        if self.app_context is not None:
            tokenizer_dict: dict = self.app_context.components.get(ComponentEnum.TOKENIZER, {})
            if self.tokenizer_name in tokenizer_dict:
                self._tokenizer = tokenizer_dict[self.tokenizer_name]

        if self._tokenizer is None:
            from ..tokenizer import RegexTokenizer
            self._tokenizer = RegexTokenizer(filter_stopwords=False)

        if self._tokenizer is not None:
            await self._tokenizer.start()

        if self.index_file.exists():
            self.load()
            self.logger.info(f"Loaded index from {self.index_dir}")

    async def _close(self) -> None:
        """Save index and cleanup tokenizer on shutdown."""
        if self.inverted_index:
            self.dump()
            self.logger.info(f"Saved index to {self.index_dir}")

        if self._tokenizer is not None:
            await self._tokenizer.close()

    def _tokenize(self, text: str) -> list[str]:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call start() first.")
        return self._tokenizer.tokenize([text])[0]

    def _tokens_to_ids(self, tokens: list[str]) -> list[int]:
        ids = []
        for token in tokens:
            token = token.strip()
            if token:
                ids.append(self.vocab.setdefault(token, len(self.vocab)))
        return ids

    def add_docs(self, docs_dict: dict[str, str]):
        """Index or update multiple documents.

        Args:
            docs_dict: Mapping of document ID to document content.
        """
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

    def reindex(self):
        """Rebuild vocab to remove unused tokens and compact token IDs."""
        # Collect all token IDs still in use
        used_token_ids: set[int] = set()
        for tid in self.inverted_index:
            used_token_ids.add(tid)

        if not used_token_ids:
            self.clear()
            return

        # Build new vocab with compact IDs
        old_to_new: dict[int, int] = {}
        new_vocab: dict[str, int] = {}
        for token, old_tid in self.vocab.items():
            if old_tid in used_token_ids:
                new_tid = len(new_vocab)
                new_vocab[token] = new_tid
                old_to_new[old_tid] = new_tid

        # Rebuild inverted_index with new token IDs
        new_inverted_index: dict[int, dict[str, int]] = {}
        for old_tid, postings in self.inverted_index.items():
            new_tid = old_to_new[old_tid]
            new_inverted_index[new_tid] = postings

        # Update doc_meta token_ids
        for doc_id, meta in self.doc_meta.items():
            meta["token_ids"] = {old_to_new[old_tid] for old_tid in meta["token_ids"] if old_tid in old_to_new}

        self.vocab = new_vocab
        self.inverted_index = new_inverted_index
        self._idf_cache = {}

    def _remove_doc(self, doc_id: str):
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
        if token_id in self._idf_cache:
            return self._idf_cache[token_id]
        df = len(self.inverted_index.get(token_id, {}))
        self._idf_cache[token_id] = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5)) if df else 0.0
        return self._idf_cache[token_id]

    def retrieve(self, query: str, k: int = 3) -> dict[str, float]:
        """Search for documents matching the query.

        Args:
            query: Search query string.
            k: Maximum number of results to return.

        Returns:
            Dict mapping document IDs to BM25 scores, sorted by score descending.
        """
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

        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]) if scores else {}

    def dump(self):
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

    def load(self):
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
            self.clear()
