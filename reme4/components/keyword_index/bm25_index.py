"""BM25 search engine with persistent index support.

Implements Okapi BM25 ranking with a numpy-vectorized inverted index for
efficient document lookup, incremental updates, and pickle-based persistence.

Storage layout (source of truth):
    vocab               : dict[token, token_id]
    _doc_ids            : list[doc_id] indexed by doc_idx
    _doc_id_to_idx      : dict[doc_id, doc_idx]
    _doc_lens           : np.ndarray[int32] indexed by doc_idx
    _deleted            : np.ndarray[bool]  indexed by doc_idx (lazy deletion)
    _doc_token_ids      : list[np.ndarray[int32]] indexed by doc_idx (unique tids per doc)
    _posting_doc_idxs   : dict[token_id, np.ndarray[int32]]  posting list doc_idxs
    _posting_tfs        : dict[token_id, np.ndarray[int32]]  posting list tfs (parallel)

Deletion is lazy: ``_remove_doc`` only flips ``_deleted[idx]``; posting entries
pointing at the dead idx are masked at query time and physically dropped by
``optimize_index``. Updating an existing doc_id marks the old slot deleted and
allocates a fresh idx for the new content.
"""

import math
import pickle
from collections import Counter

import numpy as np

from .base_keyword_index import BaseKeywordIndex
from ..component_registry import R


@R.register("bm25")
class BM25Index(BaseKeywordIndex):
    """BM25 search engine with numpy-vectorized scoring and file-based persistence.

    Args:
        k1: Term frequency saturation parameter (default 1.5).
        b: Document length normalization parameter (default 0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, **kwargs):
        super().__init__(**kwargs)
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self._doc_ids: list[str] = []
        self._doc_id_to_idx: dict[str, int] = {}
        self._doc_lens: np.ndarray = np.zeros(0, dtype=np.int32)
        self._deleted: np.ndarray = np.zeros(0, dtype=bool)
        self._doc_token_ids: list[np.ndarray] = []
        self._posting_doc_idxs: dict[int, np.ndarray] = {}
        self._posting_tfs: dict[int, np.ndarray] = {}
        self._idf_cache: dict[int, float] = {}

    # -- Properties -----------------------------------------------------------

    @property
    def n_docs(self) -> int:
        """Number of indexed (non-deleted) documents."""
        if self._deleted.size == 0:
            return 0
        return int((~self._deleted).sum())

    @property
    def total_len(self) -> int:
        """Total tokens across non-deleted documents."""
        if self._deleted.size == 0:
            return 0
        return int(self._doc_lens[~self._deleted].sum())

    @property
    def avg_len(self) -> float:
        """Average document length in tokens (non-deleted only)."""
        n = self.n_docs
        return self.total_len / n if n > 0 else 0.0

    @property
    def doc_meta(self) -> dict[str, dict]:
        """Dict-view of {doc_id: {"len", "token_ids"}} for non-deleted docs.

        Built on demand from the numpy-backed storage; kept for backward
        compatibility with callers (and tests) that read this shape.
        """
        out: dict[str, dict] = {}
        for idx, doc_id in enumerate(self._doc_ids):
            if self._deleted[idx]:
                continue
            out[doc_id] = {
                "len": int(self._doc_lens[idx]),
                "token_ids": {int(t) for t in self._doc_token_ids[idx]},
            }
        return out

    @property
    def inverted_index(self) -> dict[int, dict[str, int]]:
        """Dict-view of {token_id: {doc_id: tf}} excluding deleted docs.

        Built on demand from the numpy-backed storage; kept for backward
        compatibility. Empty posting lists (all entries deleted) are omitted.
        """
        out: dict[int, dict[str, int]] = {}
        for tid, doc_idxs in self._posting_doc_idxs.items():
            tfs = self._posting_tfs[tid]
            posting: dict[str, int] = {}
            for i, tf in zip(doc_idxs, tfs):
                i = int(i)
                if self._deleted[i]:
                    continue
                posting[self._doc_ids[i]] = int(tf)
            if posting:
                out[tid] = posting
        return out

    # -- Internal helpers -----------------------------------------------------

    def _tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """Map tokens to integer IDs, assigning new IDs on first encounter."""
        vocab = self.vocab
        ids = []
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            tid = vocab.get(token)
            if tid is None:
                tid = len(vocab)
                vocab[token] = tid
            ids.append(tid)
        return ids

    def _remove_doc(self, doc_id: str) -> None:
        """Mark a document deleted. Posting cleanup deferred to ``optimize_index``."""
        idx = self._doc_id_to_idx.get(doc_id)
        if idx is None or self._deleted[idx]:
            return
        self._deleted[idx] = True
        self._doc_id_to_idx.pop(doc_id, None)
        self._idf_cache = {}

    def _get_idf(self, token_id: int, n_docs: int | None = None) -> float:
        """Compute and cache IDF for a token ID against current active doc set."""
        if token_id in self._idf_cache:
            return self._idf_cache[token_id]
        doc_idxs = self._posting_doc_idxs.get(token_id)
        if doc_idxs is None or doc_idxs.size == 0:
            self._idf_cache[token_id] = 0.0
            return 0.0
        df = int((~self._deleted[doc_idxs]).sum())
        if df == 0:
            self._idf_cache[token_id] = 0.0
            return 0.0
        if n_docs is None:
            n_docs = self.n_docs
        self._idf_cache[token_id] = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        return self._idf_cache[token_id]

    # -- Public API -----------------------------------------------------------

    async def add_docs(self, docs_dict: dict[str, str]) -> None:
        """Index or update multiple documents. Mapping of doc_id to content.

        Updating an existing doc_id marks the old slot deleted and allocates
        a new doc_idx, so the next ``optimize_index`` reclaims its postings.
        """
        if not docs_dict:
            return

        new_doc_ids: list[str] = []
        new_doc_lens: list[int] = []
        new_doc_token_ids: list[np.ndarray] = []
        pending_postings: dict[int, list[tuple[int, int]]] = {}

        next_idx = len(self._doc_ids)

        for doc_id, content in docs_dict.items():
            old_idx = self._doc_id_to_idx.get(doc_id)
            if old_idx is not None and not self._deleted[old_idx]:
                self._deleted[old_idx] = True
                self._doc_id_to_idx.pop(doc_id, None)

            token_ids = self._tokens_to_ids(self._tokenize(content))
            if not token_ids:
                continue

            token_counts = Counter(token_ids)
            unique_tids = np.fromiter(
                token_counts.keys(), dtype=np.int32, count=len(token_counts)
            )

            idx = next_idx
            next_idx += 1
            new_doc_ids.append(doc_id)
            new_doc_lens.append(len(token_ids))
            new_doc_token_ids.append(unique_tids)
            self._doc_id_to_idx[doc_id] = idx

            for tid, tf in token_counts.items():
                pending_postings.setdefault(tid, []).append((idx, tf))

        if new_doc_ids:
            self._doc_ids.extend(new_doc_ids)
            self._doc_token_ids.extend(new_doc_token_ids)
            self._doc_lens = np.concatenate(
                [self._doc_lens, np.array(new_doc_lens, dtype=np.int32)]
            )
            self._deleted = np.concatenate(
                [self._deleted, np.zeros(len(new_doc_ids), dtype=bool)]
            )

        for tid, items in pending_postings.items():
            n = len(items)
            new_idxs = np.fromiter((idx for idx, _ in items), dtype=np.int32, count=n)
            new_tfs = np.fromiter((tf for _, tf in items), dtype=np.int32, count=n)
            if tid in self._posting_doc_idxs:
                self._posting_doc_idxs[tid] = np.concatenate([self._posting_doc_idxs[tid], new_idxs])
                self._posting_tfs[tid] = np.concatenate([self._posting_tfs[tid], new_tfs])
            else:
                self._posting_doc_idxs[tid] = new_idxs
                self._posting_tfs[tid] = new_tfs

        self._idf_cache = {}

    async def delete_docs(self, doc_ids: list[str]) -> None:
        """Remove documents by their IDs."""
        for doc_id in doc_ids:
            self._remove_doc(doc_id)
        self._idf_cache = {}

    async def retrieve(self, query: str, limit: int = 3) -> dict[str, float]:
        """Search documents. Returns {doc_id: score} sorted descending."""
        n_slots = self._doc_lens.size
        if n_slots == 0:
            return {}

        vocab = self.vocab
        query_ids = list(dict.fromkeys(vocab[t] for t in self._tokenize(query) if t in vocab))
        if not query_ids:
            return {}

        n_docs = self.n_docs
        if n_docs == 0:
            return {}

        avg_len = self.total_len / n_docs
        k1, b = self.k1, self.b
        denom_base = k1 * (1.0 - b)
        denom_norm = k1 * b / avg_len if avg_len > 0 else 0.0

        scores = np.zeros(n_slots, dtype=np.float32)

        for tid in query_ids:
            doc_idxs = self._posting_doc_idxs.get(tid)
            if doc_idxs is None or doc_idxs.size == 0:
                continue
            idf = self._get_idf(tid, n_docs=n_docs)
            if idf == 0.0:
                continue
            tfs = self._posting_tfs[tid].astype(np.float32)
            d_lens = self._doc_lens[doc_idxs].astype(np.float32)
            tf_score = tfs * (k1 + 1.0) / (tfs + denom_base + denom_norm * d_lens)
            # Each doc_idx appears at most once per posting list (Counter dedups
            # within a doc, and updates allocate a fresh idx), so direct
            # advanced-indexing assignment-add is safe.
            scores[doc_idxs] += idf * tf_score

        if self._deleted.any():
            scores[self._deleted] = 0.0

        positive_count = int((scores > 0).sum())
        if positive_count == 0:
            return {}
        k = min(limit, positive_count)
        if k >= n_slots:
            top_idxs = np.argsort(-scores)[:k]
        else:
            top_idxs = np.argpartition(-scores, k - 1)[:k]
            top_idxs = top_idxs[np.argsort(-scores[top_idxs])]

        return {
            self._doc_ids[int(i)]: float(scores[int(i)])
            for i in top_idxs
            if scores[int(i)] > 0
        }

    async def dump(self) -> None:
        """Persist index to disk via pickle (atomic rename)."""
        try:
            tmp = self.index_file.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(
                    {
                        "vocab": self.vocab,
                        "doc_ids": self._doc_ids,
                        "doc_id_to_idx": self._doc_id_to_idx,
                        "doc_lens": self._doc_lens,
                        "deleted": self._deleted,
                        "doc_token_ids": self._doc_token_ids,
                        "posting_doc_idxs": self._posting_doc_idxs,
                        "posting_tfs": self._posting_tfs,
                        "k1": self.k1,
                        "b": self.b,
                    },
                    f,
                )
            tmp.replace(self.index_file)
            self.logger.info(f"Saved {self.n_docs} docs to {self.index_file}")
        except Exception as e:
            self.logger.exception(f"Failed to write {self.index_file}: {e}")

    async def load(self) -> None:
        """Load index from disk. No-op if file missing; clears index on corruption."""
        if not self.index_file.exists():
            return
        try:
            with open(self.index_file, "rb") as f:
                data = pickle.load(f)
            self.vocab = data["vocab"]
            self._doc_ids = data["doc_ids"]
            self._doc_id_to_idx = data["doc_id_to_idx"]
            self._doc_lens = data["doc_lens"]
            self._deleted = data["deleted"]
            self._doc_token_ids = data["doc_token_ids"]
            self._posting_doc_idxs = data["posting_doc_idxs"]
            self._posting_tfs = data["posting_tfs"]
            self.k1 = data.get("k1", 1.5)
            self.b = data.get("b", 0.75)
            self._idf_cache = {}
            self.logger.info(f"Loaded {self.n_docs} docs from {self.index_file}")
        except Exception as e:
            self.logger.exception(f"Failed to load index: {e}")
            self.index_file.unlink(missing_ok=True)
            await self.clear()

    async def clear(self) -> None:
        """Reset index to empty state and remove persisted file."""
        self.vocab = {}
        self._doc_ids = []
        self._doc_id_to_idx = {}
        self._doc_lens = np.zeros(0, dtype=np.int32)
        self._deleted = np.zeros(0, dtype=bool)
        self._doc_token_ids = []
        self._posting_doc_idxs = {}
        self._posting_tfs = {}
        self._idf_cache = {}
        self.index_file.unlink(missing_ok=True)

    async def optimize_index(self) -> None:
        """Compact: drop deleted docs, reassign doc_idx, prune unused tokens."""
        if self._deleted.size == 0:
            return

        active_mask = ~self._deleted
        if not active_mask.any():
            await self.clear()
            return

        active_old_idxs = np.where(active_mask)[0]
        n_active = int(active_old_idxs.size)
        old_to_new_idx = -np.ones(self._deleted.size, dtype=np.int32)
        old_to_new_idx[active_old_idxs] = np.arange(n_active, dtype=np.int32)

        new_doc_ids = [self._doc_ids[int(i)] for i in active_old_idxs]
        new_doc_lens = self._doc_lens[active_mask].astype(np.int32, copy=True)
        new_doc_token_ids_pre = [self._doc_token_ids[int(i)] for i in active_old_idxs]
        new_doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(new_doc_ids)}

        used_tids: set[int] = set()
        for tid, doc_idxs in self._posting_doc_idxs.items():
            if active_mask[doc_idxs].any():
                used_tids.add(tid)

        old_tid_to_new: dict[int, int] = {}
        new_vocab: dict[str, int] = {}
        for token, old_tid in self.vocab.items():
            if old_tid in used_tids:
                new_tid = len(new_vocab)
                new_vocab[token] = new_tid
                old_tid_to_new[old_tid] = new_tid

        new_posting_doc_idxs: dict[int, np.ndarray] = {}
        new_posting_tfs: dict[int, np.ndarray] = {}
        for tid, doc_idxs in self._posting_doc_idxs.items():
            if tid not in old_tid_to_new:
                continue
            mask = active_mask[doc_idxs]
            kept_idxs = old_to_new_idx[doc_idxs[mask]].astype(np.int32, copy=False)
            kept_tfs = self._posting_tfs[tid][mask].astype(np.int32, copy=False)
            new_posting_doc_idxs[old_tid_to_new[tid]] = kept_idxs
            new_posting_tfs[old_tid_to_new[tid]] = kept_tfs

        new_doc_token_ids: list[np.ndarray] = []
        for arr in new_doc_token_ids_pre:
            remapped = np.fromiter(
                (old_tid_to_new[int(t)] for t in arr if int(t) in old_tid_to_new),
                dtype=np.int32,
            )
            new_doc_token_ids.append(remapped)

        self.vocab = new_vocab
        self._doc_ids = new_doc_ids
        self._doc_id_to_idx = new_doc_id_to_idx
        self._doc_lens = new_doc_lens
        self._deleted = np.zeros(n_active, dtype=bool)
        self._doc_token_ids = new_doc_token_ids
        self._posting_doc_idxs = new_posting_doc_idxs
        self._posting_tfs = new_posting_tfs
        self._idf_cache = {}
