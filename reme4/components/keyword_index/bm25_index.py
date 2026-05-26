"""BM25 inverted index with on-disk persistence.

On-disk truth source (see `_snapshot` / `_restore`):
    vocab               : dict[token, token_id]
    _doc_ids            : list[doc_id], indexed by doc_idx
    _doc_id_to_idx      : dict[doc_id, doc_idx]
    _doc_lens           : np.int32[n], indexed by doc_idx
    _deleted            : np.bool[n], lazy-delete flag per doc_idx
    _doc_token_ids      : list[np.int32[]], unique token_ids per doc
    _posting_doc_idxs   : dict[token_id, np.int32[]], posting list (doc_idx)
    _posting_tfs        : dict[token_id, np.int32[]], aligned term frequencies

Deletion is lazy: setting `_deleted[idx] = True` retires the slot. The posting
lists keep the stale entries until `optimize_index` rewrites them. Updating an
existing doc_id retires the old slot first, then allocates a fresh idx.
"""

import math
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from .base_keyword_index import BaseKeywordIndex
from ..component_registry import R


@R.register("bm25")
class BM25Index(BaseKeywordIndex):
    """BM25 inverted index with lazy deletion and on-disk persistence."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, index_version: str = "v1", **kwargs):
        super().__init__(**kwargs)
        self.k1 = k1
        self.b = b
        self.index_version = index_version

        self.vocab: dict[str, int] = {}
        self._doc_ids: list[str] = []
        self._doc_id_to_idx: dict[str, int] = {}
        self._doc_lens: np.ndarray = np.zeros(0, dtype=np.int32)
        self._deleted: np.ndarray = np.zeros(0, dtype=bool)
        self._doc_token_ids: list[np.ndarray] = []

        self._posting_doc_idxs: dict[int, np.ndarray] = {}
        self._posting_tfs: dict[int, np.ndarray] = {}

        # IDF cache; invalidated whenever live-doc count or postings change.
        self._idf_cache: dict[int, float] = {}

    # -- Properties -----------------------------------------------------------

    @property
    def index_file(self) -> Path:
        """Path of the persisted index, namespaced by tokenizer and version."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call start() first.")
        name = type(self.tokenizer).__name__.replace("Tokenizer", "").lower()
        return self.component_metadata_path / f"bm25_{name}_{self.index_version}.pkl"

    @property
    def n_docs(self) -> int:
        """Number of live (non-deleted) documents."""
        return 0 if self._deleted.size == 0 else int((~self._deleted).sum())

    @property
    def total_len(self) -> int:
        """Sum of token counts across live documents."""
        return 0 if self._deleted.size == 0 else int(self._doc_lens[~self._deleted].sum())

    @property
    def avg_len(self) -> float:
        """Average length of live documents, used for BM25 length normalization."""
        n = self.n_docs
        return self.total_len / n if n > 0 else 0.0

    @property
    def doc_meta(self) -> dict[str, dict]:
        """Per-live-doc length and unique token_id set, keyed by doc_id."""
        return {
            self._doc_ids[idx]: {
                "len": int(self._doc_lens[idx]),
                "token_ids": {int(t) for t in self._doc_token_ids[idx]},
            }
            for idx in range(len(self._doc_ids))
            if not self._deleted[idx]
        }

    @property
    def inverted_index(self) -> dict[int, dict[str, int]]:
        """Readable view of postings: token_id -> {doc_id: tf}, deleted skipped."""
        out: dict[int, dict[str, int]] = {}
        for tid, doc_idxs in self._posting_doc_idxs.items():
            tfs = self._posting_tfs[tid]
            posting = {self._doc_ids[int(i)]: int(tf) for i, tf in zip(doc_idxs, tfs) if not self._deleted[int(i)]}
            if posting:
                out[tid] = posting
        return out

    # -- Internal helpers -----------------------------------------------------

    def _tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """Map tokens to ids, allocating a fresh id for any unseen token."""
        vocab = self.vocab
        ids: list[int] = []
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
        """Lazy-delete a doc: flip `_deleted` and drop the id mapping."""
        idx = self._doc_id_to_idx.get(doc_id)
        if idx is None or self._deleted[idx]:
            return
        self._deleted[idx] = True
        self._doc_id_to_idx.pop(doc_id, None)
        self._idf_cache = {}

    def _get_idf(self, token_id: int, n_docs: int | None = None) -> float:
        """Return the cached IDF for a token, computing it on miss."""
        if token_id in self._idf_cache:
            return self._idf_cache[token_id]
        doc_idxs = self._posting_doc_idxs.get(token_id)
        if doc_idxs is None or doc_idxs.size == 0:
            self._idf_cache[token_id] = 0.0
            return 0.0
        df = int((~self._deleted[doc_idxs]).sum())
        if n_docs is None:
            n_docs = self.n_docs
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5)) if df else 0.0
        self._idf_cache[token_id] = idf
        return idf

    def _prepare_doc(self, doc_id: str, content: str) -> tuple[np.ndarray, int, Counter] | None:
        """Tokenize and count terms; retire any prior version of `doc_id`."""
        self._remove_doc(doc_id)
        token_ids = self._tokens_to_ids(self._tokenize(content))
        if not token_ids:
            return None
        counts = Counter(token_ids)
        unique_tids = np.fromiter(counts.keys(), dtype=np.int32, count=len(counts))
        return unique_tids, len(token_ids), counts

    def _append_doc_arrays(
        self,
        new_doc_ids: list[str],
        new_doc_lens: list[int],
        new_doc_token_ids: list[np.ndarray],
    ) -> None:
        """Append metadata for a batch of new docs to the doc-level arrays."""
        if not new_doc_ids:
            return
        self._doc_ids.extend(new_doc_ids)
        self._doc_token_ids.extend(new_doc_token_ids)
        self._doc_lens = np.concatenate([self._doc_lens, np.array(new_doc_lens, dtype=np.int32)])
        self._deleted = np.concatenate([self._deleted, np.zeros(len(new_doc_ids), dtype=bool)])

    def _extend_postings(self, pending: dict[int, list[tuple[int, int]]]) -> None:
        """Append pending (doc_idx, tf) pairs to each token's posting list."""
        for tid, items in pending.items():
            n = len(items)
            new_idxs = np.fromiter((idx for idx, _ in items), dtype=np.int32, count=n)
            new_tfs = np.fromiter((tf for _, tf in items), dtype=np.int32, count=n)
            if tid in self._posting_doc_idxs:
                self._posting_doc_idxs[tid] = np.concatenate([self._posting_doc_idxs[tid], new_idxs])
                self._posting_tfs[tid] = np.concatenate([self._posting_tfs[tid], new_tfs])
            else:
                self._posting_doc_idxs[tid] = new_idxs
                self._posting_tfs[tid] = new_tfs

    def _encode_query(self, query: str) -> list[int]:
        """Tokenize query; drop OOV terms; deduplicate while preserving order."""
        vocab = self.vocab
        return list(dict.fromkeys(vocab[t] for t in self._tokenize(query) if t in vocab))

    def _top_k(self, scores: np.ndarray, limit: int) -> np.ndarray:
        """Indices of the top `limit` strictly-positive scores, descending."""
        if limit <= 0:
            return np.empty(0, dtype=np.int64)
        positive_count = int((scores > 0).sum())
        if positive_count == 0:
            return np.empty(0, dtype=np.int64)
        k = min(limit, positive_count)
        if k >= scores.size:
            return np.argsort(-scores)[:k]
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.argsort(-scores[top])]

    # -- Public API -----------------------------------------------------------

    async def add_docs(self, docs_dict: dict[str, str]) -> None:
        """Add or replace documents in batch (existing doc_ids are overwritten)."""
        if not docs_dict:
            return

        new_doc_ids: list[str] = []
        new_doc_lens: list[int] = []
        new_doc_token_ids: list[np.ndarray] = []
        pending: dict[int, list[tuple[int, int]]] = {}
        next_idx = len(self._doc_ids)

        for doc_id, content in docs_dict.items():
            prepared = self._prepare_doc(doc_id, content)
            if prepared is None:
                continue
            unique_tids, n_tokens, token_counts = prepared

            idx = next_idx
            next_idx += 1
            new_doc_ids.append(doc_id)
            new_doc_lens.append(n_tokens)
            new_doc_token_ids.append(unique_tids)
            self._doc_id_to_idx[doc_id] = idx
            for tid, tf in token_counts.items():
                pending.setdefault(tid, []).append((idx, tf))

        self._append_doc_arrays(new_doc_ids, new_doc_lens, new_doc_token_ids)
        self._extend_postings(pending)
        self._idf_cache = {}

    async def delete_docs(self, doc_ids: list[str]) -> None:
        """Lazy-delete a batch of doc_ids; physical reclaim happens in optimize_index."""
        for doc_id in doc_ids:
            self._remove_doc(doc_id)
        self._idf_cache = {}

    def _score_query(self, query_ids: list[int], n_docs: int) -> np.ndarray:
        """Compute BM25 scores across all docs; deleted docs zeroed out."""
        avg_len = self.total_len / n_docs
        k1, b = self.k1, self.b
        denom_base = k1 * (1.0 - b)
        denom_norm = k1 * b / avg_len if avg_len > 0 else 0.0

        scores = np.zeros(self._doc_lens.size, dtype=np.float32)
        for tid in query_ids:
            doc_idxs = self._posting_doc_idxs.get(tid)
            if doc_idxs is None or doc_idxs.size == 0:
                continue
            idf = self._get_idf(tid, n_docs)
            if idf == 0.0:
                continue
            tfs = self._posting_tfs[tid].astype(np.float32)
            d_lens = self._doc_lens[doc_idxs].astype(np.float32)
            # Each doc_idx appears at most once per posting list (Counter dedups
            # within a doc, and updates allocate fresh idxs), so fancy-index
            # accumulation is safe here.
            scores[doc_idxs] += idf * tfs * (k1 + 1.0) / (tfs + denom_base + denom_norm * d_lens)

        if self._deleted.any():
            scores[self._deleted] = 0.0
        return scores

    async def retrieve(self, query: str, limit: int = 3) -> dict[str, float]:
        """BM25 retrieval; returns {doc_id: score} sorted by score descending."""
        n_docs = self.n_docs
        if n_docs == 0:
            return {}
        query_ids = self._encode_query(query)
        if not query_ids:
            return {}

        scores = self._score_query(query_ids, n_docs)
        top_idxs = self._top_k(scores, limit)
        return {self._doc_ids[int(i)]: float(scores[int(i)]) for i in top_idxs}

    # -- Persistence ----------------------------------------------------------

    def _snapshot(self) -> dict:
        """Bundle every persistent field; mirrors `_restore`."""
        return {
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
        }

    def _restore(self, data: dict) -> None:
        """Restore index state from a `_snapshot` dict."""
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

    async def dump(self) -> None:
        """Persist the index via temp file + atomic rename to avoid torn writes."""
        try:
            tmp = self.index_file.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(self._snapshot(), f)
            tmp.replace(self.index_file)
            self.logger.info(f"Saved {self.n_docs} docs to {self.index_file}")
        except Exception as e:
            self.logger.exception(f"Failed to write {self.index_file}: {e}")

    async def load(self) -> None:
        """Load from disk; missing file is a no-op, corrupt file resets state."""
        if not self.index_file.exists():
            return
        try:
            with open(self.index_file, "rb") as f:
                data = pickle.load(f)
            self._restore(data)
            self.logger.info(f"Loaded {self.n_docs} docs from {self.index_file}")
        except Exception as e:
            self.logger.exception(f"Failed to load index: {e}")
            self.index_file.unlink(missing_ok=True)
            await self.clear()

    async def clear(self) -> None:
        """Reset in-memory state and remove the persisted file."""
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

    # -- Compaction -----------------------------------------------------------

    def _build_idx_remap(self, active_mask: np.ndarray) -> tuple[np.ndarray, int]:
        """Build an old_idx -> new_idx array (-1 for retired slots)."""
        active_old_idxs = np.where(active_mask)[0]
        n_active = int(active_old_idxs.size)
        remap = -np.ones(self._deleted.size, dtype=np.int32)
        remap[active_old_idxs] = np.arange(n_active, dtype=np.int32)
        return remap, n_active

    def _compact_vocab(self, active_mask: np.ndarray) -> tuple[dict[str, int], dict[int, int]]:
        """Keep only tokens still referenced by a live doc; renumber contiguously."""
        used_tids = {tid for tid, doc_idxs in self._posting_doc_idxs.items() if active_mask[doc_idxs].any()}
        new_vocab: dict[str, int] = {}
        old_to_new: dict[int, int] = {}
        for token, old_tid in self.vocab.items():
            if old_tid in used_tids:
                new_tid = len(new_vocab)
                new_vocab[token] = new_tid
                old_to_new[old_tid] = new_tid
        return new_vocab, old_to_new

    def _compact_postings(
        self,
        active_mask: np.ndarray,
        old_to_new_idx: np.ndarray,
        old_tid_to_new: dict[int, int],
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Drop deleted entries and rewrite postings under new idx/tid numbering."""
        new_idxs: dict[int, np.ndarray] = {}
        new_tfs: dict[int, np.ndarray] = {}
        for tid, doc_idxs in self._posting_doc_idxs.items():
            if tid not in old_tid_to_new:
                continue
            mask = active_mask[doc_idxs]
            new_tid = old_tid_to_new[tid]
            new_idxs[new_tid] = old_to_new_idx[doc_idxs[mask]].astype(np.int32, copy=False)
            new_tfs[new_tid] = self._posting_tfs[tid][mask].astype(np.int32, copy=False)
        return new_idxs, new_tfs

    def _compact_docs(
        self,
        active_mask: np.ndarray,
        old_tid_to_new: dict[int, int],
    ) -> tuple[list[str], list[np.ndarray]]:
        """Rebuild doc_id list and unique-token arrays under the new vocab."""
        active_old_idxs = np.where(active_mask)[0]
        new_doc_ids = [self._doc_ids[int(i)] for i in active_old_idxs]
        new_doc_token_ids = [
            np.fromiter(
                (old_tid_to_new[int(t)] for t in self._doc_token_ids[int(i)] if int(t) in old_tid_to_new),
                dtype=np.int32,
            )
            for i in active_old_idxs
        ]
        return new_doc_ids, new_doc_token_ids

    async def optimize_index(self) -> None:
        """Physically reclaim deleted docs and unused vocab entries."""
        if self._deleted.size == 0:
            return
        active_mask = ~self._deleted
        if not active_mask.any():
            await self.clear()
            return

        old_to_new_idx, n_active = self._build_idx_remap(active_mask)
        new_vocab, old_tid_to_new = self._compact_vocab(active_mask)
        new_posting_idxs, new_posting_tfs = self._compact_postings(
            active_mask,
            old_to_new_idx,
            old_tid_to_new,
        )
        new_doc_ids, new_doc_token_ids = self._compact_docs(active_mask, old_tid_to_new)

        self.vocab = new_vocab
        self._doc_ids = new_doc_ids
        self._doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(new_doc_ids)}
        self._doc_lens = self._doc_lens[active_mask].astype(np.int32, copy=True)
        self._deleted = np.zeros(n_active, dtype=bool)
        self._doc_token_ids = new_doc_token_ids
        self._posting_doc_idxs = new_posting_idxs
        self._posting_tfs = new_posting_tfs
        self._idf_cache = {}
