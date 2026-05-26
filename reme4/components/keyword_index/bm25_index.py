"""基于 BM25 的倒排索引实现，支持持久化。

核心存储结构（落盘真相源）：
    vocab               : dict[token, token_id]
    _doc_ids            : list[doc_id]，按 doc_idx 索引
    _doc_id_to_idx      : dict[doc_id, doc_idx]
    _doc_lens           : np.ndarray[int32]，按 doc_idx 索引
    _deleted            : np.ndarray[bool]，按 doc_idx 索引（懒删除标记）
    _doc_token_ids      : list[np.ndarray[int32]]，每篇文档去重后的 token_id
    _posting_doc_idxs   : dict[token_id, np.ndarray[int32]]，倒排表的 doc_idx
    _posting_tfs        : dict[token_id, np.ndarray[int32]]，与上方一一对应的词频

删除采用懒标记：_deleted[idx] = True 即视为删除，倒排表中的物理回收由
optimize_index 统一完成；更新已存在的 doc_id 时，先把旧槽位标记删除，再
分配新的 idx。
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

    def __init__(self, k1: float = 1.5, b: float = 0.75, index_version: str = "v1", **kwargs):
        super().__init__(**kwargs)
        self.k1 = k1
        self.b = b
        self.index_version = index_version

        # 词表与文档元数据
        self.vocab: dict[str, int] = {}
        self._doc_ids: list[str] = []
        self._doc_id_to_idx: dict[str, int] = {}
        self._doc_lens: np.ndarray = np.zeros(0, dtype=np.int32)
        self._deleted: np.ndarray = np.zeros(0, dtype=bool)
        self._doc_token_ids: list[np.ndarray] = []

        # 倒排表：token_id -> (doc_idxs, tfs)
        self._posting_doc_idxs: dict[int, np.ndarray] = {}
        self._posting_tfs: dict[int, np.ndarray] = {}

        # IDF 缓存，对增删与重建索引时失效
        self._idf_cache: dict[int, float] = {}

    # -- Properties -----------------------------------------------------------

    @property
    def index_file(self) -> Path:
        """落盘文件路径，包含分词器名与索引版本，便于区分不同配置。"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call start() first.")
        name = type(self.tokenizer).__name__.replace("Tokenizer", "").lower()
        return self.component_metadata_path / f"bm25_{name}_{self.index_version}.pkl"

    @property
    def n_docs(self) -> int:
        """当前存活文档数（排除懒删除）。"""
        return 0 if self._deleted.size == 0 else int((~self._deleted).sum())

    @property
    def total_len(self) -> int:
        """所有存活文档的 token 总数。"""
        return 0 if self._deleted.size == 0 else int(self._doc_lens[~self._deleted].sum())

    @property
    def avg_len(self) -> float:
        """存活文档的平均长度，用于 BM25 长度归一化。"""
        n = self.n_docs
        return self.total_len / n if n > 0 else 0.0

    @property
    def doc_meta(self) -> dict[str, dict]:
        """对外暴露每篇存活文档的长度与去重后的 token_id 集合。"""
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
        """重建可读形式的倒排表：token_id -> {doc_id: tf}，跳过已删除文档。"""
        out: dict[int, dict[str, int]] = {}
        for tid, doc_idxs in self._posting_doc_idxs.items():
            tfs = self._posting_tfs[tid]
            posting = {
                self._doc_ids[int(i)]: int(tf)
                for i, tf in zip(doc_idxs, tfs)
                if not self._deleted[int(i)]
            }
            if posting:
                out[tid] = posting
        return out

    # -- Internal helpers -----------------------------------------------------

    def _tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """将 token 转为 id；遇到新词时自动分配新的 token_id。"""
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
        """懒删除：仅置 _deleted 位并解除 doc_id 映射，不动倒排表。"""
        idx = self._doc_id_to_idx.get(doc_id)
        if idx is None or self._deleted[idx]:
            return
        self._deleted[idx] = True
        self._doc_id_to_idx.pop(doc_id, None)
        self._idf_cache = {}

    def _get_idf(self, token_id: int, n_docs: int | None = None) -> float:
        """计算并缓存 token 的 IDF；存活文档数发生变化时缓存会被清空。"""
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
        """分词并统计词频；若 doc_id 已存在则先标记旧版为删除。空文档返回 None。"""
        self._remove_doc(doc_id)
        token_ids = self._tokens_to_ids(self._tokenize(content))
        if not token_ids:
            return None
        counts = Counter(token_ids)
        unique_tids = np.fromiter(counts.keys(), dtype=np.int32, count=len(counts))
        return unique_tids, len(token_ids), counts

    def _append_doc_arrays(
        self, new_doc_ids: list[str], new_doc_lens: list[int], new_doc_token_ids: list[np.ndarray]
    ) -> None:
        """把一批新文档的元数据一次性追加到文档数组中。"""
        if not new_doc_ids:
            return
        self._doc_ids.extend(new_doc_ids)
        self._doc_token_ids.extend(new_doc_token_ids)
        self._doc_lens = np.concatenate([self._doc_lens, np.array(new_doc_lens, dtype=np.int32)])
        self._deleted = np.concatenate([self._deleted, np.zeros(len(new_doc_ids), dtype=bool)])

    def _extend_postings(self, pending: dict[int, list[tuple[int, int]]]) -> None:
        """把待写入的 (doc_idx, tf) 增量按 token 追加到倒排表。"""
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
        """切词、过滤未登录词并去重，返回查询的 token_id 列表。"""
        vocab = self.vocab
        return list(dict.fromkeys(vocab[t] for t in self._tokenize(query) if t in vocab))

    def _top_k(self, scores: np.ndarray, limit: int) -> np.ndarray:
        """挑出得分前 limit 名（且严格大于 0）的索引，按得分降序排列。"""
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
        """批量加入文档；已存在的 doc_id 会被替换为新版本。"""
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
        """批量懒删除；倒排表中的物理回收由 optimize_index 完成。"""
        for doc_id in doc_ids:
            self._remove_doc(doc_id)
        self._idf_cache = {}

    def _score_query(self, query_ids: list[int], n_docs: int) -> np.ndarray:
        """对所有文档计算 BM25 得分，已删除文档置 0。"""
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
            # 同一倒排表中每个 doc_idx 至多出现一次：Counter 已在文档内去重，
            # 文档更新也会分配新的 idx，因此可安全使用花式索引累加。
            scores[doc_idxs] += idf * tfs * (k1 + 1.0) / (tfs + denom_base + denom_norm * d_lens)

        if self._deleted.any():
            scores[self._deleted] = 0.0
        return scores

    async def retrieve(self, query: str, limit: int = 3) -> dict[str, float]:
        """对查询做 BM25 召回，返回 {doc_id: score}，按得分降序。"""
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
        """收集需要落盘的全部字段，集中在一处以便与 _restore 对齐。"""
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
        """从 _snapshot 产生的字典还原索引内部状态。"""
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
        """通过临时文件 + 原子替换的方式持久化索引，避免半写状态。"""
        try:
            tmp = self.index_file.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(self._snapshot(), f)
            tmp.replace(self.index_file)
            self.logger.info(f"Saved {self.n_docs} docs to {self.index_file}")
        except Exception as e:
            self.logger.exception(f"Failed to write {self.index_file}: {e}")

    async def load(self) -> None:
        """读取持久化文件并还原索引；文件不存在则不做事，损坏则清空。"""
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
        """清空内存中的索引并删除持久化文件。"""
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
        """构造 old_idx → new_idx 的映射数组（被删槽位为 -1），并返回存活数量。"""
        active_old_idxs = np.where(active_mask)[0]
        n_active = int(active_old_idxs.size)
        remap = -np.ones(self._deleted.size, dtype=np.int32)
        remap[active_old_idxs] = np.arange(n_active, dtype=np.int32)
        return remap, n_active

    def _compact_vocab(self, active_mask: np.ndarray) -> tuple[dict[str, int], dict[int, int]]:
        """只保留仍被任意存活文档引用的 token，重排成连续的新 token_id。"""
        used_tids = {
            tid for tid, doc_idxs in self._posting_doc_idxs.items()
            if active_mask[doc_idxs].any()
        }
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
        """剔除删除文档并按新 idx/tid 重写倒排表。"""
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
        self, active_mask: np.ndarray, old_tid_to_new: dict[int, int]
    ) -> tuple[list[str], list[np.ndarray]]:
        """在压缩后的词表下重建存活文档的 doc_id 列表与去重 token_id 数组。"""
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
        """物理回收懒删除的文档与未被引用的词表项，重建紧凑索引。"""
        if self._deleted.size == 0:
            return
        active_mask = ~self._deleted
        if not active_mask.any():
            await self.clear()
            return

        old_to_new_idx, n_active = self._build_idx_remap(active_mask)
        new_vocab, old_tid_to_new = self._compact_vocab(active_mask)
        new_posting_idxs, new_posting_tfs = self._compact_postings(
            active_mask, old_to_new_idx, old_tid_to_new
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
