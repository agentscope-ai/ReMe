"""FAISS-backed file store: chunk JSONL stays authoritative; FAISS replaces the linear vector scan."""

import json

import aiofiles
import numpy as np

from .local_file_store import LocalFileStore
from ..component_registry import R
from ...schema import FileChunk, FileNode


@R.register("faiss")
class FaissLocalFileStore(LocalFileStore):
    """LocalFileStore variant whose vector_search is backed by a FAISS IndexFlatIP.

    Chunk persistence is unchanged (JSONL, owned by the parent). FAISS state is
    stored alongside as a binary index plus an id-map sidecar. If either file
    is missing or stale, the index is rebuilt from ``self.file_chunks``, which
    remains the source of truth.

    faiss is imported lazily inside ``__init__`` so that merely importing this
    module (e.g. via ``reme version``) does not trigger the SWIG bindings and
    their associated DeprecationWarnings.
    """

    def __init__(
        self,
        normalize: bool = True,
        max_tombstones: int = 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._faiss = self._import_faiss()
        self.normalize = normalize
        self.max_tombstones = max_tombstones
        self.faiss_path = self.component_metadata_path / f"faiss_index_{self.name}_{self.store_version}.bin"
        self.faiss_idmap_path = self.component_metadata_path / f"faiss_idmap_{self.name}_{self.store_version}.json"
        self._faiss_index = None  # faiss.Index | None
        self._id_map: list[str] = []  # row -> chunk_id
        self._id_to_row: dict[str, int] = {}  # chunk_id -> row (live entries only)
        self._tombstones: set[int] = set()  # rows whose chunk_id was deleted

    @staticmethod
    def _import_faiss():
        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "faiss is required for FaissLocalFileStore. Install with `pip install faiss-cpu`.",
            ) from e
        return faiss

    # -- helpers ----------------------------------------------------------

    @property
    def _dim(self) -> int:
        return self.embedding_store.dimensions if self.embedding_store is not None else 0

    def _new_index(self):
        return self._faiss.IndexFlatIP(self._dim)

    def _prepare(self, vec: np.ndarray) -> np.ndarray:
        """Cast to float32 (FAISS requirement) and L2-normalize so IndexFlatIP gives cosine."""
        v = np.ascontiguousarray(vec, dtype=np.float32)
        if v.ndim == 1:
            v = v[None, :]
        if self.normalize:
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            v = v / norms
        return v

    def _add_to_index(self, chunk_ids: list[str], vectors: np.ndarray) -> None:
        if not chunk_ids or vectors.size == 0:
            return
        v = self._prepare(vectors)
        start = self._faiss_index.ntotal
        self._faiss_index.add(v)
        for offset, cid in enumerate(chunk_ids):
            row = start + offset
            old_row = self._id_to_row.get(cid)
            if old_row is not None:
                self._tombstones.add(old_row)
            self._id_map.append(cid)
            self._id_to_row[cid] = row

    def _tombstone(self, chunk_id: str) -> None:
        row = self._id_to_row.pop(chunk_id, None)
        if row is not None:
            self._tombstones.add(row)

    def _rebuild_index(self) -> None:
        """Rebuild FAISS state from self.file_chunks (the source of truth)."""
        self._faiss_index = self._new_index()
        self._id_map = []
        self._id_to_row = {}
        self._tombstones.clear()
        chunks = [c for c in self.file_chunks.values() if c.embedding is not None]
        if not chunks:
            return
        vectors = np.stack([c.embedding for c in chunks])
        self._add_to_index([c.id for c in chunks], vectors)

    def _compact_if_needed(self) -> None:
        if len(self._tombstones) >= self.max_tombstones:
            self._rebuild_index()

    # -- persistence ------------------------------------------------------

    async def load(self) -> None:
        """Load chunks via the parent, then attach FAISS state (sidecar or rebuild)."""
        await super().load()
        if self.embedding_store is None or self._dim == 0:
            self._faiss_index = None
            return
        if not await self._try_load_sidecar():
            self._rebuild_index()

    async def _try_load_sidecar(self) -> bool:
        """Read the binary index plus id-map sidecar. On any mismatch or read error,
        wipe the partial files so the caller can rebuild from chunks cleanly.
        """
        if not (self.faiss_path.exists() and self.faiss_idmap_path.exists()):
            return False
        try:
            index = self._faiss.read_index(str(self.faiss_path))
            if index.d != self._dim:
                raise ValueError(f"FAISS dim {index.d} != embedding dim {self._dim}")
            async with aiofiles.open(self.faiss_idmap_path, encoding=self.encoding) as f:
                data = json.loads(await f.read())
            id_map = list(data.get("id_map", []))
            if len(id_map) != index.ntotal:
                raise ValueError(f"id_map size {len(id_map)} != index ntotal {index.ntotal}")
            self._faiss_index = index
            self._id_map = id_map
            self._tombstones = set(data.get("tombstones", []))
            self._id_to_row = {cid: i for i, cid in enumerate(self._id_map) if i not in self._tombstones}
            self.logger.info(f"Loaded FAISS index: {index.ntotal} vectors from {self.faiss_path}")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to load FAISS index, will rebuild: {e}")
            self.faiss_path.unlink(missing_ok=True)
            self.faiss_idmap_path.unlink(missing_ok=True)
            return False

    async def dump(self) -> None:
        """Persist chunks JSONL via the parent, then write the FAISS sidecar atomically."""
        await super().dump()
        if self._faiss_index is None or self.embedding_store is None:
            return
        try:
            self._compact_if_needed()
            await self._write_sidecar()
            self.logger.info(f"Saved FAISS index: {self._faiss_index.ntotal} vectors to {self.faiss_path}")
        except Exception as e:
            self.logger.exception(f"Failed to write FAISS index: {e}")

    async def _write_sidecar(self) -> None:
        tmp_index = self.faiss_path.with_suffix(".tmp")
        self._faiss.write_index(self._faiss_index, str(tmp_index))
        tmp_index.replace(self.faiss_path)

        tmp_idmap = self.faiss_idmap_path.with_suffix(".tmp")
        payload = json.dumps({"id_map": self._id_map, "tombstones": sorted(self._tombstones)})
        async with aiofiles.open(tmp_idmap, "w", encoding=self.encoding) as f:
            await f.write(payload)
        tmp_idmap.replace(self.faiss_idmap_path)

    # -- CRUD overrides ---------------------------------------------------

    async def upsert(self, files: list[tuple[FileNode, list[FileChunk]]]) -> None:
        if not files:
            return
        assert self.file_graph is not None

        # Snapshot pre-upsert chunk_ids so we can diff against the post-upsert state.
        old_ids_by_path = {
            n.path: set(n.chunk_ids) for n in await self.file_graph.get_nodes([node.path for node, _ in files])
        }
        await super().upsert(files)

        if self._faiss_index is None or self.embedding_store is None:
            return
        self._sync_index_after_upsert(files, old_ids_by_path)

    def _sync_index_after_upsert(
        self,
        files: list[tuple[FileNode, list[FileChunk]]],
        old_ids_by_path: dict[str, set[str]],
    ) -> None:
        """Apply add/tombstone deltas to FAISS based on chunk_id set differences."""
        existing = set(self._id_to_row)
        to_add: list[FileChunk] = []
        for node, _ in files:
            new_ids = set(node.chunk_ids)
            for cid in old_ids_by_path.get(node.path, set()) - new_ids:
                self._tombstone(cid)
            for cid in new_ids - existing:
                chunk = self.file_chunks.get(cid)
                if chunk is not None and chunk.embedding is not None:
                    to_add.append(chunk)

        if to_add:
            vectors = np.stack([c.embedding for c in to_add])
            self._add_to_index([c.id for c in to_add], vectors)
        self._compact_if_needed()

    async def delete(self, path: str | list[str]) -> None:
        assert self.file_graph is not None
        paths = [path] if isinstance(path, str) else path
        nodes = await self.file_graph.get_nodes(paths)
        deleted_ids = [cid for n in nodes for cid in n.chunk_ids]
        await super().delete(path)
        if self._faiss_index is None:
            return
        for cid in deleted_ids:
            self._tombstone(cid)
        self._compact_if_needed()

    async def clear(self) -> None:
        await super().clear()
        self._faiss_index = self._new_index() if self.embedding_store is not None else None
        self._id_map = []
        self._id_to_row = {}
        self._tombstones.clear()
        self.faiss_path.unlink(missing_ok=True)
        self.faiss_idmap_path.unlink(missing_ok=True)

    # -- search -----------------------------------------------------------

    async def vector_search(self, query: str, limit: int, search_filter: dict) -> list[FileChunk]:
        if self.embedding_store is None or not query or self._faiss_index is None:
            return []
        if self._faiss_index.ntotal == 0:
            return []

        try:
            query_embedding = await self.embedding_store.get_embedding(query)
        except Exception as e:
            self._disable_embedding(f"search: {type(e).__name__}: {e}")
            return []
        if query_embedding is None:
            return []

        # Over-fetch by len(tombstones) so dropped rows can't starve the result set.
        q = self._prepare(query_embedding)
        k = min(self._faiss_index.ntotal, limit + len(self._tombstones))
        scores, rows = self._faiss_index.search(q, k)
        return self._collect_hits(rows[0].tolist(), scores[0].tolist(), limit)

    def _collect_hits(self, rows: list[int], scores: list[float], limit: int) -> list[FileChunk]:
        """Map raw FAISS rows back to chunks, skipping tombstones and stale ids."""
        results: list[FileChunk] = []
        for raw_row, score in zip(rows, scores):
            row = int(raw_row)
            if row < 0 or row in self._tombstones or row >= len(self._id_map):
                continue
            chunk = self.file_chunks.get(self._id_map[row])
            if chunk is None:
                continue
            results.append(chunk.model_copy(update={"scores": {"vector": float(score), "score": float(score)}}))
            if len(results) >= limit:
                break
        return results
