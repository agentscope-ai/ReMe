"""FAISS-backed file store: chunk JSONL stays authoritative; FAISS HNSW replaces the linear vector scan."""

import asyncio
import json
from contextlib import suppress
from uuid import uuid4

import aiofiles
import numpy as np

from .local_file_store import LocalFileStore
from ..component_registry import R
from ...schema import FileChunk, FileNode

_REINDEX_ADD_BATCH_SIZE = 1024


@R.register("faiss")
class FaissLocalFileStore(LocalFileStore):
    """LocalFileStore variant whose vector_search is backed by a FAISS IndexHNSWFlat.

    Chunk persistence is unchanged (JSONL, owned by the parent). FAISS state is
    stored alongside as a binary index plus an id-map sidecar. If either file
    is missing or stale, the index is rebuilt from ``self.file_chunks``, which
    remains the source of truth.

    HNSW parameters (``hnsw_m``, ``hnsw_ef_construction``) control graph
    connectivity and build-time quality.  ``efSearch`` is not a stored property;
    it is set to ``limit * 5`` at query time so the candidate pool scales with
    the number of results requested.  FAISS internally raises the beam width to
    ``max(efSearch, k)`` when ``k`` exceeds this value during progressive recall.

    Rebuilding an HNSW graph is expensive. When ``async_reindex`` is enabled the
    compaction rebuild is moved off the request path: the new index is built in a
    worker thread (FAISS releases the GIL during ``add``) from a snapshot while the
    current index keeps serving searches, then reconciled against concurrent writes
    and atomically swapped in. Only one reindex runs at a time and the most recent
    request wins. ``async_reindex`` defaults to ``False`` so behavior is unchanged
    unless explicitly opted in; the ``load`` and backfill rebuilds stay synchronous.

    faiss is imported lazily inside ``__init__`` so that merely importing this
    module (e.g. via ``reme version``) does not trigger the SWIG bindings and
    their associated DeprecationWarnings.
    """

    def __init__(
        self,
        normalize: bool = True,
        max_tombstones: int = 1024,
        hnsw_m: int = 64,
        hnsw_ef_construction: int = 64,
        async_reindex: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._faiss = self._import_faiss()
        self.normalize = normalize
        self.max_tombstones = max_tombstones
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.async_reindex = async_reindex
        self.faiss_path = self.component_metadata_path / f"faiss_index_{self.name}_{self.store_version}.bin"
        self.faiss_idmap_path = self.component_metadata_path / f"faiss_idmap_{self.name}_{self.store_version}.json"
        self._faiss_index = None  # faiss.Index | None
        self._id_map: list[str] = []  # row -> chunk_id
        self._id_to_row: dict[str, int] = {}  # chunk_id -> row (live entries only)
        self._tombstones: set[int] = set()  # rows whose chunk_id was deleted
        self._faiss_dump_lock = asyncio.Lock()
        # Async reindex machinery (only exercised when async_reindex is True).
        # ``_reindex_generation`` is the single source of truth for build validity:
        # a build is current iff its captured generation still matches. Superseding
        # or cancelling a reindex advances the generation, so an in-flight build
        # aborts at its next batch and can never be revived by a reset flag.
        self._reindex_task: asyncio.Task | None = None
        self._reindex_lock = asyncio.Lock()
        self._reindex_generation = 0
        self._closing = False  # set during _close() to stop spawning background reindexes

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
        index = self._faiss.IndexHNSWFlat(self._dim, self.hnsw_m, self._faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.hnsw_ef_construction
        index.hnsw.efSearch = 64  # safe default; overwritten at query time by _set_ef_search
        return index

    def _set_ef_search(self, limit: int) -> None:
        """Set HNSW efSearch to ``limit * 5`` for a good recall/speed balance.

        FAISS internally uses ``max(efSearch, k)`` during search, so when the
        over-fetch ``k`` exceeds this value the beam width is raised automatically.
        """
        self._faiss_index.hnsw.efSearch = limit * 5

    def _prepare(self, vec: np.ndarray) -> np.ndarray:
        """Cast to float32 (FAISS requirement) and L2-normalize so inner product gives cosine."""
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
        chunks = [c for c in self.file_chunks.values() if self._embedding_dim_matches(c.embedding)]
        if not chunks:
            return
        vectors = np.stack([c.embedding for c in chunks])
        self._add_to_index([c.id for c in chunks], vectors)

    def _compact_if_needed(self) -> None:
        if len(self._tombstones) < self.max_tombstones:
            return
        if not self.async_reindex:
            self._rebuild_index()
            return
        # Async mode: schedule a background rebuild. A reindex already running will
        # reconcile the latest chunks at swap time, so don't restart it on every
        # write (that would starve the build under sustained churn).
        if self._reindex_task is None or self._reindex_task.done():
            self._schedule_reindex()

    async def _after_embedding_backfill(self) -> None:
        """Make newly backfilled vectors visible to FAISS before persistence."""
        self._rebuild_index()

    # -- async reindex ----------------------------------------------------

    def _build_index_blocking(self, dim: int, vectors: "np.ndarray | None", gen: int):
        """Build a fresh HNSW index off the event loop (worker thread).

        FAISS releases the GIL during ``add``, so the event loop keeps serving
        searches on the current index while this runs. Between batches the build
        checks whether its ``gen`` is still the current ``_reindex_generation``;
        once superseded or cancelled the generation has advanced, so the build
        aborts early and returns ``None``. Because the generation only moves
        forward, a stale build can never be accidentally revived.
        """
        index = self._faiss.IndexHNSWFlat(dim, self.hnsw_m, self._faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.hnsw_ef_construction
        index.hnsw.efSearch = 64
        if vectors is None or vectors.size == 0:
            return index
        prepared = self._prepare(vectors)
        for start in range(0, prepared.shape[0], _REINDEX_ADD_BATCH_SIZE):
            if gen != self._reindex_generation:
                return None
            index.add(prepared[start : start + _REINDEX_ADD_BATCH_SIZE])
        return index

    def _schedule_reindex(self) -> None:
        """Start a background reindex, superseding any in-flight one.

        Only one reindex runs at a time and the most recent request wins: bumping
        ``_reindex_generation`` invalidates any running build (it aborts at its next
        batch) and the superseded task is cancelled before a fresh build starts.
        """
        if self._closing:
            return  # do not spawn background work while shutting down
        prev_task = self._reindex_task
        self._reindex_generation += 1
        gen = self._reindex_generation
        self._reindex_task = asyncio.ensure_future(self._reindex_runner(prev_task, gen))

    async def _reindex_runner(self, prev_task: "asyncio.Task | None", gen: int) -> None:
        """Serialize reindex builds: wait out the superseded task, then rebuild."""
        if prev_task is not None and not prev_task.done():
            prev_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await prev_task
        async with self._reindex_lock:
            if gen != self._reindex_generation:
                return  # superseded again while waiting for the lock
            try:
                await self._reindex_async(gen)
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception as e:  # pragma: no cover - defensive
                self.logger.exception(f"{self.name}: async reindex failed: {e}")

    async def _reindex_async(self, gen: int) -> None:
        """Build a new index from a snapshot without blocking searches, reconcile
        against concurrent writes, then atomically swap it in for the old index.
        """
        if self.embedding_store is None or self._dim == 0:
            return
        dim = self._dim
        items = [
            (cid, chunk.embedding)
            for cid, chunk in self.file_chunks.items()
            if self._embedding_dim_matches(chunk.embedding)
        ]
        snapshot_ids = [cid for cid, _ in items]
        snapshot_emb = dict(items)
        vectors = np.stack([emb for _, emb in items]) if items else None

        new_index = await asyncio.to_thread(self._build_index_blocking, dim, vectors, gen)
        if new_index is None or gen != self._reindex_generation:
            return  # superseded / cancelled: keep the current index serving

        # Atomic swap into the snapshot state (no await between assignments).
        self._faiss_index = new_index
        self._id_map = list(snapshot_ids)
        self._id_to_row = {cid: row for row, cid in enumerate(snapshot_ids)}
        self._tombstones = set()

        # Fold in writes that landed on the old index while the build ran.
        self._reconcile_after_swap(snapshot_emb)
        self.logger.info(f"Async reindex complete: {self._faiss_index.ntotal} rows, live={len(self._id_to_row)}")

    def _reconcile_after_swap(self, snapshot_emb: dict[str, np.ndarray]) -> None:
        """Bring the freshly swapped snapshot index in line with current chunks.

        Deleted-during-build ids become tombstones; changed embeddings are
        tombstoned and re-added; ids created during the build are appended. This
        reproduces exactly what a synchronous rebuild from ``file_chunks`` yields.
        """
        live_now = {
            cid: chunk for cid, chunk in self.file_chunks.items() if self._embedding_dim_matches(chunk.embedding)
        }
        to_add: list[FileChunk] = []
        for cid in list(self._id_to_row):
            chunk = live_now.get(cid)
            if chunk is None:
                self._tombstone(cid)  # deleted during build
            elif not np.array_equal(chunk.embedding, snapshot_emb.get(cid)):
                self._tombstone(cid)  # embedding/text changed during build
                to_add.append(chunk)
        for cid, chunk in live_now.items():
            if cid not in snapshot_emb:
                to_add.append(chunk)  # created during build
        if to_add:
            vectors = np.stack([chunk.embedding for chunk in to_add])
            self._add_to_index([chunk.id for chunk in to_add], vectors)

    async def _cancel_reindex(self) -> None:
        """Stop any in-flight reindex; used by close() and clear().

        Advancing the generation invalidates the running build so it aborts at its
        next batch; cancelling the task detaches the awaiting coroutine promptly.
        """
        self._reindex_generation += 1
        task = self._reindex_task
        self._reindex_task = None
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task

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
            # Reject legacy / incompatible index types (e.g. a pre-HNSW IndexFlatIP
            # sidecar) up front: they load fine and pass the dim/id_map checks but
            # lack the ``.hnsw`` attribute, so _set_ef_search would raise
            # AttributeError on the first vector_search. Failing here triggers a
            # clean rebuild into an IndexHNSWFlat instead.
            if not isinstance(index, self._faiss.IndexHNSWFlat):
                raise ValueError(f"FAISS index type {type(index).__name__} is not IndexHNSWFlat")
            if index.d != self._dim:
                raise ValueError(f"FAISS dim {index.d} != embedding dim {self._dim}")
            async with aiofiles.open(self.faiss_idmap_path, encoding=self.encoding) as f:
                data = json.loads(await f.read())
            id_map = list(data.get("id_map", []))
            if len(id_map) != index.ntotal:
                raise ValueError(f"id_map size {len(id_map)} != index ntotal {index.ntotal}")
            tombstones = {int(row) for row in data.get("tombstones", [])}
            if any(row < 0 or row >= len(id_map) for row in tombstones):
                raise ValueError("FAISS tombstones contain out-of-range rows")
            live_ids = [cid for i, cid in enumerate(id_map) if i not in tombstones]
            if len(live_ids) != len(set(live_ids)):
                raise ValueError("FAISS id_map contains duplicate live chunk ids")
            expected_ids = {
                cid for cid, chunk in self.file_chunks.items() if self._embedding_dim_matches(chunk.embedding)
            }
            if set(live_ids) != expected_ids:
                raise ValueError("FAISS sidecar live ids do not match persisted chunks")
            self._faiss_index = index
            self._id_map = id_map
            self._tombstones = tombstones
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
        async with self._faiss_dump_lock:
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
        token = uuid4().hex
        tmp_index = self.faiss_path.with_name(f".{self.faiss_path.name}.{token}.tmp")
        tmp_idmap = self.faiss_idmap_path.with_name(f".{self.faiss_idmap_path.name}.{token}.tmp")
        payload = json.dumps({"id_map": list(self._id_map), "tombstones": sorted(self._tombstones)})
        try:
            self._faiss.write_index(self._faiss_index, str(tmp_index))
            async with aiofiles.open(tmp_idmap, "w", encoding=self.encoding) as f:
                await f.write(payload)

            # Publish only after both parts of the sidecar have been written successfully.
            tmp_index.replace(self.faiss_path)
            tmp_idmap.replace(self.faiss_idmap_path)
        finally:
            tmp_index.unlink(missing_ok=True)
            tmp_idmap.unlink(missing_ok=True)

    # -- CRUD overrides ---------------------------------------------------

    async def upsert(self, files: list[tuple[FileNode, list[FileChunk]]]) -> None:
        if not files:
            return
        assert self.file_graph is not None

        # Snapshot pre-upsert chunk_ids so we can diff against the post-upsert state.
        old_nodes = await self.file_graph.get_nodes([node.path for node, _ in files])
        old_ids_by_path = {n.path: set(n.chunk_ids) for n in old_nodes}
        old_text_by_id = {
            cid: chunk.text
            for n in old_nodes
            for cid in n.chunk_ids
            if (chunk := self.file_chunks.get(cid)) is not None
        }
        await super().upsert(files)

        if self._faiss_index is None or self.embedding_store is None:
            return
        self._sync_index_after_upsert(files, old_ids_by_path, old_text_by_id)

    def _sync_index_after_upsert(
        self,
        files: list[tuple[FileNode, list[FileChunk]]],
        old_ids_by_path: dict[str, set[str]],
        old_text_by_id: dict[str, str],
    ) -> None:
        """Apply add/tombstone deltas to FAISS based on chunk_id set differences."""
        existing = set(self._id_to_row)
        to_add: list[FileChunk] = []
        for node, _ in files:
            new_ids = set(node.chunk_ids)
            for cid in old_ids_by_path.get(node.path, set()) - new_ids:
                self._tombstone(cid)
            for cid in new_ids:
                chunk = self.file_chunks.get(cid)
                if chunk is None or not self._embedding_dim_matches(chunk.embedding):
                    continue
                if cid in existing and old_text_by_id.get(cid) == chunk.text:
                    continue
                # Reaching here means the chunk is new or its text changed; the old
                # row (if any) is tombstoned and the fresh vector is re-added.
                if cid in existing:
                    self._tombstone(cid)
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
        await self._delete_nodes(nodes)  # reuse resolved nodes; avoids a second get_nodes
        if self._faiss_index is None:
            return
        for cid in deleted_ids:
            self._tombstone(cid)
        self._compact_if_needed()

    async def _close(self) -> None:
        """Cancel any in-flight reindex before the parent persists and tears down.

        ``_closing`` is set first so the parent's final ``dump`` cannot re-schedule a
        background reindex (which would leak as an orphan task).
        """
        self._closing = True
        await self._cancel_reindex()
        await super()._close()

    async def clear(self) -> None:
        # Serialize with dump so a concurrent _write_sidecar cannot re-create the
        # sidecar files we are about to unlink, or persist a half-reset index.
        # Cancel the reindex *inside* the lock: a dump holding the lock can schedule
        # a fresh reindex via _compact_if_needed, so cancelling before acquiring the
        # lock would let that task leak and run against the just-cleared state. The
        # reindex path only takes _reindex_lock (never _faiss_dump_lock), so
        # cancelling while holding _faiss_dump_lock cannot deadlock.
        async with self._faiss_dump_lock:
            await self._cancel_reindex()
            await super().clear()
            self._faiss_index = self._new_index() if self.embedding_store is not None else None
            self._id_map = []
            self._id_to_row = {}
            self._tombstones.clear()
            self.faiss_path.unlink(missing_ok=True)
            self.faiss_idmap_path.unlink(missing_ok=True)

    # -- search -----------------------------------------------------------

    async def vector_search(self, query: str, limit: int, search_filter: dict) -> list[FileChunk]:
        if (
            self.embedding_store is None
            or not query
            or limit <= 0
            or self._faiss_index is None
            or self._faiss_index.ntotal == 0
        ):
            return []

        try:
            query_embedding = await self.embedding_store.get_embedding(query)
        except Exception as e:
            self._disable_embedding(f"search: {type(e).__name__}: {e}")
            return []
        if query_embedding is None:
            return []
        if not self._embedding_dim_matches(query_embedding):
            self._disable_embedding(
                f"search: query embedding dimension {len(query_embedding)} != {self.embedding_store.dimensions}",
            )
            return []

        q = self._prepare(query_embedding)
        ntotal = self._faiss_index.ntotal

        if not search_filter:
            # No filter: simple over-fetch to cover tombstones.
            k = min(ntotal, limit + len(self._tombstones))
            self._set_ef_search(limit)
            scores, rows = self._faiss_index.search(q, k)
            return self._collect_hits(rows[0].tolist(), scores[0].tolist(), limit, search_filter)

        # With filter: progressively increase k until we collect enough results
        # or exhaust the entire index.
        k = min(ntotal, 3 * limit)
        while True:
            self._set_ef_search(limit)
            scores, rows = self._faiss_index.search(q, k)
            results = self._collect_hits(rows[0].tolist(), scores[0].tolist(), limit, search_filter)
            if len(results) >= limit or k >= ntotal:
                return results
            k = min(ntotal, k * 2)

    def _collect_hits(
        self,
        rows: list[int],
        scores: list[float],
        limit: int,
        search_filter: dict | None = None,
    ) -> list[FileChunk]:
        """Map raw FAISS rows back to chunks, skipping tombstones and stale ids."""
        results: list[FileChunk] = []
        for raw_row, score in zip(rows, scores):
            row = int(raw_row)
            if row < 0 or row in self._tombstones or row >= len(self._id_map):
                continue
            chunk = self.file_chunks.get(self._id_map[row])
            if chunk is None or not self._matches_search_filter(chunk, search_filter):
                continue
            results.append(chunk.model_copy(update={"scores": {"vector": float(score), "score": float(score)}}))
            if len(results) >= limit:
                break
        return results
