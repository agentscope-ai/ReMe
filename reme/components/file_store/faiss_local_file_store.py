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
    rebuild is moved off the request path as a submit-a-flag job: mutations set a
    boolean request flag and a single long-lived worker coroutine consumes it,
    building the new index in a worker thread (FAISS releases the GIL during
    ``add``) from a snapshot while the current index keeps serving searches, then
    atomically swaps it in. Exactly one worker runs, so only one reindex proceeds
    at a time; repeated submissions coalesce into the flag, and writes that land
    during a build re-arm it so a follow-up rebuild folds them in (eventually
    consistent). ``async_reindex`` defaults to ``False`` so behavior is unchanged
    unless opted in; ``load`` (no old index to serve) always rebuilds inline, and
    the backfill rebuild follows ``async_reindex``.

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
        # Submitting a rebuild just sets ``_reindex_event``; a single long-lived
        # worker coroutine consumes it, so at most one reindex ever runs at a time
        # and repeated submissions coalesce into the boolean flag.
        self._reindex_event = asyncio.Event()  # set == "rebuild requested"
        self._reindex_worker_task: asyncio.Task | None = None
        self._reindex_busy = False  # True while a build is in flight (single worker)
        self._index_writes = 0  # bumped on every index mutation; used to re-arm the flag
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

    def _set_ef_search(self, index, limit: int) -> None:
        """Set HNSW efSearch to ``limit * 5`` for a good recall/speed balance.

        FAISS internally uses ``max(efSearch, k)`` during search, so when the
        over-fetch ``k`` exceeds this value the beam width is raised automatically.
        """
        index.hnsw.efSearch = limit * 5

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
        self._index_writes += 1

    def _tombstone(self, chunk_id: str) -> None:
        row = self._id_to_row.pop(chunk_id, None)
        if row is not None:
            self._tombstones.add(row)
            self._index_writes += 1

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
        if self.async_reindex:
            self._submit_reindex()  # flag the worker; it coalesces repeated requests
        else:
            self._rebuild_index()

    async def _after_embedding_backfill(self) -> None:
        """Make newly backfilled vectors visible to FAISS.

        Synchronous by default so the vectors are indexed before the caller's dump.
        When ``async_reindex`` is enabled the rebuild is submitted to the worker
        (eventually consistent); the chunk JSONL stays authoritative, so a lagging
        sidecar self-heals on the next load.
        """
        if self.async_reindex:
            self._submit_reindex()
        else:
            self._rebuild_index()

    # -- async reindex ----------------------------------------------------

    def _submit_reindex(self) -> None:
        """Submit a rebuild request: ensure the worker exists and raise the flag.

        Idempotent -- the flag is a boolean, so repeated submissions while a build
        is running or pending collapse into (at most) one follow-up rebuild.
        Setting the flag is the only way to submit; the worker owns execution.
        """
        if self._closing:
            return  # do not spawn background work while shutting down
        self._ensure_reindex_worker()
        self._reindex_event.set()

    def _ensure_reindex_worker(self) -> None:
        """Start the single long-lived reindex worker if it is not running."""
        if self._reindex_worker_task is None or self._reindex_worker_task.done():
            self._reindex_worker_task = asyncio.ensure_future(self._reindex_worker())

    async def _reindex_worker(self) -> None:
        """Single consumer of reindex requests.

        Because exactly one worker drains the flag, only one reindex ever runs at a
        time. The flag is cleared before snapshotting, so any write that lands
        during the build re-arms it (see ``_reindex_async``) and triggers exactly
        one more rebuild afterwards -- the system converges without reconciliation.
        """
        while not self._closing:
            await self._reindex_event.wait()
            # Mark busy before clearing the flag (no await in between) so an
            # observer can never see "flag clear and not busy" mid-handoff.
            self._reindex_busy = True
            self._reindex_event.clear()
            if self._closing:
                self._reindex_busy = False
                return
            try:
                await self._reindex_async()
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception as e:  # pragma: no cover - defensive
                self.logger.exception(f"{self.name}: async reindex failed: {e}")
            finally:
                self._reindex_busy = False

    async def _reindex_async(self) -> None:
        """Build a new index from a snapshot without blocking searches, then swap it
        in atomically. Writes that land during the build bump ``_index_writes``; if
        the counter moved we re-arm the flag so the next round folds them in.
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
        vectors = np.stack([emb for _, emb in items]) if items else None
        writes_at_snapshot = self._index_writes  # captured with the snapshot (no await yet)

        new_index = await asyncio.to_thread(self._build_index_blocking, dim, vectors)

        # Atomic swap into the snapshot state (no await between assignments) so a
        # concurrent search never observes a torn index / id_map / tombstone triple.
        self._faiss_index = new_index
        self._id_map = list(snapshot_ids)
        self._id_to_row = {cid: row for row, cid in enumerate(snapshot_ids)}
        self._tombstones = set()

        # Writes landed on the old index while we built; a full rebuild is the
        # simplest way to fold them in, so request exactly one more round.
        if self._index_writes != writes_at_snapshot:
            self._reindex_event.set()
        self.logger.info(f"Async reindex complete: {self._faiss_index.ntotal} rows, live={len(self._id_to_row)}")

    def _build_index_blocking(self, dim: int, vectors: "np.ndarray | None"):
        """Build a fresh HNSW index off the event loop (worker thread).

        FAISS releases the GIL during ``add``, so the event loop keeps serving
        searches on the current index while this runs.
        """
        index = self._faiss.IndexHNSWFlat(dim, self.hnsw_m, self._faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.hnsw_ef_construction
        index.hnsw.efSearch = 64
        if vectors is not None and vectors.size:
            index.add(self._prepare(vectors))
        return index

    async def _stop_reindex_worker(self) -> None:
        """Stop the reindex worker; used by close() and clear().

        Cancelling detaches the awaiting coroutine promptly. An orphaned build
        thread (if any) finishes into a local index that is simply discarded, so no
        shared state is corrupted.
        """
        self._reindex_event.clear()
        task = self._reindex_worker_task
        self._reindex_worker_task = None
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
        self._reindex_busy = False

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
        """Stop any in-flight reindex before the parent persists and tears down.

        ``_closing`` is set first so the parent's final ``dump`` cannot submit a new
        background reindex (which would leak as an orphan task).
        """
        self._closing = True
        await self._stop_reindex_worker()
        await super()._close()

    async def clear(self) -> None:
        # Serialize with dump so a concurrent _write_sidecar cannot re-create the
        # sidecar files we are about to unlink, or persist a half-reset index.
        # Stop the reindex *inside* the lock: a dump holding the lock can submit a
        # fresh reindex via _compact_if_needed, so stopping before acquiring the
        # lock would let that worker run against the just-cleared state. The worker
        # never takes _faiss_dump_lock, so stopping it while holding the lock cannot
        # deadlock.
        async with self._faiss_dump_lock:
            await self._stop_reindex_worker()
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
        if query_embedding is None or not self._embedding_dim_matches(query_embedding):
            if query_embedding is not None:
                self._disable_embedding(
                    f"search: query embedding dimension {len(query_embedding)} != {self.embedding_store.dimensions}",
                )
            return []

        # get_embedding above yielded control; a concurrent clear() drops the
        # index to None once embedding is disabled, and a reindex may have swapped
        # it out. Re-read after the await before dereferencing ``.ntotal`` so we
        # never touch a None/emptied index (mirrors the entry guard).
        index = self._faiss_index
        if index is None or index.ntotal == 0:
            return []

        q = self._prepare(query_embedding)
        ntotal = index.ntotal

        if not search_filter:
            # No filter: simple over-fetch to cover tombstones.
            k = min(ntotal, limit + len(self._tombstones))
            self._set_ef_search(index, limit)
            scores, rows = index.search(q, k)
            return self._collect_hits(rows[0].tolist(), scores[0].tolist(), limit, search_filter)

        # With filter: progressively increase k until we collect enough results
        # or exhaust the entire index.
        k = min(ntotal, 3 * limit)
        while True:
            self._set_ef_search(index, limit)
            scores, rows = index.search(q, k)
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
