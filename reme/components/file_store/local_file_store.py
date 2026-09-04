"""In-memory file store with compressed JSONL persistence on close."""

import asyncio
import base64
import datetime
import heapq
import json
import time
from collections.abc import Iterable
from contextlib import suppress

import numpy as np

from .base_file_store import BaseFileStore
from ..component_registry import R
from ..embedding_store import BaseEmbeddingStore
from ..file_graph import BaseFileGraph
from ..keyword_index import BaseKeywordIndex
from ..tag_index import BaseTagIndex
from ...enumeration import LinkScopeEnum
from ...schema import FileChunk, FileLink, FileNode
from ...utils import batch_cosine_similarity
from ...utils.async_utils import complete_in_thread
from ...utils.jsonl_zst import read_jsonl_zst, write_jsonl_zst

CachedEmbedding = tuple[str, np.ndarray]
_EMBEDDING_F16_B64_FIELD = "_embedding_f16_b64"
_EMBEDDING_F16_DTYPE = np.dtype("<f2")
_VECTOR_SEARCH_BATCH_SIZE = 1024
_PROGRESS_LOG_PERCENT_STEP = 10
_KEYWORD_REBUILD_BATCH_SIZE = 200


@R.register("local")
class LocalFileStore(BaseFileStore):
    """In-memory file store with deferred JSONL persistence.

    Composes ``embedding_store`` for vector retrieval, ``keyword_index`` for
    full-text retrieval, ``file_graph`` for node/link storage, and an optional
    ``tag_index`` derived from file-node frontmatter. ``file_graph`` is mandatory;
    at least one of embedding / keyword must be present.
    """

    def __init__(
        self,
        embedding_store: str = "default",
        keyword_index: str = "default",
        file_graph: str = "default",
        tag_index: str = "",
        encoding: str = "utf-8",
        store_version: str = "v1",
        embedding_rebuild_required: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        from ..embedding_store import LocalEmbeddingStore
        from ..file_graph import LocalFileGraph
        from ..keyword_index import BM25Index
        from ..tag_index import LocalTagIndex

        if not embedding_store and not keyword_index:
            raise ValueError("At least one of embedding_store or keyword_index must be set.")
        if not file_graph:
            raise ValueError("file_graph is required for LocalFileStore.")

        self.embedding_store = self.bind(embedding_store, BaseEmbeddingStore, default_factory=LocalEmbeddingStore)
        self.keyword_index = self.bind(keyword_index, BaseKeywordIndex, default_factory=BM25Index)
        self.file_graph = self.bind(file_graph, BaseFileGraph, default_factory=LocalFileGraph)
        self.tag_index = self.bind(tag_index, BaseTagIndex, default_factory=LocalTagIndex)

        self.encoding = encoding
        self.store_version = store_version
        self.file_chunks: dict[str, FileChunk] = {}
        self.chunks_path = self.component_metadata_path / f"file_chunks_{self.name}_{self.store_version}.jsonl.zst"
        self._embedding_backfill_task: asyncio.Task | None = None
        self._embedding_backfill_pending = False
        self._embedding_rebuild_pending = bool(embedding_rebuild_required)
        self._embedding_space_generation = 0
        self._mutation_generation = 0
        self._tag_index_rebuild_required = False
        self._closing = False

    # -- lifecycle ------------------------------------------------------------

    async def _start(self) -> None:
        self._closing = False
        started_at = time.monotonic()
        self.component_metadata_path.mkdir(parents=True, exist_ok=True)
        await super()._start()

        load_started_at = time.monotonic()
        await self.load()
        self.logger.info(
            f"{self.name}: file store load complete: chunks={len(self.file_chunks)}, "
            f"elapsed={time.monotonic() - load_started_at:.3f}s",
        )

        backfill_started_at = time.monotonic()
        self._start_embedding_backfill()
        self.logger.info(
            f"{self.name}: embedding backfill scheduling complete: "
            f"scheduled={self._embedding_backfill_task is not None}, "
            f"elapsed={time.monotonic() - backfill_started_at:.3f}s",
        )
        self.logger.info(
            f"{self.name}: file store startup complete: " f"elapsed={time.monotonic() - started_at:.3f}s",
        )

    async def _close(self) -> None:
        self._closing = True
        await self._cancel_embedding_backfill()
        # Dependencies are closed separately by Application (reverse
        # topological order) or BaseComponent (owned standalone dependencies).
        # Persist only this store's local state here so each component writes
        # exactly once during shutdown. Preserve the historical dump() hook for
        # third-party subclasses that override it to write additional state.
        if self.is_started:
            if type(self).dump is LocalFileStore.dump:
                await self._dump_owned_state()
            else:
                await self.dump()
        self.file_chunks.clear()
        await super()._close()

    def _mark_embedding_unhealthy(self, reason: str) -> None:
        """Record a temporary provider failure while preserving the component."""
        if self.embedding_store is None:
            return
        self.embedding_store.is_healthy = False
        self.logger.error(f"{self.name}: embedding unavailable, {reason}; keyword search remains active")

    async def _recover_after_real_request(self, was_healthy: bool) -> None:
        """Schedule repair when a real, non-cache provider request recovers."""
        if (
            self.embedding_store is None
            or self._embedding_rebuild_pending
            or was_healthy
            or not getattr(self.embedding_store, "is_healthy", True)
        ):
            return
        self.logger.info(f"{self.name}: embedding provider recovered; scheduling missing-vector backfill")
        await self.resume_embedding(verified=True)

    async def resume_embedding(self, *, verified: bool = False) -> bool:
        """Resume same-vector-space repair after provider recovery."""
        if self.embedding_store is None or self._closing:
            return False
        if verified:
            self.embedding_store.is_healthy = True
        if self._embedding_rebuild_pending:
            return True
        self._start_embedding_backfill(skip_health_check=verified)
        return True

    def _embedding_dim_matches(self, embedding: np.ndarray | None) -> bool:
        """Return whether an index embedding matches the active embedding model."""
        # With no active embedding store, no persisted/index vector is trustworthy.
        if self.embedding_store is None or embedding is None:
            return False
        return len(embedding) == self.embedding_store.dimensions

    def _drop_stale_embedding(self, chunk: FileChunk, context: str) -> bool:
        """Drop a chunk embedding when it does not match the active model dimension."""
        if self.embedding_store is None:
            return False
        if chunk.embedding is None or self._embedding_dim_matches(chunk.embedding):
            return False
        self.logger.warning(
            f"{self.name}: stale embedding for chunk {chunk.id} during {context}: "
            f"{len(chunk.embedding)} != {self.embedding_store.dimensions}; re-embedding",
        )
        chunk.embedding = None
        return True

    def _drop_stale_embeddings(self, chunks: Iterable[FileChunk], context: str) -> None:
        for chunk in chunks:
            self._drop_stale_embedding(chunk, context)

    def _embedding_request_is_current(
        self,
        embedding_store: BaseEmbeddingStore,
        embedding_generation: int,
    ) -> bool:
        """Return whether an in-flight provider request still belongs to the active vector space."""
        return (
            not self._embedding_rebuild_pending
            and embedding_generation == self._embedding_space_generation
            and embedding_store is self.embedding_store
        )

    async def _get_query_embedding(self, query: str) -> np.ndarray | None:
        """Embed a query only while its provider and vector space remain current."""
        embedding_store = self.embedding_store
        if embedding_store is None or self._embedding_rebuild_pending or not query:
            return None

        embedding_generation = self._embedding_space_generation
        was_healthy = bool(getattr(embedding_store, "is_healthy", True))
        try:
            query_embedding = await embedding_store.get_embedding(query)
        except Exception as e:
            if self._embedding_request_is_current(embedding_store, embedding_generation):
                self._mark_embedding_unhealthy(f"search: {type(e).__name__}: {e}")
            return None

        if query_embedding is None or not self._embedding_request_is_current(
            embedding_store,
            embedding_generation,
        ):
            return None
        if not self._embedding_dim_matches(query_embedding):
            self._mark_embedding_unhealthy(
                f"search: query embedding dimension {len(query_embedding)} != {embedding_store.dimensions}",
            )
            return None
        await self._recover_after_real_request(was_healthy)
        return query_embedding if self._embedding_request_is_current(embedding_store, embedding_generation) else None

    # -- persistence ----------------------------------------------------------

    async def load(self) -> None:
        """Load chunks from the JSONL file into memory; missing file is a no-op."""
        started_at = time.monotonic()
        chunk_load_started_at = time.monotonic()
        if self.chunks_path.exists():
            try:
                for line in read_jsonl_zst(self.chunks_path, self.encoding):
                    line = line.strip()
                    if line:
                        chunk = self._deserialize_chunk(line)
                        self.file_chunks[chunk.id] = chunk
                self.logger.info(f"Loaded {len(self.file_chunks)} chunks from {self.chunks_path}")
            except Exception as e:
                self.logger.exception(f"Failed to load {self.chunks_path}: {e}")
        self.logger.info(
            f"{self.name}: chunk store load complete: chunks={len(self.file_chunks)}, "
            f"elapsed={time.monotonic() - chunk_load_started_at:.3f}s",
        )

        graph_repair_started_at = time.monotonic()
        graph_repaired = await self._repair_graph_chunk_consistency()
        self.logger.info(
            f"{self.name}: graph consistency check complete: repaired={graph_repaired}, "
            f"elapsed={time.monotonic() - graph_repair_started_at:.3f}s",
        )

        tag_sync_started_at = time.monotonic()
        tag_synced = await self._rebuild_tag_index("startup synchronization")
        self.logger.info(
            f"{self.name}: tag index sync complete: enabled={self.tag_index is not None}, "
            f"healthy={tag_synced}, "
            f"elapsed={time.monotonic() - tag_sync_started_at:.3f}s",
        )

        keyword_sync_started_at = time.monotonic()
        await self._sync_keyword_index_from_chunks()
        keyword_backend = type(self.keyword_index).__name__ if self.keyword_index is not None else "disabled"
        keyword_docs = getattr(self.keyword_index, "n_docs", 0) if self.keyword_index is not None else 0
        self.logger.info(
            f"{self.name}: BM25/keyword index sync complete: backend={keyword_backend}, docs={keyword_docs}, "
            f"elapsed={time.monotonic() - keyword_sync_started_at:.3f}s",
        )
        self.logger.info(
            f"{self.name}: file store load phases complete: " f"elapsed={time.monotonic() - started_at:.3f}s",
        )

    async def _repair_graph_chunk_consistency(self) -> bool:
        """Clear torn graph/chunk state so the filesystem scan rebuilds it.

        ``InitChangesStep`` uses file-graph nodes as the indexed-file snapshot.
        A graph that survives a missing, truncated, or stale chunk store would
        otherwise make the source files look up to date and permanently hide
        the broken search index.
        """
        assert self.file_graph is not None
        nodes = await self.file_graph.get_nodes()
        graph_chunk_ids = {chunk_id for node in nodes for chunk_id in node.chunk_ids}
        stored_chunk_ids = set(self.file_chunks)
        missing = graph_chunk_ids - stored_chunk_ids
        orphaned = stored_chunk_ids - graph_chunk_ids
        if not missing and not orphaned:
            return False

        self.logger.warning(
            f"{self.name}: graph/chunk mismatch: nodes={len(nodes)}, graph_chunks={len(graph_chunk_ids)}, "
            f"stored_chunks={len(stored_chunk_ids)}, missing={len(missing)}, orphaned={len(orphaned)}; "
            "clearing derived index state for automatic rebuild",
        )
        # Clearing graph nodes is required: the next InitChangesStep scan will
        # then classify every watched source file as added and rebuild graph,
        # chunks, and search indexes from the user-owned files.
        await self.clear()
        return True

    @staticmethod
    def _deserialize_chunk(line: str) -> FileChunk:
        """Read compact vectors while retaining legacy JSON-list compatibility."""
        payload = json.loads(line)
        encoded = payload.pop(_EMBEDDING_F16_B64_FIELD, None)
        if encoded is not None:
            raw = base64.b64decode(encoded, validate=True)
            if len(raw) % _EMBEDDING_F16_DTYPE.itemsize:
                raise ValueError("Invalid float16 embedding byte length")
            payload["embedding"] = np.frombuffer(raw, dtype=_EMBEDDING_F16_DTYPE)
        return FileChunk.model_validate(payload)

    @staticmethod
    def _serialize_chunk(chunk: FileChunk) -> str:
        """Serialize embeddings without expanding float16 values into Python floats."""
        payload = chunk.model_dump(mode="json", exclude={"embedding"})
        if chunk.embedding is not None:
            embedding = np.asarray(chunk.embedding, dtype=_EMBEDDING_F16_DTYPE)
            if embedding.ndim != 1:
                raise ValueError("FileChunk embedding must be one-dimensional")
            raw = np.ascontiguousarray(embedding).tobytes()
            payload[_EMBEDDING_F16_B64_FIELD] = base64.b64encode(raw).decode("ascii")
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _invalidate_stale_embeddings(self) -> None:
        """Drop persisted embeddings whose dimension no longer matches the active model."""
        if self.embedding_store is None:
            return
        self._drop_stale_embeddings(self.file_chunks.values(), "load")

    def _start_embedding_backfill(self, *, skip_health_check: bool = False) -> None:
        """Schedule startup embedding repair without delaying component readiness."""
        started_at = time.monotonic()
        if self._closing or self._embedding_rebuild_pending:
            reason = "closing" if self._closing else "manual_reindex_required"
            self.logger.info(f"{self.name}: embedding backfill skipped: reason={reason}")
            return
        if not self.embedding_store:
            self.logger.info(
                f"{self.name}: embedding backfill skipped: reason=embedding_disabled, "
                f"elapsed={time.monotonic() - started_at:.3f}s",
            )
            return
        if self._embedding_backfill_task is not None and not self._embedding_backfill_task.done():
            self._embedding_backfill_pending |= skip_health_check
            self.logger.info(
                f"{self.name}: embedding backfill scheduling skipped: reason=already_running, "
                f"elapsed={time.monotonic() - started_at:.3f}s",
            )
            return
        if not self.file_chunks:
            self.logger.info(
                f"{self.name}: embedding backfill skipped: reason=no_chunks, "
                f"elapsed={time.monotonic() - started_at:.3f}s",
            )
            return
        self._embedding_backfill_task = asyncio.create_task(
            self._run_embedding_backfill(skip_health_check=skip_health_check),
            name=f"embedding-backfill:{self.name}",
        )
        self.logger.info(
            f"{self.name}: embedding backfill scheduled: chunks={len(self.file_chunks)}, "
            f"elapsed={time.monotonic() - started_at:.3f}s",
        )

    async def require_embedding_rebuild(self) -> None:
        """Make all existing vectors unavailable without rebuilding them."""
        self._embedding_space_generation += 1
        self._embedding_rebuild_pending = True
        await self._cancel_embedding_backfill()

    @BaseFileStore.serialized
    async def reindex(self, scope: str) -> dict:
        """Rebuild derived search indexes from ``file_chunks`` without touching files or the graph."""
        if scope not in {"all", "bm25", "embedding"}:
            raise ValueError("reindex scope must be one of: all, bm25, embedding")
        if scope == "bm25":
            return await self._reindex_bm25()
        if scope == "embedding":
            return await self._reindex_embedding()
        return {
            "scope": "all",
            "bm25": await self._reindex_bm25(),
            "embedding": await self._reindex_embedding(),
        }

    async def _reindex_bm25(self) -> dict:
        if self.keyword_index is None:
            return {"indexed": 0, "scope": "bm25"}
        while True:
            generation = self._mutation_generation
            docs = {chunk_id: chunk.text for chunk_id, chunk in self.file_chunks.items() if chunk.text}
            await self._rebuild_keyword_index(docs)
            if generation == self._mutation_generation:
                return {"indexed": len(docs), "scope": "bm25"}

    async def _reindex_embedding(self) -> dict:
        await self.require_embedding_rebuild()
        try:
            while True:
                generation = self._mutation_generation
                embedding_generation = self._embedding_space_generation
                chunks = tuple(self.file_chunks.values())
                for start in range(0, len(chunks), 512):
                    for chunk in chunks[start : start + 512]:
                        chunk.embedding = None
                    await asyncio.sleep(0)
                await self._reset_vector_index()
                if self.embedding_store is not None:
                    await self._backfill_missing_embeddings_inner(
                        skip_health_check=False,
                        started_at=time.monotonic(),
                    )
                    missing = [
                        chunk.id
                        for chunk in self.file_chunks.values()
                        if chunk.text and not self._embedding_dim_matches(chunk.embedding)
                    ]
                    if missing:
                        raise RuntimeError(f"embedding reindex incomplete: {len(missing)} chunks failed")
                    await self._finalize_embedding_reindex()
                await self._dump_owned_state()
                if self.embedding_store is not None:
                    await self.embedding_store.dump()
                if generation == self._mutation_generation and embedding_generation == self._embedding_space_generation:
                    self._embedding_rebuild_pending = False
                    indexed = len(chunks) if self.embedding_store is not None else 0
                    return {"indexed": indexed, "scope": "embedding"}
        except BaseException:
            self._embedding_rebuild_pending = True
            raise

    async def _run_embedding_backfill(self, *, skip_health_check: bool) -> None:
        """Run one repair and honor a verified request queued behind it."""
        current_task = asyncio.current_task()
        try:
            await self._backfill_missing_embeddings(skip_health_check=skip_health_check)
        finally:
            if self._embedding_backfill_task is current_task:
                self._embedding_backfill_task = None
            pending = self._embedding_backfill_pending
            self._embedding_backfill_pending = False
            if pending and not self._closing and self.embedding_store is not None:
                self.embedding_store.is_healthy = True
                self._start_embedding_backfill(skip_health_check=True)

    async def _cancel_embedding_backfill(self) -> None:
        """Cancel and collect the startup repair task during component shutdown."""
        task = self._embedding_backfill_task
        self._embedding_backfill_task = None
        self._embedding_backfill_pending = False
        if task is None:
            return
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            self.logger.exception(f"{self.name}: embedding backfill task failed during shutdown")

    def _log_progress(self, operation: str, current: int, total: int, next_percent: int) -> int:
        """Log progress at fixed percentage boundaries and return the next boundary."""
        if total <= 0:
            return 100
        percent = min(100, current * 100 // total)
        if current < total and percent < next_percent:
            return next_percent
        self.logger.info(f"{self.name}: {operation} progress: {current}/{total} ({percent}%)")
        while next_percent <= percent:
            next_percent += _PROGRESS_LOG_PERCENT_STEP
        return next_percent

    async def _backfill_missing_embeddings(self, *, skip_health_check: bool = False) -> None:
        """Background-repair persisted chunks that do not have usable vectors."""
        started_at = time.monotonic()
        await self._backfill_missing_embeddings_inner(
            skip_health_check=skip_health_check,
            started_at=started_at,
        )

    async def _backfill_missing_embeddings_inner(self, *, skip_health_check: bool, started_at: float) -> None:
        """Perform one backfill pass; the caller owns rebuild finalization."""
        if not self.embedding_store or not self.file_chunks:
            self.logger.info(
                f"{self.name}: embedding backfill finished without work: "
                f"elapsed={time.monotonic() - started_at:.3f}s",
            )
            return

        scan_started_at = time.monotonic()
        self._invalidate_stale_embeddings()
        missing = [chunk for chunk in self.file_chunks.values() if chunk.text and chunk.embedding is None]
        self.logger.info(
            f"{self.name}: embedding backfill scan complete: chunks={len(self.file_chunks)}, "
            f"missing={len(missing)}, elapsed={time.monotonic() - scan_started_at:.3f}s",
        )
        if not missing:
            self.logger.info(
                f"{self.name}: embedding backfill complete: filled=0/0, "
                f"elapsed={time.monotonic() - started_at:.3f}s",
            )
            return

        total = len(missing)
        batch_size = max(1, int(getattr(self.embedding_store, "max_batch_size", 10)))
        self.logger.info(f"{self.name}: embedding backfill started: total={total}, batch_size={batch_size}")
        try:
            if not skip_health_check:
                health_check_started_at = time.monotonic()
                is_healthy = await self.embedding_store.health_check()
                self.logger.info(
                    f"{self.name}: embedding health check complete: healthy={is_healthy}, "
                    f"elapsed={time.monotonic() - health_check_started_at:.3f}s",
                )
                if not is_healthy:
                    self.logger.warning(
                        f"{self.name}: embedding backfill skipped: processed=0/{total}, reason=health check failed",
                    )
                    return

            processed = 0
            batch_count = 0
            embedding_started_at = time.monotonic()
            next_percent = _PROGRESS_LOG_PERCENT_STEP
            for start in range(0, total, batch_size):
                batch = missing[start : start + batch_size]
                await self.embedding_store.get_node_embeddings(batch)
                self._drop_stale_embeddings(batch, "backfill")
                processed += len(batch)
                batch_count += 1
                next_percent = self._log_progress("embedding backfill", processed, total, next_percent)
            self.logger.info(
                f"{self.name}: embedding batches complete: processed={processed}/{total}, "
                f"batches={batch_count}, elapsed={time.monotonic() - embedding_started_at:.3f}s",
            )
        except asyncio.CancelledError:
            elapsed = time.monotonic() - started_at
            self.logger.info(
                f"{self.name}: embedding backfill cancelled: processed={processed if 'processed' in locals() else 0}/"
                f"{total}, elapsed={elapsed:.2f}s",
            )
            raise

        except Exception as e:
            self._mark_embedding_unhealthy(f"backfill: {type(e).__name__}: {e}")
            elapsed = time.monotonic() - started_at
            self.logger.exception(
                f"{self.name}: embedding backfill failed: processed={processed if 'processed' in locals() else 0}/"
                f"{total}, elapsed={elapsed:.2f}s",
            )
            return

        filled = sum(1 for chunk in missing if chunk.embedding is not None)
        elapsed = time.monotonic() - started_at
        self.logger.info(
            f"{self.name}: embedding backfill complete: filled={filled}/{total}, elapsed={elapsed:.2f}s",
        )
        if filled and not self._embedding_rebuild_pending:
            try:
                await self._after_embedding_backfill()
                await self.dump()
            except Exception:
                self.logger.exception(f"{self.name}: failed to persist completed embedding backfill")

    async def _reset_vector_index(self) -> None:
        """Drop a derived vector index before rebuilding a changed vector space."""

    async def _after_embedding_backfill(self) -> None:
        """Backend hook for refreshing derived vector indexes after backfill."""

    async def _finalize_embedding_reindex(self) -> None:
        """Synchronously publish derived indexes for an explicit reindex job."""
        await self._after_embedding_backfill()

    async def _sync_keyword_index_from_chunks(self) -> None:
        """Repair keyword index when its persisted state does not match chunks."""
        if not self.keyword_index:
            return

        docs = {cid: chunk.text for cid, chunk in self.file_chunks.items() if chunk.text}
        expected_ids = set(docs)
        live_ids = None
        with suppress(Exception):
            live_ids = set(self.keyword_index.document_ids)

        # A non-empty chunk may still be unrepresentable by a lexical backend
        # (for example, punctuation-only text produces no BM25 tokens).  Check
        # only missing IDs so the normal matching path does not tokenize the
        # entire corpus during every startup.
        if live_ids is not None:
            unindexable_ids = {cid for cid in expected_ids - live_ids if not self.keyword_index.is_indexable(docs[cid])}
            expected_ids -= unindexable_ids

        if live_ids == expected_ids:
            return

        missing_count = len(expected_ids - live_ids) if live_ids is not None else len(expected_ids)
        extra_count = len(live_ids - expected_ids) if live_ids is not None else -1
        indexed_count = len(live_ids) if live_ids is not None else getattr(self.keyword_index, "n_docs", -1)
        self.logger.warning(
            f"{self.name}: keyword index mismatch: indexed={indexed_count}, expected={len(expected_ids)}, "
            f"missing={missing_count}, extra={extra_count}; rebuilding",
        )
        await self._rebuild_keyword_index(docs)

    async def _rebuild_keyword_index(self, docs: dict[str, str]) -> None:
        """Synchronously rebuild keyword search in batches with progress logs."""
        assert self.keyword_index is not None
        total = len(docs)
        started_at = time.monotonic()
        self.logger.info(
            f"{self.name}: keyword index rebuild started: total={total}, batch_size={_KEYWORD_REBUILD_BATCH_SIZE}",
        )
        try:
            # All built-in keyword indexes support clear/add/dump. Keep a fallback
            # for third-party implementations that only expose reset_index.
            if not all(hasattr(self.keyword_index, method) for method in ("clear", "add_docs", "dump")):
                await self.keyword_index.reset_index(docs)
                self._log_progress("keyword index rebuild", total, total, _PROGRESS_LOG_PERCENT_STEP)
            else:
                await self.keyword_index.clear()
                items = list(docs.items())
                next_percent = _PROGRESS_LOG_PERCENT_STEP
                for start in range(0, total, _KEYWORD_REBUILD_BATCH_SIZE):
                    batch = dict(items[start : start + _KEYWORD_REBUILD_BATCH_SIZE])
                    await self.keyword_index.add_docs(batch)
                    current = min(start + len(batch), total)
                    next_percent = self._log_progress("keyword index rebuild", current, total, next_percent)
                await self.keyword_index.dump()
        except Exception:
            elapsed = time.monotonic() - started_at
            self.logger.exception(f"{self.name}: keyword index rebuild failed after {elapsed:.2f}s")
            raise
        elapsed = time.monotonic() - started_at
        self.logger.info(f"{self.name}: keyword index rebuild complete: total={total}, elapsed={elapsed:.2f}s")

    async def _rebuild_tag_index(self, reason: str) -> bool:
        """Best-effort rebuild that never lets an optional tag index block the file store."""
        if self.tag_index is None:
            self._tag_index_rebuild_required = False
            return True
        nodes = await self.file_graph.get_nodes()
        try:
            await self.tag_index.rebuild(nodes)
        except Exception:
            self._tag_index_rebuild_required = True
            self.logger.exception(
                f"{self.name}: tag index rebuild failed during {reason}; keeping file store available",
            )
            return False
        self._tag_index_rebuild_required = False
        return True

    async def _upsert_tag_nodes(self, nodes: list[FileNode]) -> None:
        """Update tags without allowing optional-index failures to block other indexes."""
        if self.tag_index is None:
            return
        if self._tag_index_rebuild_required:
            await self._rebuild_tag_index("retry before incremental update")
            return
        try:
            await self.tag_index.upsert_nodes(nodes)
        except Exception:
            self._tag_index_rebuild_required = True
            self.logger.exception(f"{self.name}: incremental tag index update failed; rebuilding from graph")
            await self._rebuild_tag_index("incremental update recovery")

    async def _delete_tag_paths(self, paths: list[str]) -> None:
        """Delete tag paths without allowing optional-index failures to block other indexes."""
        if self.tag_index is None:
            return
        if self._tag_index_rebuild_required:
            await self._rebuild_tag_index("retry before incremental delete")
            return
        try:
            await self.tag_index.delete(paths)
        except Exception:
            self._tag_index_rebuild_required = True
            self.logger.exception(f"{self.name}: incremental tag index delete failed; rebuilding from graph")
            await self._rebuild_tag_index("incremental delete recovery")

    async def _dump_owned_state(self) -> None:
        """Persist state owned by this store, excluding dependency snapshots."""
        try:
            # Keep snapshotting synchronous so concurrent mutation cannot produce a
            # mixed-generation checkpoint. Move it off-loop only if profiling shows
            # this copy, rather than serialization/compression, is a material stall.
            chunks = tuple(chunk.model_copy(deep=True) for chunk in self.file_chunks.values())
            await complete_in_thread(self._dump_chunks_sync, chunks)
            self.logger.info(f"Saved {len(self.file_chunks)} chunks to {self.chunks_path}")
        except Exception as e:
            self.logger.exception(f"Failed to write {self.chunks_path}: {e}")
            raise

    def _dump_chunks_sync(self, chunks: tuple[FileChunk, ...]) -> None:
        write_jsonl_zst(
            self.chunks_path,
            (self._serialize_chunk(chunk) for chunk in chunks),
            self.encoding,
        )

    @BaseFileStore.serialized
    async def dump(self) -> None:
        """Persist a complete store/index/graph consistency checkpoint."""
        assert self.file_graph is not None
        await self._dump_owned_state()
        if self.keyword_index:
            await self.keyword_index.dump()
        await self.file_graph.dump()

    # -- maintenance -----------------------------------------------------------

    async def optimize_index(self) -> None:
        """Idle-time maintenance: compact the keyword index when present.

        The in-memory chunk map and file graph carry no deferred compaction
        debt; the keyword index may hold lazy-deleted docs, so delegate to its
        ``optimize_index`` for physical reclaim.
        """
        if self.keyword_index:
            await self.keyword_index.optimize_index()

    # -- CRUD -----------------------------------------------------------------

    @BaseFileStore.serialized
    async def upsert(self, files: list[tuple[FileNode, list[FileChunk]]]) -> None:
        if not files:
            return
        assert self.file_graph is not None

        old_map = {n.path: n for n in await self.file_graph.get_nodes([node.path for node, _ in files])}
        old_chunk_ids = {cid for n in old_map.values() for cid in n.chunk_ids}
        new_nodes, needs_embed, keyword_docs = self._stage_upsert(files, old_map)

        await self.file_graph.upsert_nodes(new_nodes)
        await self._upsert_tag_nodes(new_nodes)
        await self._embed_pending(needs_embed)
        if self.keyword_index and old_chunk_ids:
            await self.keyword_index.delete_docs(list(old_chunk_ids))
        if self.keyword_index and keyword_docs:
            await self.keyword_index.add_docs(keyword_docs)
        self._mutation_generation += 1

    def _stage_upsert(
        self,
        files: list[tuple[FileNode, list[FileChunk]]],
        old_map: dict[str, FileNode],
    ) -> tuple[list[FileNode], list[FileChunk], dict[str, str]]:
        """Mutate self.file_chunks for each file and collect the work the I/O step needs:
        new graph nodes, chunks still needing an embedding, and keyword docs to index.
        """
        new_nodes: list[FileNode] = []
        needs_embed: list[FileChunk] = []
        keyword_docs: dict[str, str] = {}
        for node, chunks in files:
            cached = self._evict_prior_chunks(old_map.get(node.path))
            node.chunk_ids = []
            for c in chunks:
                self._reuse_or_queue_embedding(c, cached, needs_embed)
                node.chunk_ids.append(c.id)
                self.file_chunks[c.id] = c
                if c.text:
                    keyword_docs[c.id] = c.text
            new_nodes.append(node)
        return new_nodes, needs_embed, keyword_docs

    def _evict_prior_chunks(self, old_node: FileNode | None) -> dict[str, CachedEmbedding]:
        """Drop chunks for the path being re-upserted; keep their embeddings around so
        a new chunk reusing the same id and text avoids a redundant embedding call.
        """
        cached: dict[str, CachedEmbedding] = {}
        if old_node is None:
            return cached
        for cid in old_node.chunk_ids:
            old = self.file_chunks.pop(cid, None)
            if self.embedding_store and old and old.embedding is not None:
                cached[cid] = (old.text, old.embedding)
        return cached

    def _reuse_or_queue_embedding(
        self,
        chunk: FileChunk,
        cached: dict[str, CachedEmbedding],
        needs_embed: list[FileChunk],
    ) -> None:
        if self._embedding_rebuild_pending:
            chunk.embedding = None
            return
        if not self.embedding_store:
            return
        if chunk.embedding is not None:
            if self._embedding_dim_matches(chunk.embedding):
                return
            self._drop_stale_embedding(chunk, "upsert")
        if (
            chunk.id in cached
            and cached[chunk.id][0] == chunk.text
            and self._embedding_dim_matches(cached[chunk.id][1])
        ):
            chunk.embedding = cached[chunk.id][1]
        elif chunk.text:
            needs_embed.append(chunk)

    async def _embed_pending(self, chunks: list[FileChunk]) -> None:
        if not (chunks and self.embedding_store) or self._embedding_rebuild_pending:
            return
        embedding_store = self.embedding_store
        embedding_generation = self._embedding_space_generation
        was_healthy = bool(getattr(embedding_store, "is_healthy", True))
        try:
            await embedding_store.get_node_embeddings(chunks)
        except Exception as e:
            if self._embedding_request_is_current(embedding_store, embedding_generation):
                self._mark_embedding_unhealthy(f"upsert: {type(e).__name__}: {e}")
            return
        if not self._embedding_request_is_current(embedding_store, embedding_generation):
            for chunk in chunks:
                chunk.embedding = None
            if not self._embedding_rebuild_pending:
                self._start_embedding_backfill(skip_health_check=True)
            return
        self._drop_stale_embeddings(chunks, "upsert")
        if any(chunk.embedding is not None for chunk in chunks):
            await self._recover_after_real_request(was_healthy)

    @BaseFileStore.serialized
    async def delete(self, path: str | list[str]) -> None:
        assert self.file_graph is not None
        paths = [path] if isinstance(path, str) else path
        nodes: list[FileNode] = await self.file_graph.get_nodes(paths)
        await self._delete_nodes(nodes)
        if nodes:
            self._mutation_generation += 1

    async def _delete_nodes(self, nodes: list[FileNode]) -> None:
        """Delete already-resolved nodes and their chunks.

        Split out so subclasses that need the node list before deletion (e.g. to
        capture chunk ids for a vector index) can reuse it instead of querying the
        graph a second time.
        """
        assert self.file_graph is not None
        if not nodes:
            return
        deleted_chunk_ids = [cid for n in nodes for cid in n.chunk_ids]
        for cid in deleted_chunk_ids:
            self.file_chunks.pop(cid, None)
        await self.file_graph.delete_nodes([str(n.path) for n in nodes])
        await self._delete_tag_paths([str(n.path) for n in nodes])
        if self.keyword_index and deleted_chunk_ids:
            await self.keyword_index.delete_docs(deleted_chunk_ids)

    async def get_nodes(self, paths: list[str] | None = None) -> list[FileNode]:
        assert self.file_graph is not None
        return await self.file_graph.get_nodes(paths)

    async def get_outlinks(
        self,
        path: str,
        scope: LinkScopeEnum = LinkScopeEnum.REAL,
    ) -> list[FileLink]:
        assert self.file_graph is not None
        return await self.file_graph.get_outlinks(path, scope)

    async def get_inlinks(
        self,
        path: str,
        scope: LinkScopeEnum = LinkScopeEnum.REAL,
    ) -> list[FileLink]:
        assert self.file_graph is not None
        return await self.file_graph.get_inlinks(path, scope)

    @BaseFileStore.serialized
    async def clear(self) -> None:
        assert self.file_graph is not None
        self.file_chunks.clear()
        self.chunks_path.unlink(missing_ok=True)
        if self.keyword_index:
            await self.keyword_index.clear()
        await self.file_graph.clear()
        if self.tag_index is not None:
            try:
                await self.tag_index.clear()
                self._tag_index_rebuild_required = False
            except Exception:
                self._tag_index_rebuild_required = True
                self.logger.exception(f"{self.name}: tag index clear failed; rebuilding from empty graph")
                await self._rebuild_tag_index("clear recovery")
        self._mutation_generation += 1

    # -- search ---------------------------------------------------------------

    async def vector_search(self, query: str, limit: int, search_filter: dict) -> list[FileChunk]:
        if limit <= 0:
            return []

        query_embedding = await self._get_query_embedding(query)
        if query_embedding is None:
            return []

        top: list[tuple[float, int, FileChunk]] = []
        candidates: list[FileChunk] = []
        embeddings: list[np.ndarray] = []
        order = 0

        def score_batch() -> None:
            nonlocal order
            if not candidates:
                return
            matrix = np.stack(embeddings)
            similarities = batch_cosine_similarity(query_embedding.reshape(1, -1), matrix)[0]
            for candidate, similarity in zip(candidates, similarities):
                score = float(similarity)
                item = (score, -order, candidate)
                if len(top) < limit:
                    heapq.heappush(top, item)
                elif item[:2] > top[0][:2]:
                    heapq.heapreplace(top, item)
                order += 1
            candidates.clear()
            embeddings.clear()

        for candidate in self.file_chunks.values():
            if not self._embedding_dim_matches(candidate.embedding) or not self._matches_search_filter(
                candidate,
                search_filter,
            ):
                continue
            candidates.append(candidate)
            embeddings.append(candidate.embedding)
            if len(candidates) >= _VECTOR_SEARCH_BATCH_SIZE:
                score_batch()
        score_batch()

        ranked = sorted(top, key=lambda item: (-item[0], -item[1]))
        return [
            candidate.model_copy(update={"scores": {"vector": score, "score": score}})
            for score, _neg_order, candidate in ranked
        ]

    async def keyword_search(self, query: str, limit: int, search_filter: dict) -> list[FileChunk]:
        if not self.keyword_index:
            return []

        query = query.strip()
        if not query:
            return []

        retrieve_limit = limit
        if search_filter:
            retrieve_limit = max(limit, getattr(self.keyword_index, "n_docs", limit))
        doc_id_score_dict = await self.keyword_index.retrieve(query, limit=retrieve_limit)
        results = []
        for doc_id, score in doc_id_score_dict.items():
            chunk = self.file_chunks.get(doc_id)
            if chunk and self._matches_search_filter(chunk, search_filter):
                results.append(chunk.model_copy(update={"scores": {"keyword": score, "score": score}}))
                if len(results) >= limit:
                    break

        return results

    # -- extensions -----------------------------------------------------------

    @staticmethod
    def _as_filter_values(value) -> set:
        if isinstance(value, (list, tuple, set, frozenset)):
            return set(value)
        return {value}

    @classmethod
    def _value_matches(cls, actual, expected) -> bool:
        if isinstance(expected, (list, tuple, set, frozenset)):
            return actual in set(expected)
        return actual == expected

    @staticmethod
    def _extract_date_from_path(path: str) -> str | None:
        """Extract the date from a file path following the project path convention.

        Paths with dates always place them at the 2nd segment:
            daily/2026-05-18/note.md
            resource/2026-06-06/report.pdf
            daily/2026-05-18.md  (day index, date is stem of segment)

        Returns a validated YYYY-MM-DD string, or None if no date is found.
        """
        parts = path.split("/")
        if len(parts) < 2:
            return None
        # Accept only exact "YYYY-MM-DD" (dir) or "YYYY-MM-DD.md" (day index).
        segment = parts[1]
        candidate = segment if "." not in segment else segment.rsplit(".", 1)[0]
        if segment != candidate and not segment.endswith(".md"):
            return None
        try:
            return datetime.date.fromisoformat(candidate).isoformat()
        except ValueError:
            return None

    @classmethod
    def _matches_search_filter(cls, chunk: FileChunk, search_filter: dict | None) -> bool:
        """Conservative post-filter shared by vector and keyword search."""
        if not search_filter:
            return True

        exact_paths = set()
        for key in ("path", "paths"):
            if key in search_filter:
                exact_paths.update(cls._as_filter_values(search_filter[key]))
        if exact_paths and chunk.path not in exact_paths:
            return False

        prefixes = []
        for key in ("path_prefix", "path_prefixes", "prefix", "prefixes"):
            if key in search_filter:
                prefixes.extend(str(v) for v in cls._as_filter_values(search_filter[key]))
        if prefixes and not any(chunk.path.startswith(prefix) for prefix in prefixes):
            return False

        # Date range filtering based on date embedded in chunk path.
        # strict_date_filter (default False): when True and at least one date
        # bound is set, chunks whose path yields no date are excluded.
        start_date = search_filter.get("start_date")
        end_date = search_filter.get("end_date")
        strict_date = bool(search_filter.get("strict_date_filter", False))
        if start_date or end_date:
            path_date = cls._extract_date_from_path(chunk.path)
            if not path_date:
                if strict_date:
                    return False
            elif (start_date and path_date < start_date) or (end_date and path_date > end_date):
                return False

        metadata_filter = dict(search_filter.get("metadata") or {})
        reserved = {
            "path",
            "paths",
            "path_prefix",
            "path_prefixes",
            "prefix",
            "prefixes",
            "metadata",
            "start_date",
            "end_date",
            "strict_date_filter",
        }
        for key, value in search_filter.items():
            if key not in reserved:
                metadata_filter[key] = value

        return all(cls._value_matches(chunk.metadata.get(key), value) for key, value in metadata_filter.items())

    async def rebuild_links(self) -> None:
        """Rebuild graph links via the underlying file graph."""
        assert self.file_graph is not None
        return await self.file_graph.rebuild_links()
