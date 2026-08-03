"""Zvec-backed file store: chunk JSONL stays authoritative; a zvec collection replaces the linear vector scan."""

import hashlib
import json
import shutil
from uuid import uuid4

import aiofiles
import numpy as np

from .local_file_store import LocalFileStore
from ..component_registry import R
from ...schema import FileChunk, FileNode

# Batch size for bulk inserts during a rebuild.
_ZVEC_INSERT_BATCH_SIZE = 1024


@R.register("zvec")
class ZvecLocalFileStore(LocalFileStore):
    """LocalFileStore variant whose vector_search is backed by a zvec collection.

    Chunk persistence is unchanged (JSONL, owned by the parent); the zvec
    collection only stores ``(chunk_id, embedding)`` pairs and serves ANN
    queries. ``self.file_chunks`` remains the source of truth: if the
    collection directory or its digest sidecar is missing or stale, the
    collection is rebuilt from the chunks.

    zvec (https://zvec.org) is an in-process vector database, so unlike the
    FAISS backend there is no tombstone bookkeeping: documents are updated and
    removed natively via ``Collection.upsert`` / ``Collection.delete``, and
    the collection persists itself inside its own directory. A small JSON
    sidecar records an order-independent digest of the live
    ``(chunk_id, embedding)`` set at dump time; on load a digest mismatch
    (crash between the chunk dump and the collection flush, externally mixed
    files, model change) triggers a clean rebuild.

    HNSW parameters (``hnsw_m``, ``hnsw_ef_construction``) are baked into the
    collection schema at creation time; changing ``hnsw_m`` on an existing
    store triggers a rebuild so the graph topology matches the configuration.

    ``optimize_index`` maps directly onto ``Collection.optimize()``, zvec's
    idle-time index compaction hook.

    zvec is imported lazily inside ``__init__`` so that merely importing this
    module does not require the optional dependency.
    """

    def __init__(
        self,
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._zvec = self._import_zvec()
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.zvec_path = self.component_metadata_path / f"zvec_index_{self.name}_{self.store_version}"
        self.zvec_sidecar_path = self.component_metadata_path / f"zvec_sidecar_{self.name}_{self.store_version}.json"
        self._collection = None  # zvec.Collection | None
        self._indexed_ids: set[str] = set()  # chunk ids currently present in the collection

    @staticmethod
    def _import_zvec():
        try:
            import zvec
        except ImportError as e:
            raise ImportError(
                "zvec is required for ZvecLocalFileStore. Install with `pip install zvec`.",
            ) from e
        return zvec

    # -- helpers ----------------------------------------------------------

    @property
    def _dim(self) -> int:
        return self.embedding_store.dimensions if self.embedding_store is not None else 0

    def _collection_schema(self):
        """Vector-only schema: chunk data lives in the JSONL, zvec is pure ANN."""
        zvec = self._zvec
        return zvec.CollectionSchema(
            name=self.name,
            fields=[],
            vectors=[
                zvec.VectorSchema(
                    name="embedding",
                    data_type=zvec.DataType.VECTOR_FP32,
                    dimension=self._dim,
                    index_param=zvec.HnswIndexParam(
                        metric_type=zvec.MetricType.COSINE,
                        m=self.hnsw_m,
                        ef_construction=self.hnsw_ef_construction,
                    ),
                ),
            ],
        )

    def _create_collection(self):
        """Create a fresh (empty) collection, replacing any directory on disk."""
        # Release any open handle first: zvec holds an in-process lock on the
        # collection directory, so the old object must be dropped before the
        # directory is wiped and re-created.
        self._collection = None
        if self.zvec_path.exists():
            shutil.rmtree(self.zvec_path, ignore_errors=True)
        self._indexed_ids = set()
        return self._zvec.create_and_open(path=str(self.zvec_path), schema=self._collection_schema())

    def _to_doc(self, chunk: FileChunk):
        """Build a zvec Doc carrying only the id and the float32 vector."""
        vector = np.asarray(chunk.embedding, dtype=np.float32).tolist()
        return self._zvec.Doc(id=chunk.id, vectors={"embedding": vector})

    def _upsert_docs(self, chunks: list[FileChunk]) -> None:
        if not chunks or self._collection is None:
            return
        self._collection.upsert([self._to_doc(c) for c in chunks])
        self._indexed_ids.update(c.id for c in chunks)

    def _delete_docs(self, chunk_ids: list[str]) -> None:
        if self._collection is None:
            return
        stale = [cid for cid in chunk_ids if cid in self._indexed_ids]
        if not stale:
            return
        self._collection.delete(ids=stale)
        self._indexed_ids.difference_update(stale)

    def _rebuild_collection(self) -> None:
        """Rebuild the zvec collection from self.file_chunks (the source of truth)."""
        self._collection = self._create_collection()
        chunks = [c for c in self.file_chunks.values() if self._embedding_dim_matches(c.embedding)]
        for start in range(0, len(chunks), _ZVEC_INSERT_BATCH_SIZE):
            batch = chunks[start : start + _ZVEC_INSERT_BATCH_SIZE]
            self._collection.insert([self._to_doc(c) for c in batch])
            self._indexed_ids.update(c.id for c in batch)
        self.logger.info(f"{self.name}: rebuilt zvec collection with {len(chunks)} vectors at {self.zvec_path}")

    async def _after_embedding_backfill(self) -> None:
        """Make newly backfilled vectors visible to zvec.

        zvec upserts are incremental by nature, so backfilled vectors are added
        directly to the live collection; no rebuild or tombstone accounting is
        needed.
        """
        if self.embedding_store is None or self._dim == 0:
            return
        if self._collection is None:
            self._rebuild_collection()
            return
        to_add = [
            chunk
            for cid, chunk in self.file_chunks.items()
            if cid not in self._indexed_ids and self._embedding_dim_matches(chunk.embedding)
        ]
        self._upsert_docs(to_add)

    # -- maintenance ------------------------------------------------------

    async def optimize_index(self) -> None:
        """Idle-time maintenance: delegate to zvec's native index optimization.

        ``Collection.optimize()`` merges staged segments into the persistent
        HNSW index, the direct analogue of a tombstone compaction pass.
        """
        await super().optimize_index()
        if self._collection is None:
            return
        try:
            self._collection.optimize()
            self.logger.info(f"{self.name}: zvec collection optimized")
        except Exception as e:
            self.logger.exception(f"{self.name}: zvec optimize failed: {e}")

    # -- lifecycle ----------------------------------------------------------

    async def _close(self) -> None:
        """Persist via the parent, then release the collection handle.

        zvec holds an in-process lock on the collection directory for the
        lifetime of the ``Collection`` object; dropping the reference releases
        it so another store instance can reopen the same directory.
        """
        await super()._close()  # parent dump() flushes the live collection
        self._collection = None
        self._indexed_ids = set()

    # -- persistence ------------------------------------------------------

    def _chunks_embedding_digest(self) -> str:
        """Order-independent digest of the live (chunk_id, embedding) set.

        Hashes float16 canonical bytes (the chunk JSONL serialization dtype) so
        the value is identical whether an embedding sits fresh in memory or has
        round-tripped through the JSONL. Written into the sidecar at dump time
        and recomputed from ``self.file_chunks`` at load time: a mismatch means
        the collection belongs to a different chunk generation than the
        authoritative JSONL.
        """
        digest = hashlib.sha256()
        eligible = sorted(
            cid for cid, chunk in self.file_chunks.items() if self._embedding_dim_matches(chunk.embedding)
        )
        for cid in eligible:
            digest.update(cid.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(np.asarray(self.file_chunks[cid].embedding, dtype=np.float16).tobytes())
        return digest.hexdigest()

    async def load(self) -> None:
        """Load chunks via the parent, then attach the zvec collection (open or rebuild)."""
        await super().load()
        if self.embedding_store is None or self._dim == 0:
            self._collection = None
            return
        if not await self._try_open_collection():
            self._rebuild_collection()

    async def _try_open_collection(self) -> bool:
        """Open the persisted collection and validate it against the chunks.

        On any mismatch or open error the collection directory and sidecar are
        wiped so the caller can rebuild from chunks cleanly. Validated:

        - vector dimension against the active embedding model;
        - HNSW ``M`` against the active config (baked into the graph topology);
        - the sidecar embedding digest against the authoritative JSONL.
        """
        if not (self.zvec_path.exists() and self.zvec_sidecar_path.exists()):
            return False
        collection = None
        try:
            async with aiofiles.open(self.zvec_sidecar_path, encoding=self.encoding) as f:
                sidecar = json.loads(await f.read())
            if sidecar.get("digest") != self._chunks_embedding_digest():
                raise ValueError("zvec sidecar embedding digest does not match persisted chunks")
            indexed_ids = set(sidecar.get("ids", []))
            expected_ids = {
                cid for cid, chunk in self.file_chunks.items() if self._embedding_dim_matches(chunk.embedding)
            }
            if indexed_ids != expected_ids:
                raise ValueError("zvec sidecar ids do not match persisted chunks")

            collection = self._zvec.open(path=str(self.zvec_path))
            vector_schema = collection.schema.vectors[0]
            if vector_schema.dimension != self._dim:
                raise ValueError(f"zvec dim {vector_schema.dimension} != embedding dim {self._dim}")
            persisted_m = vector_schema.index_param.m
            if persisted_m != self.hnsw_m:
                raise ValueError(f"zvec HNSW M mismatch: persisted={persisted_m}, configured={self.hnsw_m}")

            self._collection = collection
            self._indexed_ids = indexed_ids
            self.logger.info(f"Opened zvec collection: {len(indexed_ids)} vectors from {self.zvec_path}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to open zvec collection, will rebuild: {e}")
            collection = None  # release the handle before wiping the directory
            if self.zvec_path.exists():
                shutil.rmtree(self.zvec_path, ignore_errors=True)
            self.zvec_sidecar_path.unlink(missing_ok=True)
            return False

    async def dump(self) -> None:
        """Persist chunks JSONL via the parent, then flush zvec and write the sidecar."""
        await super().dump()
        if self._collection is None or self.embedding_store is None:
            return
        try:
            self._collection.flush()
            await self._write_sidecar()
            self.logger.info(f"Saved zvec collection: {len(self._indexed_ids)} vectors to {self.zvec_path}")
        except Exception as e:
            self.logger.exception(f"Failed to persist zvec collection: {e}")

    async def _write_sidecar(self) -> None:
        """Atomically write the digest sidecar binding the collection to the chunk generation."""
        tmp = self.zvec_sidecar_path.with_name(f".{self.zvec_sidecar_path.name}.{uuid4().hex}.tmp")
        payload = json.dumps(
            {
                "ids": sorted(self._indexed_ids),
                "digest": self._chunks_embedding_digest(),
            },
        )
        try:
            async with aiofiles.open(tmp, "w", encoding=self.encoding) as f:
                await f.write(payload)
            tmp.replace(self.zvec_sidecar_path)
        finally:
            tmp.unlink(missing_ok=True)

    # -- CRUD overrides ---------------------------------------------------

    async def upsert(self, files: list[tuple[FileNode, list[FileChunk]]]) -> None:
        if not files:
            return
        assert self.file_graph is not None

        # Snapshot pre-upsert chunk state so we can diff after the parent ran.
        old_nodes = await self.file_graph.get_nodes([node.path for node, _ in files])
        old_ids_by_path = {n.path: set(n.chunk_ids) for n in old_nodes}
        old_text_by_id = {
            cid: chunk.text
            for n in old_nodes
            for cid in n.chunk_ids
            if (chunk := self.file_chunks.get(cid)) is not None
        }
        await super().upsert(files)

        if self._collection is None or self.embedding_store is None:
            return
        self._sync_collection_after_upsert(files, old_ids_by_path, old_text_by_id)

    def _sync_collection_after_upsert(
        self,
        files: list[tuple[FileNode, list[FileChunk]]],
        old_ids_by_path: dict[str, set[str]],
        old_text_by_id: dict[str, str],
    ) -> None:
        """Apply add / delete deltas to the zvec collection natively."""
        to_delete: list[str] = []
        to_upsert: list[FileChunk] = []
        for node, _ in files:
            new_ids = set(node.chunk_ids)
            to_delete.extend(old_ids_by_path.get(node.path, set()) - new_ids)
            for cid in new_ids:
                chunk = self.file_chunks.get(cid)
                if chunk is None or not self._embedding_dim_matches(chunk.embedding):
                    continue
                if cid in self._indexed_ids and old_text_by_id.get(cid) == chunk.text:
                    continue  # unchanged text -> vector already in the collection
                to_upsert.append(chunk)
        self._delete_docs(to_delete)
        self._upsert_docs(to_upsert)

    async def delete(self, path: str | list[str]) -> None:
        assert self.file_graph is not None
        paths = [path] if isinstance(path, str) else path
        nodes = await self.file_graph.get_nodes(paths)
        deleted_ids = [cid for n in nodes for cid in n.chunk_ids]
        await self._delete_nodes(nodes)  # reuse resolved nodes; avoids a second get_nodes
        self._delete_docs(deleted_ids)

    async def clear(self) -> None:
        await super().clear()
        if self._collection is not None:
            try:
                self._collection.destroy()  # drops the collection directory
            except Exception:  # pragma: no cover - defensive
                shutil.rmtree(self.zvec_path, ignore_errors=True)
        elif self.zvec_path.exists():
            shutil.rmtree(self.zvec_path, ignore_errors=True)
        self._indexed_ids = set()
        self.zvec_sidecar_path.unlink(missing_ok=True)
        self._collection = self._create_collection() if self.embedding_store is not None else None

    # -- search -----------------------------------------------------------

    async def vector_search(self, query: str, limit: int, search_filter: dict) -> list[FileChunk]:
        if self.embedding_store is None or not query or limit <= 0 or self._collection is None or not self._indexed_ids:
            return []

        query_embedding = None
        try:
            query_embedding = await self.embedding_store.get_embedding(query)
        except Exception as e:
            self._disable_embedding(f"search: {type(e).__name__}: {e}")
        if query_embedding is None or not self._embedding_dim_matches(query_embedding):
            if query_embedding is not None:
                self._disable_embedding(
                    f"search: query embedding dimension {len(query_embedding)} != {self.embedding_store.dimensions}",
                )
            return []

        # get_embedding above yielded control; a concurrent clear() may have
        # swapped or dropped the collection. Re-read before dereferencing.
        collection = self._collection
        if collection is None or not self._indexed_ids:
            return []

        vector = np.asarray(query_embedding, dtype=np.float32).tolist()
        ntotal = len(self._indexed_ids)

        if not search_filter:
            hits = self._query_collection(collection, vector, min(limit, ntotal))
            return self._collect_hits(hits, limit, search_filter)

        # With a post-filter: progressively increase k until we collect enough
        # results or exhaust the reachable index.
        k = min(ntotal, 3 * limit)
        while True:
            hits = self._query_collection(collection, vector, k)
            results = self._collect_hits(hits, limit, search_filter)
            if len(results) >= limit or k >= ntotal:
                return results
            k = min(ntotal, k * 2)

    def _query_collection(self, collection, vector: list[float], topk: int) -> list[tuple[str, float]]:
        """Run an ANN query and convert cosine distance to similarity (higher = closer)."""
        docs = collection.query(
            self._zvec.VectorQuery(field_name="embedding", vector=vector),
            topk=max(1, topk),
        )
        return [(doc.id, 1.0 - float(doc.score)) for doc in docs]

    def _collect_hits(
        self,
        hits: list[tuple[str, float]],
        limit: int,
        search_filter: dict | None = None,
    ) -> list[FileChunk]:
        """Map zvec hits back to chunks, skipping stale ids and filtered-out chunks."""
        results: list[FileChunk] = []
        for chunk_id, score in hits:
            chunk = self.file_chunks.get(chunk_id)
            if chunk is None or not self._matches_search_filter(chunk, search_filter):
                continue
            results.append(chunk.model_copy(update={"scores": {"vector": score, "score": score}}))
            if len(results) >= limit:
                break
        return results
