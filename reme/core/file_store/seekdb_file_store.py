"""seekdb storage backend for file store."""

import time
from pathlib import Path

from loguru import logger

from .base_file_store import BaseFileStore
from ..enumeration import MemorySource
from ..schema import FileMetadata, MemoryChunk, MemorySearchResult

try:
    import pyseekdb
    from pyseekdb import Configuration, HNSWConfiguration, FulltextIndexConfig

    PYSEEKDB_AVAILABLE = True
except ImportError:
    PYSEEKDB_AVAILABLE = False
    pyseekdb = None
    Configuration = None
    HNSWConfiguration = None
    FulltextIndexConfig = None


def _escape_sql(s: str) -> str:
    """Escape single quotes for SQL string literals (MySQL/seekdb)."""
    return s.replace("\\", "\\\\").replace("'", "''")


class SeekdbFileStore(BaseFileStore):
    """seekdb file storage with vector and full-text search (embedded mode only).

    File metadata is stored in a DB table (same as SqliteFileStore); uses pyseekdb
    BaseClient execute/_execute for SQL. No env vars, uses db_path and store_name only.
    """

    def __init__(
        self,
        **kwargs,
    ):
        if not PYSEEKDB_AVAILABLE:
            raise ImportError(
                "pyseekdb is required for SeekdbFileStore. "
                "Install it with: pip install reme-ai (pyseekdb is included)",
            )

        super().__init__(**kwargs)
        self.client: "pyseekdb.Client | None" = None
        self.collection = None

    @property
    def collection_name(self) -> str:
        """Collection name for chunks."""
        return f"chunks_{self.store_name}"

    @property
    def files_table_name(self) -> str:
        """Table name for file metadata (same as SQLite)."""
        return f"files_{self.store_name}"

    def _client_kwargs(self) -> dict:
        """Build Client kwargs for embedded mode from db_path and store_name (same as Chroma/SQLite)."""
        return {"path": str(self.db_path / "seekdb"), "database": self.store_name}

    def _sql_client(self):
        """Underlying BaseClient for raw SQL (pyseekdb Client proxy exposes _server)."""
        if self.client is None:
            return None
        return getattr(self.client, "_server", self.client)

    def _execute_sql(self, sql: str):
        """Execute SQL via pyseekdb embedded/server client. Returns fetchall() for SELECT/SHOW/DESCRIBE."""
        client = self._sql_client()
        if client is None:
            raise RuntimeError("seekdb client not initialized")
        execute_fn = getattr(client, "execute", None) or getattr(client, "_execute", None)
        if execute_fn is None:
            raise RuntimeError("seekdb client has no execute/_execute for raw SQL")
        return execute_fn(sql)

    def _create_files_table(self) -> None:
        """Create file metadata table (same schema as SQLite)."""
        sql = f"""
        CREATE TABLE IF NOT EXISTS `{self.files_table_name}` (
            path VARCHAR(1024),
            source VARCHAR(128),
            hash VARCHAR(256),
            mtime REAL,
            size BIGINT,
            PRIMARY KEY (path, source)
        )
        """
        self._execute_sql(sql)
        logger.debug(f"seekdb files table: {self.files_table_name}")

    async def start(self) -> None:
        """Initialize seekdb client, collection, and files table."""
        if self.client is not None:
            return

        kwargs = self._client_kwargs()
        if "path" in kwargs:
            path = Path(kwargs["path"])
            path.mkdir(parents=True, exist_ok=True)
            database = kwargs.get("database", self.store_name)
            try:
                admin = pyseekdb.AdminClient(path=str(path))
                if not any(db.name == database for db in admin.list_databases()):
                    admin.create_database(database)
            except Exception as e:
                logger.debug("seekdb AdminClient create_database: %s", e)
        self.client = pyseekdb.Client(**kwargs)

        dim = self.embedding_dim
        config = Configuration(
            hnsw=HNSWConfiguration(dimension=dim, distance="cosine"),
            fulltext_config=FulltextIndexConfig(analyzer="space"),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            configuration=config,
            embedding_function=None,
        )
        self._create_files_table()

        logger.info(
            f"seekdb initialized with collection: {self.collection_name}, "
            f"files table: {self.files_table_name}",
        )

    async def upsert_file(
        self,
        file_meta: FileMetadata,
        source: MemorySource,
        chunks: list[MemoryChunk],
    ) -> None:
        """Insert or update file and its chunks."""
        if not chunks:
            return

        await self.delete_file(file_meta.path, source)
        chunks = await self.get_chunk_embeddings(chunks)

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        now = int(time.time() * 1000)
        for chunk in chunks:
            ids.append(chunk.id)
            documents.append(chunk.text)
            embeddings.append(chunk.embedding)
            metadatas.append(
                {
                    "path": file_meta.path,
                    "source": source.value,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "hash": chunk.hash,
                    "updated_at": now,
                },
            )

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        # File metadata in DB table (same as SQLite)
        p, s = _escape_sql(file_meta.path), _escape_sql(source.value)
        h = _escape_sql(file_meta.hash)
        mtime = file_meta.mtime_ms
        size = file_meta.size
        sql = (
            f"REPLACE INTO `{self.files_table_name}` (path, source, hash, mtime, size) "
            f"VALUES ('{p}', '{s}', '{h}', {mtime}, {size})"
        )
        self._execute_sql(sql)

    async def delete_file(self, path: str, source: MemorySource) -> None:
        """Delete file and all its chunks."""
        results = self.collection.get(
            where={"$and": [{"path": path}, {"source": source.value}]},
            include=[],
        )
        if results.get("ids"):
            self.collection.delete(ids=results["ids"])
        p, s = _escape_sql(path), _escape_sql(source.value)
        self._execute_sql(f"DELETE FROM `{self.files_table_name}` WHERE path = '{p}' AND source = '{s}'")

    async def delete_file_chunks(self, path: str, chunk_ids: list[str]) -> None:
        """Delete specific chunks for a file (chunk count comes from collection at get_file_metadata)."""
        if not chunk_ids:
            return
        self.collection.delete(ids=chunk_ids)

    async def upsert_chunks(
        self,
        chunks: list[MemoryChunk],
        source: MemorySource,
    ) -> None:
        """Insert or update specific chunks."""
        if not chunks:
            return
        chunks = await self.get_chunk_embeddings(chunks)
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        now = int(time.time() * 1000)
        for chunk in chunks:
            ids.append(chunk.id)
            documents.append(chunk.text)
            embeddings.append(chunk.embedding)
            metadatas.append(
                {
                    "path": chunk.path,
                    "source": source.value,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "hash": chunk.hash,
                    "updated_at": now,
                },
            )
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    async def list_files(self, source: MemorySource) -> list[str]:
        """List all indexed files for a source (from files table)."""
        s = _escape_sql(source.value)
        rows = self._execute_sql(f"SELECT path FROM `{self.files_table_name}` WHERE source = '{s}'")
        if not rows:
            return []
        return [row[0] if isinstance(row, (list, tuple)) else row.get("path") for row in rows]

    async def get_file_metadata(
        self,
        path: str,
        source: MemorySource,
    ) -> FileMetadata | None:
        """Get file metadata from files table and chunk count from collection."""
        p, s = _escape_sql(path), _escape_sql(source.value)
        rows = self._execute_sql(
            f"SELECT hash, mtime, size FROM `{self.files_table_name}` WHERE path = '{p}' AND source = '{s}'"
        )
        if not rows:
            return None
        row = rows[0]
        if isinstance(row, (list, tuple)):
            hash_val, mtime, size = row[0], row[1], row[2]
        else:
            hash_val, mtime, size = row["hash"], row["mtime"], row["size"]
        results = self.collection.get(
            where={"$and": [{"path": path}, {"source": source.value}]},
            include=[],
        )
        chunk_count = len(results.get("ids") or [])
        return FileMetadata(
            path=path,
            hash=hash_val or "",
            mtime_ms=mtime or 0,
            size=size or 0,
            chunk_count=chunk_count,
        )

    async def update_file_metadata(self, file_meta: FileMetadata, source: MemorySource) -> None:
        """Update file metadata in files table without affecting chunks."""
        p = _escape_sql(file_meta.path)
        s = _escape_sql(source.value)
        h = _escape_sql(file_meta.hash)
        mtime = file_meta.mtime_ms
        size = file_meta.size
        sql = (
            f"REPLACE INTO `{self.files_table_name}` (path, source, hash, mtime, size) "
            f"VALUES ('{p}', '{s}', '{h}', {mtime}, {size})"
        )
        self._execute_sql(sql)

    async def get_file_chunks(
        self,
        path: str,
        source: MemorySource,
    ) -> list[MemoryChunk]:
        """Get all chunks for a file."""
        results = self.collection.get(
            where={"$and": [{"path": path}, {"source": source.value}]},
            include=["documents", "embeddings", "metadatas"],
        )
        chunks = []
        ids = results.get("ids") or []
        documents = results.get("documents") or []
        embeddings = results.get("embeddings") or []
        metadatas = results.get("metadatas") or []
        for i, chunk_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            doc = documents[i] if i < len(documents) else ""
            emb = embeddings[i] if embeddings and i < len(embeddings) else None
            chunks.append(
                MemoryChunk(
                    id=chunk_id,
                    path=meta.get("path", path),
                    source=MemorySource(meta.get("source", source.value)),
                    start_line=meta.get("start_line", 0),
                    end_line=meta.get("end_line", 0),
                    text=doc,
                    hash=meta.get("hash", ""),
                    embedding=emb,
                ),
            )
        chunks.sort(key=lambda c: c.start_line)
        return chunks

    async def vector_search(
        self,
        query: str,
        limit: int,
        sources: list[MemorySource] | None = None,
    ) -> list[MemorySearchResult]:
        """Perform vector similarity search."""
        if not self.vector_enabled or not query:
            return []
        query_embedding = await self.get_embedding(query)
        if not query_embedding:
            return []

        where_filter = None
        if sources:
            if len(sources) == 1:
                where_filter = {"source": sources[0].value}
            else:
                where_filter = {"source": {"$in": [s.value for s in sources]}}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results.get("ids") and results["ids"][0]:
            for i, _ in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                score = max(0.0, 1.0 - distance / 2.0)
                search_results.append(
                    MemorySearchResult(
                        path=metadata["path"],
                        start_line=metadata["start_line"],
                        end_line=metadata["end_line"],
                        score=score,
                        snippet=results["documents"][0][i],
                        source=MemorySource(metadata["source"]),
                        raw_metric=distance,
                    ),
                )
        search_results.sort(key=lambda r: r.score, reverse=True)
        return search_results

    async def keyword_search(
        self,
        query: str,
        limit: int,
        sources: list[MemorySource] | None = None,
    ) -> list[MemorySearchResult]:
        """Perform full-text keyword search."""
        if not self.fts_enabled or not query or not query.strip():
            return []

        where_filter = None
        if sources:
            if len(sources) == 1:
                where_filter = {"source": sources[0].value}
            else:
                where_filter = {"source": {"$in": [s.value for s in sources]}}

        results = self.collection.get(
            where=where_filter,
            where_document={"$contains": query.strip()},
            limit=limit,
            include=["documents", "metadatas"],
        )

        search_results = []
        ids = results.get("ids") or []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        query_lower = query.lower()
        words = query_lower.split()
        n_words = max(1, len(words))

        for i, _ in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            text = documents[i] if i < len(documents) else ""
            match_count = sum(1 for w in words if w in text.lower())
            score = min(1.0, match_count / n_words + (0.2 if query_lower in text.lower() else 0.0))
            search_results.append(
                MemorySearchResult(
                    path=meta.get("path", ""),
                    start_line=meta.get("start_line", 0),
                    end_line=meta.get("end_line", 0),
                    score=score,
                    snippet=text,
                    source=MemorySource(meta.get("source", "memory")),
                ),
            )
        search_results.sort(key=lambda r: r.score, reverse=True)
        return search_results[:limit]

    async def hybrid_search(
        self,
        query: str,
        limit: int,
        sources: list[MemorySource] | None = None,
        vector_weight: float = 0.7,
        candidate_multiplier: float = 3.0,
    ) -> list[MemorySearchResult]:
        """Perform hybrid search using seekdb native hybrid_search when possible."""
        if not query or not query.strip():
            return []

        assert 0.0 <= vector_weight <= 1.0
        candidates = min(200, max(1, int(limit * candidate_multiplier)))
        where_filter = None
        if sources:
            if len(sources) == 1:
                where_filter = {"source": sources[0].value}
            else:
                where_filter = {"source": {"$in": [s.value for s in sources]}}

        if self.vector_enabled and self.fts_enabled:
            try:
                query_embedding = await self.get_embedding(query)
                if not query_embedding:
                    return await self.keyword_search(query, limit, sources)

                results = self.collection.hybrid_search(
                    query={
                        "where_document": {"$contains": query.strip()},
                        "where": where_filter,
                        "n_results": candidates,
                    },
                    knn={
                        "query_embeddings": [query_embedding],
                        "where": where_filter,
                        "n_results": candidates,
                    },
                    rank={"rrf": {}},
                    n_results=limit,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception as e:
                logger.warning(f"seekdb hybrid_search failed, fallback to merge: {e}")
                return await self._hybrid_search_merge(
                    query, limit, sources, vector_weight, candidate_multiplier,
                )
        else:
            return await self._hybrid_search_merge(
                query, limit, sources, vector_weight, candidate_multiplier,
            )

        search_results = []
        raw_ids = results.get("ids") or []
        ids = raw_ids[0] if raw_ids and isinstance(raw_ids[0], list) else raw_ids
        if not ids:
            return []
        documents = results.get("documents")
        metadatas = results.get("metadatas")
        distances = results.get("distances")
        doc_list = (documents[0] if documents and isinstance(documents[0], list) else documents) or []
        meta_list = (metadatas[0] if metadatas and isinstance(metadatas[0], list) else metadatas) or []
        dist_list = (distances[0] if distances and isinstance(distances[0], list) else distances) or []
        for i, chunk_id in enumerate(ids):
            meta = meta_list[i] if i < len(meta_list) else {}
            doc = doc_list[i] if i < len(doc_list) else ""
            dist = dist_list[i] if i < len(dist_list) else 0.0
            score = max(0.0, 1.0 - dist / 2.0)
            search_results.append(
                MemorySearchResult(
                    path=meta.get("path", ""),
                    start_line=meta.get("start_line", 0),
                    end_line=meta.get("end_line", 0),
                    score=score,
                    snippet=doc,
                    source=MemorySource(meta.get("source", "memory")),
                    raw_metric=dist,
                ),
            )
        return search_results[:limit]

    async def _hybrid_search_merge(
        self,
        query: str,
        limit: int,
        sources: list[MemorySource] | None,
        vector_weight: float,
        candidate_multiplier: float,
    ) -> list[MemorySearchResult]:
        """Fallback: merge vector and keyword results like Chroma."""
        candidates = min(200, max(1, int(limit * candidate_multiplier)))
        text_weight = 1.0 - vector_weight

        if self.vector_enabled and self.fts_enabled:
            keyword_results = await self.keyword_search(query, candidates, sources)
            vector_results = await self.vector_search(query, candidates, sources)
            if not keyword_results:
                return vector_results[:limit]
            if not vector_results:
                return keyword_results[:limit]
            merged = self._merge_hybrid_results(
                vector=vector_results,
                keyword=keyword_results,
                vector_weight=vector_weight,
                text_weight=text_weight,
            )
            return merged[:limit]
        if self.vector_enabled:
            return await self.vector_search(query, limit, sources)
        return await self.keyword_search(query, limit, sources)

    @staticmethod
    def _merge_hybrid_results(
        vector: list[MemorySearchResult],
        keyword: list[MemorySearchResult],
        vector_weight: float,
        text_weight: float,
    ) -> list[MemorySearchResult]:
        """Merge vector and keyword results with weighted scoring."""
        merged: dict[str, MemorySearchResult] = {}
        for result in vector:
            result.score = result.score * vector_weight
            merged[result.merge_key] = result
        for result in keyword:
            key = result.merge_key
            if key in merged:
                merged[key].score += result.score * text_weight
            else:
                result.score = result.score * text_weight
                merged[key] = result
        results = list(merged.values())
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    async def clear_all(self) -> None:
        """Clear all indexed data (collection + files table)."""
        self.client.delete_collection(self.collection_name)
        dim = self.embedding_dim
        config = Configuration(
            hnsw=HNSWConfiguration(dimension=dim, distance="cosine"),
            fulltext_config=FulltextIndexConfig(analyzer="space"),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            configuration=config,
            embedding_function=None,
        )
        self._execute_sql(f"DELETE FROM `{self.files_table_name}`")
        logger.info(f"Cleared all data from seekdb collection: {self.collection_name} and files table")

    async def close(self) -> None:
        """Close client (file metadata is in DB, no persist needed)."""
        self.client = None
        self.collection = None
