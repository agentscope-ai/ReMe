"""ChromaDB chunk storage backend."""

import time

from .base_chunk_store import BaseChunkStore
from ..component_registry import R
from ...schema import ChunkFilter, FileChunk

try:
    import chromadb
    from chromadb.config import Settings

    _CHROMADB_IMPORT_ERROR: Exception | None = None
except Exception as e:
    _CHROMADB_IMPORT_ERROR = e
    chromadb = None
    Settings = None


@R.register("chroma")
class ChromaChunkStore(BaseChunkStore):
    """ChromaDB chunk storage with vector and full-text search.

    Uses ChromaDB's native vector search and `where_document` $contains
    for keyword matching.
    """

    def __init__(self, **kwargs):
        if _CHROMADB_IMPORT_ERROR is not None:
            raise _CHROMADB_IMPORT_ERROR
        super().__init__(**kwargs)
        self.client: "chromadb.ClientAPI | None" = None
        self.chunks_collection: "chromadb.Collection | None" = None

    @property
    def collection_name(self) -> str:
        return f"chunks_{self.store_name}"

    # -- Lifecycle ----------------------------------------------------------

    async def _start(self, app_context=None) -> None:
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(allow_reset=True, anonymized_telemetry=False),
        )
        self.chunks_collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.logger.info(f"ChromaChunkStore '{self.store_name}' ready: collection={self.collection_name}")
        await super()._start(app_context)

    async def _close(self) -> None:
        self.client = None
        self.chunks_collection = None
        await super()._close()

    # -- Filter helper ------------------------------------------------------

    @staticmethod
    def _path_where(chunk_filter: ChunkFilter | None) -> dict | None:
        if chunk_filter is None or chunk_filter.resolved_paths is None:
            return None
        paths = chunk_filter.resolved_paths
        if not paths:
            return {"path": "__nonexistent__"}
        if len(paths) == 1:
            return {"path": next(iter(paths))}
        return {"path": {"$in": list(paths)}}

    # -- Write operations ---------------------------------------------------

    async def upsert_chunks(self, path: str, chunks: list[FileChunk]) -> None:
        await self.delete_chunks(path)

        if not chunks:
            return

        chunks = await self.get_chunk_embeddings(chunks)

        ids, documents, embeddings, metadatas = [], [], [], []
        now = int(time.time() * 1000)
        for chunk in chunks:
            ids.append(chunk.id)
            documents.append(chunk.text)
            embeddings.append(chunk.embedding if chunk.embedding else [0.0] * self.embedding_dim)
            metadatas.append({
                "path": path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "hash": chunk.hash,
                "updated_at": now,
            })

        self.chunks_collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    async def delete_chunks(self, path: str) -> None:
        results = self.chunks_collection.get(where={"path": path}, include=[])
        if results["ids"]:
            self.chunks_collection.delete(ids=results["ids"])

    # -- Read operations ----------------------------------------------------

    async def get_chunks(self, path: str) -> list[FileChunk]:
        results = self.chunks_collection.get(where={"path": path}, include=["documents", "metadatas"])
        chunks: list[FileChunk] = []
        for cid, md, text in zip(results["ids"], results["metadatas"], results["documents"]):
            chunks.append(self._chunk_from_chroma(cid, md, text))
        chunks.sort(key=lambda c: c.start_line)
        return chunks

    # -- Search helpers -----------------------------------------------------

    @staticmethod
    def _chunk_from_chroma(chunk_id: str, md: dict, text: str, embedding=None) -> FileChunk:
        return FileChunk(
            id=chunk_id,
            path=md["path"],
            start_line=md["start_line"],
            end_line=md["end_line"],
            text=text,
            hash=md["hash"],
            embedding=embedding,
        )

    # -- Search operations --------------------------------------------------

    async def vector_search(
            self,
            query: str,
            limit: int,
            chunk_filter: ChunkFilter | None = None,
    ) -> list[FileChunk]:
        if not self.vector_enabled or not query:
            return []

        query_embedding = await self.get_embedding(query)
        if not query_embedding:
            return []

        try:
            results = self.chunks_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=self._path_where(chunk_filter),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []

        chunks = []
        if results["ids"] and results["ids"][0]:
            for i, cid in enumerate(results["ids"][0]):
                md = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                score = max(0.0, 1.0 - distance / 2.0)

                chunk = self._chunk_from_chroma(cid, md, results["documents"][0][i])
                chunk.scores = {"vector": score, "score": score}
                chunks.append(chunk)

        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks[:limit]

    async def keyword_search(
            self,
            query: str,
            limit: int,
            chunk_filter: ChunkFilter | None = None,
    ) -> list[FileChunk]:
        if not self.fts_enabled or not query:
            return []

        words = query.split()
        if not words:
            return []

        word_variants = set()
        for word in words:
            word_variants.add(word)
            word_variants.add(word.lower())
            word_variants.add(word.capitalize())
            word_variants.add(word.upper())
        variants_list = list(word_variants)

        if len(variants_list) == 1:
            where_document: dict = {"$contains": variants_list[0]}
        else:
            where_document = {"$or": [{"$contains": w} for w in variants_list]}

        results = self.chunks_collection.get(
            where=self._path_where(chunk_filter),
            where_document=where_document,
            include=["documents", "metadatas"],
        )

        chunks = []
        for i, cid in enumerate(results["ids"]):
            md = results["metadatas"][i]
            text = results["documents"][i]
            score = self._score_keyword_match(query, text)
            if score == 0.0:
                continue

            chunk = self._chunk_from_chroma(cid, md, text)
            chunk.scores = {"keyword": score, "score": score}
            chunks.append(chunk)

        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks[:limit]

    # -- Clear --------------------------------------------------------------

    async def clear_all(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.chunks_collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.logger.info(f"Cleared all data from ChromaChunkStore '{self.store_name}'")
