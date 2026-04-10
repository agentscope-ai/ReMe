"""seekdb vector store implementation for the ReMe framework."""

from pathlib import Path
from typing import Any

from loguru import logger

from .base_vector_store import BaseVectorStore
from ..embedding import BaseEmbeddingModel
from ..schema import VectorNode

# Optional: preserve original exception for "raise ... from _PYSEEKDB_IMPORT_ERROR" (better diagnostics)
_PYSEEKDB_IMPORT_ERROR = None

try:
    import pyseekdb
    from pyseekdb import Configuration, HNSWConfiguration

    PYSEEKDB_AVAILABLE = True
except ImportError as e:
    _PYSEEKDB_IMPORT_ERROR = e
    pyseekdb = None
    Configuration = None
    HNSWConfiguration = None


class SeekdbVectorStore(BaseVectorStore):
    """Vector store implementation using seekdb (pyseekdb) for embedded vector search.

    Uses the same embedded Client + Collection API as SeekdbFileStore; supports
    vector similarity search and metadata filtering. No full-text index by default
    (vector_store use case is vector search + filter).
    """

    def __init__(
        self,
        collection_name: str,
        db_path: str | Path,
        embedding_model: BaseEmbeddingModel,
        database: str = "reme_vector",
        distance: str = "cosine",
        **kwargs: Any,
    ):
        """Initialize the seekdb vector store.

        Args:
            collection_name: Name of the collection.
            db_path: Local path for embedded seekdb storage (e.g. working_dir / "vector_store").
            embedding_model: Model used for generating vector embeddings.
            database: seekdb database name (all vector_store collections live in this DB).
            distance: Similarity metric: cosine, euclid, dot.
            **kwargs: Additional options (ignored for compatibility).
        """
        if _PYSEEKDB_IMPORT_ERROR is not None:
            raise ImportError(
                "seekdb vector store requires pyseekdb. Install with `pip install pyseekdb` or `pip install reme-ai`",
            ) from _PYSEEKDB_IMPORT_ERROR

        super().__init__(
            collection_name=collection_name,
            db_path=db_path,
            embedding_model=embedding_model,
            **kwargs,
        )
        self.database = database
        self.distance = distance.lower()
        self.client: Any = None
        self.collection: Any = None

    def _client_kwargs(self) -> dict:
        """Build Client kwargs for embedded mode."""
        return {
            "path": str(self.db_path / "seekdb"),
            "database": self.database,
        }

    @staticmethod
    def _build_where(filters: dict | None) -> dict | None:
        """Build seekdb/Chroma-style where clause from universal filter format.

        Supports exact match and range: {"key": value} or {"key": [start, end]}.
        """
        if not filters:
            return None
        conditions = []
        for key, value in filters.items():
            if value == "*":
                continue
            if isinstance(value, list) and len(value) == 2:
                conditions.append({key: {"$gte": value[0]}})
                conditions.append({key: {"$lte": value[1]}})
            elif isinstance(value, dict) and ("gte" in value or "lte" in value or "gt" in value or "lt" in value):
                for op, val in value.items():
                    if op in ("gte", "lte", "gt", "lt") and val is not None:
                        conditions.append({key: {"$" + op: val}})
            else:
                conditions.append({key: {"$eq": value}})
        if not conditions:
            return None
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    @staticmethod
    def _parse_results(
        ids: list,
        documents: list | None = None,
        metadatas: list | None = None,
        embeddings: list | None = None,
        distances: list | None = None,
        include_score: bool = False,
    ) -> list[VectorNode]:
        """Convert seekdb get/query result to list of VectorNode."""
        nodes = []
        documents = documents or []
        metadatas = metadatas or []
        embeddings = embeddings or []
        distances = distances or []
        if ids and isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
            if documents and isinstance(documents[0], list):
                documents = documents[0]
            if metadatas and isinstance(metadatas[0], list):
                metadatas = metadatas[0]
            if embeddings and isinstance(embeddings[0], list):
                embeddings = embeddings[0]
            if distances and isinstance(distances[0], (list, tuple)):
                distances = distances[0]
        for i, vector_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) and metadatas[i] is not None else {}
            if include_score and i < len(distances):
                meta = dict(meta)
                meta["score"] = 1.0 - (float(distances[i]) / 2.0) if distances[i] is not None else 0.0
            nodes.append(
                VectorNode(
                    vector_id=str(vector_id),
                    content=documents[i] if i < len(documents) and documents[i] is not None else "",
                    vector=embeddings[i] if i < len(embeddings) else None,
                    metadata=meta,
                ),
            )
        return nodes

    async def list_collections(self) -> list[str]:
        """List collection names in the current database."""
        if self.client is None:
            return []
        try:
            colls = self.client.list_collections()
            return [c.name if hasattr(c, "name") else str(c) for c in colls]
        except Exception as e:
            logger.debug("seekdb list_collections: %s", e)
            return [self.collection_name]

    async def create_collection(self, collection_name: str, **kwargs: Any) -> None:
        """Create or get collection with HNSW vector index."""
        if self.client is None:
            raise RuntimeError("seekdb client not initialized; call start() first")
        dimensions = kwargs.get("dimensions", self.embedding_model.dimensions)
        config = Configuration(
            hnsw=HNSWConfiguration(dimension=dimensions, distance=self.distance),
        )
        coll = self.client.get_or_create_collection(
            name=collection_name,
            configuration=config,
            embedding_function=None,
        )
        if collection_name == self.collection_name:
            self.collection = coll
        logger.info(f"seekdb collection {collection_name} ready (dim={dimensions})")

    async def delete_collection(self, collection_name: str, **kwargs: Any) -> None:
        """Remove the collection from the database."""
        if self.client is None:
            return
        try:
            self.client.delete_collection(collection_name)
            if collection_name == self.collection_name:
                self.collection = None
            logger.info(f"Deleted seekdb collection {collection_name}")
        except Exception as e:
            logger.warning("seekdb delete_collection %s: %s", collection_name, e)

    async def copy_collection(self, collection_name: str, **kwargs: Any) -> None:
        """Copy current collection to a new one."""
        if self.collection is None:
            raise RuntimeError("No current collection")
        data = self.collection.get(include=["documents", "metadatas", "embeddings"])
        ids = data.get("ids") or []
        if not ids:
            logger.warning("Source collection is empty")
            return
        dims = self.embedding_model.dimensions
        embs = data.get("embeddings")
        if embs and (isinstance(embs[0], list) and embs[0]) or (not isinstance(embs[0], list) and embs):
            dims = len(embs[0]) if isinstance(embs[0], list) else len(embs)
        config = Configuration(
            hnsw=HNSWConfiguration(dimension=dims, distance=self.distance),
        )
        self.client.get_or_create_collection(
            name=collection_name,
            configuration=config,
            embedding_function=None,
        )
        new_coll = self.client.get_collection(name=collection_name, embedding_function=None)
        new_coll.upsert(
            ids=ids,
            documents=data.get("documents", []),
            embeddings=data.get("embeddings", []),
            metadatas=data.get("metadatas", []),
        )
        logger.info(f"Copied {self.collection_name} to {collection_name}")

    async def insert(self, nodes: VectorNode | list[VectorNode], **kwargs: Any) -> None:
        """Insert vector nodes; generate embeddings for nodes that lack them."""
        if isinstance(nodes, VectorNode):
            nodes = [nodes]
        if not nodes:
            return
        nodes_without_vectors = [n for n in nodes if n.vector is None]
        if nodes_without_vectors:
            nodes_with_vectors = await self.get_node_embeddings(nodes_without_vectors)
            vector_map = {n.vector_id: n for n in nodes_with_vectors}
            nodes_to_insert = [vector_map.get(n.vector_id, n) if n.vector is None else n for n in nodes]
        else:
            nodes_to_insert = nodes
        ids = [n.vector_id for n in nodes_to_insert]
        documents = [n.content for n in nodes_to_insert]
        embeddings = [n.vector for n in nodes_to_insert]
        metadatas = [n.metadata for n in nodes_to_insert]
        self.collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        logger.info(f"Inserted {len(nodes_to_insert)} nodes into {self.collection_name}")

    async def search(
        self,
        query: str,
        limit: int = 5,
        filters: dict | None = None,
        **kwargs: Any,
    ) -> list[VectorNode]:
        """Vector similarity search with optional metadata filter."""
        query_vector = await self.get_embedding(query)
        where = self._build_where(filters)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = results.get("ids") or []
        documents = results.get("documents")
        metadatas = results.get("metadatas")
        distances = results.get("distances")
        nodes = self._parse_results(
            ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=results.get("embeddings"),
            distances=distances,
            include_score=True,
        )
        score_threshold = kwargs.get("score_threshold")
        if score_threshold is not None:
            nodes = [n for n in nodes if n.metadata.get("score", 0) >= score_threshold]
        return nodes

    async def delete(self, vector_ids: str | list[str], **kwargs: Any) -> None:
        """Delete points by id."""
        if isinstance(vector_ids, str):
            vector_ids = [vector_ids]
        if not vector_ids:
            return
        self.collection.delete(ids=vector_ids)
        logger.info(f"Deleted {len(vector_ids)} nodes from {self.collection_name}")

    async def delete_all(self, **kwargs: Any) -> None:
        """Remove all points from the collection."""
        data = self.collection.get(include=[])
        ids = data.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)
        logger.info(f"Deleted all {len(ids)} nodes from {self.collection_name}")

    async def update(self, nodes: VectorNode | list[VectorNode], **kwargs: Any) -> None:
        """Update nodes (upsert by id)."""
        if isinstance(nodes, VectorNode):
            nodes = [nodes]
        if not nodes:
            return
        nodes_without_vectors = [n for n in nodes if n.vector is None and n.content]
        if nodes_without_vectors:
            nodes_with_vectors = await self.get_node_embeddings(nodes_without_vectors)
            vector_map = {n.vector_id: n for n in nodes_with_vectors}
            nodes_to_update = [vector_map.get(n.vector_id, n) if n.vector is None and n.content else n for n in nodes]
        else:
            nodes_to_update = nodes
        ids = [n.vector_id for n in nodes_to_update]
        documents = [n.content for n in nodes_to_update]
        embeddings = [n.vector for n in nodes_to_update]
        metadatas = [n.metadata for n in nodes_to_update]
        self.collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        logger.info(f"Updated {len(nodes_to_update)} nodes in {self.collection_name}")

    async def get(self, vector_ids: str | list[str]) -> VectorNode | list[VectorNode]:
        """Fetch nodes by id."""
        single = isinstance(vector_ids, str)
        ids = [vector_ids] if single else list(vector_ids)
        if not ids:
            return None if single else []
        results = self.collection.get(
            ids=ids,
            include=["documents", "metadatas", "embeddings"],
        )
        rids = results.get("ids") or []
        nodes = self._parse_results(
            rids,
            documents=results.get("documents"),
            metadatas=results.get("metadatas"),
            embeddings=results.get("embeddings"),
        )
        if single:
            return nodes[0] if nodes else None
        return nodes

    async def list(
        self,
        filters: dict | None = None,
        limit: int | None = None,
        sort_key: str | None = None,
        reverse: bool = True,
    ) -> list[VectorNode]:
        """List nodes with optional filter, limit, and sort by metadata key."""
        where = self._build_where(filters)
        # When sorting in memory, fetch candidates first (cap like default list); do not pass
        # user limit to get() or we sort an arbitrary first page only (see ChromaVectorStore.list).
        if sort_key is not None:
            fetch_limit = 10000
        else:
            fetch_limit = limit if limit is not None else 10000
        results = self.collection.get(
            where=where,
            limit=fetch_limit,
            include=["documents", "metadatas", "embeddings"],
        )
        ids = results.get("ids") or []
        nodes = self._parse_results(
            ids,
            documents=results.get("documents"),
            metadatas=results.get("metadatas"),
            embeddings=results.get("embeddings"),
        )
        if sort_key:
            def key_fn(n):
                v = n.metadata.get(sort_key)
                if v is None:
                    return float("-inf") if not reverse else float("inf")
                return v
            nodes.sort(key=key_fn, reverse=reverse)
        if limit is not None:
            nodes = nodes[:limit]
        return nodes

    async def start(self) -> None:
        """Initialize seekdb client and ensure collection exists."""
        path = Path(self._client_kwargs()["path"])
        path.mkdir(parents=True, exist_ok=True)
        try:
            admin = pyseekdb.AdminClient(path=str(path))
            if not any(db.name == self.database for db in admin.list_databases()):
                admin.create_database(self.database)
        except Exception as e:
            logger.debug("seekdb AdminClient: %s", e)
        self.client = pyseekdb.Client(**self._client_kwargs())
        await self.create_collection(self.collection_name)
        logger.info(f"seekdb vector store {self.collection_name} initialized")

    async def close(self) -> None:
        """Release client; no explicit close in pyseekdb, clear references."""
        self.client = None
        self.collection = None
        logger.info("seekdb vector store closed")
