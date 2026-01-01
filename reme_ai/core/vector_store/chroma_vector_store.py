"""ChromaDB vector store implementation for the ReMe framework."""

from typing import Any

from loguru import logger

from .base_vector_store import BaseVectorStore
from ..context import C
from ..embedding import BaseEmbeddingModel
from ..schema import VectorNode

_CHROMADB_IMPORT_ERROR = None

try:
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    _CHROMADB_IMPORT_ERROR = e
    chromadb = None
    Settings = None


@C.register_vector_store("chroma")
class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-based vector store implementation for local or remote storage."""

    def __init__(
        self,
        collection_name: str,
        embedding_model: BaseEmbeddingModel,
        client: chromadb.ClientAPI | None = None,
        host: str | None = None,
        port: int | None = None,
        path: str | None = None,
        api_key: str | None = None,
        tenant: str | None = None,
        database: str | None = None,
        **kwargs,
    ):
        """Initialize the ChromaDB vector store with the provided configuration."""
        if _CHROMADB_IMPORT_ERROR is not None:
            raise ImportError(
                "ChromaDB requires extra dependencies. Install with `pip install chromadb`",
            ) from _CHROMADB_IMPORT_ERROR

        super().__init__(
            collection_name=collection_name,
            embedding_model=embedding_model,
            **kwargs,
        )

        self.client: chromadb.ClientAPI
        self.collection: chromadb.Collection

        if client:
            self.client = client
        elif api_key and tenant:
            logger.info("Initializing ChromaDB Cloud client")
            self.client = chromadb.CloudClient(
                api_key=api_key,
                tenant=tenant,
                database=database or "default",
            )
        elif host and port:
            logger.info(f"Initializing ChromaDB HTTP client at {host}:{port}")
            self.client = chromadb.HttpClient(host=host, port=port)
        else:
            if path is None:
                path = "./chroma_db"
            logger.info(f"Initializing local ChromaDB at {path}")
            self.client = chromadb.PersistentClient(
                path=path,
                settings=Settings(anonymized_telemetry=False),
            )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _parse_results(
        results: dict,
        include_score: bool = False,
    ) -> list[VectorNode]:
        """Convert ChromaDB query results into a list of VectorNode objects."""
        nodes = []

        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        embeddings = results.get("embeddings") if results.get("embeddings") is not None else []
        distances = results.get("distances") if results.get("distances") is not None else []

        if ids and isinstance(ids[0], list):
            ids = ids[0] if ids else []
            documents = documents[0] if documents else []
            metadatas = metadatas[0] if metadatas else []
            embeddings = embeddings[0] if embeddings and len(embeddings) > 0 else []
            distances = distances[0] if distances and len(distances) > 0 else []

        for i, vector_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}

            if include_score and distances and i < len(distances):
                metadata["_score"] = 1.0 - distances[i]

            node = VectorNode(
                vector_id=vector_id,
                content=documents[i] if i < len(documents) and documents[i] else "",
                vector=embeddings[i] if len(embeddings) > i else None,
                metadata=metadata,
            )
            nodes.append(node)

        return nodes

    @staticmethod
    def _generate_where_clause(filters: dict | None) -> dict | None:
        """Convert the universal filter format to a ChromaDB-compatible where clause."""
        if not filters:
            return None

        def convert_condition(k: str, v: Any) -> dict | None:
            """Convert a single filter condition to ChromaDB operator format."""
            if v == "*":
                return None
            if isinstance(v, dict):
                chroma_condition = {}
                for op, val in v.items():
                    mapping = {
                        "eq": "$eq",
                        "ne": "$ne",
                        "gt": "$gt",
                        "gte": "$gte",
                        "lt": "$lt",
                        "lte": "$lte",
                        "in": "$in",
                        "nin": "$nin",
                    }
                    chroma_op = mapping.get(op, "$eq")
                    chroma_condition[k] = {chroma_op: val}
                return chroma_condition
            if isinstance(v, list):
                return {k: {"$in": v}}
            return {k: {"$eq": v}}

        processed_filters = []

        for key, value in filters.items():
            if key == "$or":
                or_conditions = []
                for condition in value:
                    or_condition = {}
                    for sub_key, sub_value in condition.items():
                        converted = convert_condition(sub_key, sub_value)
                        if converted:
                            or_condition.update(converted)
                    if or_condition:
                        or_conditions.append(or_condition)
                if len(or_conditions) > 1:
                    processed_filters.append({"$or": or_conditions})
                elif len(or_conditions) == 1:
                    processed_filters.append(or_conditions[0])

            elif key == "$and":
                for condition in value:
                    for sub_key, sub_value in condition.items():
                        converted = convert_condition(sub_key, sub_value)
                        if converted:
                            processed_filters.append(converted)
            elif key == "$not":
                continue
            else:
                converted = convert_condition(key, value)
                if converted:
                    processed_filters.append(converted)

        if not processed_filters:
            return None
        return processed_filters[0] if len(processed_filters) == 1 else {"$and": processed_filters}

    async def list_collections(self) -> list[str]:
        """Retrieve a list of all existing collection names."""

        def _list():
            return [col.name for col in self.client.list_collections()]

        return await self._run_sync_in_executor(_list)

    async def create_collection(self, collection_name: str, **kwargs):
        """Create a new collection with specified distance metrics and metadata."""

        def _create():
            distance_metric = kwargs.get("distance_metric", "cosine")
            metadata = kwargs.get("metadata", {})
            metadata["hnsw:space"] = distance_metric
            return self.client.get_or_create_collection(name=collection_name, metadata=metadata)

        new_collection = await self._run_sync_in_executor(_create)
        if collection_name == self.collection_name:
            self.collection = new_collection
        logger.info(f"Created collection {collection_name}")

    async def delete_collection(self, collection_name: str, **kwargs):
        """Delete a specified collection from the database."""

        def _delete():
            try:
                self.client.delete_collection(name=collection_name)
                return True
            except Exception as e:
                logger.warning(f"Failed to delete collection {collection_name}: {e}")
                return False

        deleted = await self._run_sync_in_executor(_delete)
        if deleted and collection_name == self.collection_name:
            self.collection = None
        logger.info(f"Deleted collection {collection_name}")

    async def copy_collection(self, collection_name: str, **kwargs):
        """Copy all data from the current collection to a new collection."""

        def _copy():
            source_data = self.collection.get(include=["documents", "metadatas", "embeddings"])
            if not source_data["ids"]:
                logger.warning(f"Source collection {self.collection_name} is empty")
                return

            target_collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            target_collection.add(
                ids=source_data["ids"],
                documents=source_data["documents"],
                metadatas=source_data["metadatas"],
                embeddings=source_data["embeddings"],
            )

        await self._run_sync_in_executor(_copy)
        logger.info(f"Copied collection {self.collection_name} to {collection_name}")

    async def insert(self, nodes: VectorNode | list[VectorNode], **kwargs):
        """Insert vector nodes into the current collection in batches."""
        if isinstance(nodes, VectorNode):
            nodes = [nodes]
        if not nodes:
            return

        # Batch generate embeddings for nodes that need them
        nodes_without_vectors = [node for node in nodes if node.vector is None]
        if nodes_without_vectors:
            nodes_with_vectors = await self.get_node_embeddings(nodes_without_vectors)
            # Create a mapping for quick lookup
            vector_map = {n.vector_id: n for n in nodes_with_vectors}
            nodes_to_insert = [vector_map.get(n.vector_id, n) if n.vector is None else n for n in nodes]
        else:
            nodes_to_insert = nodes

        batch_size = kwargs.get("batch_size", 100)

        def _insert_batch(batch_nodes: list[VectorNode]):
            self.collection.add(
                ids=[n.vector_id for n in batch_nodes],
                documents=[n.content for n in batch_nodes],
                embeddings=[n.vector for n in batch_nodes],
                metadatas=[n.metadata for n in batch_nodes],
            )

        for i in range(0, len(nodes_to_insert), batch_size):
            await self._run_sync_in_executor(_insert_batch, nodes_to_insert[i : i + batch_size])
        logger.info(f"Inserted {len(nodes_to_insert)} nodes into {self.collection_name}")

    async def search(
        self,
        query: str,
        limit: int = 5,
        filters: dict | None = None,
        **kwargs,
    ) -> list[VectorNode]:
        """Search for the most similar vector nodes based on a text query."""
        query_vector = await self.get_embedding(query)
        where_clause = self._generate_where_clause(filters)
        include_embeddings = kwargs.get("include_embeddings", False)

        def _search():
            include: list = ["documents", "metadatas", "distances"]
            if include_embeddings:
                include.append("embeddings")
            return self.collection.query(
                query_embeddings=[query_vector],
                n_results=limit,
                where=where_clause,
                include=include,
            )

        results = await self._run_sync_in_executor(_search)
        nodes = self._parse_results(results, include_score=True)

        score_threshold = kwargs.get("score_threshold")
        if score_threshold is not None:
            nodes = [n for n in nodes if n.metadata.get("_score", 0) >= score_threshold]
        return nodes

    async def delete(self, vector_ids: str | list[str], **kwargs):
        """Delete specific vector nodes by their IDs."""
        if isinstance(vector_ids, str):
            vector_ids = [vector_ids]
        if not vector_ids:
            return

        def _delete():
            self.collection.delete(ids=vector_ids)

        await self._run_sync_in_executor(_delete)
        logger.info(f"Deleted {len(vector_ids)} nodes from {self.collection_name}")

    async def update(self, nodes: VectorNode | list[VectorNode], **kwargs):
        """Update existing vector nodes with new content or metadata."""
        if isinstance(nodes, VectorNode):
            nodes = [nodes]
        if not nodes:
            return

        # Batch generate embeddings for nodes that need them
        nodes_without_vectors = [node for node in nodes if node.vector is None and node.content]
        if nodes_without_vectors:
            nodes_with_vectors = await self.get_node_embeddings(nodes_without_vectors)
            # Create a mapping for quick lookup
            vector_map = {n.vector_id: n for n in nodes_with_vectors}
            nodes_to_update = [vector_map.get(n.vector_id, n) if n.vector is None and n.content else n for n in nodes]
        else:
            nodes_to_update = nodes

        def _update():
            self.collection.upsert(
                ids=[n.vector_id for n in nodes_to_update],
                documents=[n.content for n in nodes_to_update],
                embeddings=[n.vector for n in nodes_to_update if n.vector] or None,
                metadatas=[n.metadata for n in nodes_to_update],
            )

        await self._run_sync_in_executor(_update)
        logger.info(f"Updated {len(nodes_to_update)} nodes in {self.collection_name}")

    async def get(self, vector_ids: str | list[str]) -> VectorNode | list[VectorNode] | None:
        """Fetch vector nodes by their IDs from the collection."""
        is_single = isinstance(vector_ids, str)
        ids = [vector_ids] if is_single else vector_ids

        def _get():
            return self.collection.get(ids=ids, include=["documents", "metadatas", "embeddings"])

        results = await self._run_sync_in_executor(_get)
        nodes = self._parse_results(results)
        return nodes[0] if is_single and nodes else (nodes if not is_single else None)

    async def list(
        self,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[VectorNode]:
        """List vector nodes matching optional metadata filters."""
        where_clause = self._generate_where_clause(filters)

        def _list():
            return self.collection.get(
                where=where_clause,
                limit=limit,
                include=["documents", "metadatas", "embeddings"],
            )

        results = await self._run_sync_in_executor(_list)
        return self._parse_results(results)

    async def count(self) -> int:
        """Return the total number of vectors in the current collection."""
        return await self._run_sync_in_executor(self.collection.count)

    async def reset(self):
        """Reset the current collection by clearing all its data."""
        logger.warning(f"Resetting collection {self.collection_name}...")
        await self.delete_collection(self.collection_name)

        def _recreate():
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        await self._run_sync_in_executor(_recreate)
        logger.info(f"Collection {self.collection_name} has been reset")

    async def close(self):
        """Close the vector store and log the shutdown process."""
        logger.info(f"ChromaDB vector store for collection {self.collection_name} closed")
