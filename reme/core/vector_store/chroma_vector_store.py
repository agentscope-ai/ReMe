"""ChromaDB vector store implementation for the ReMe framework."""

from pathlib import Path
from typing import Any

from loguru import logger

from .base_vector_store import BaseVectorStore
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


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-based vector store implementation for local or remote storage."""

    def __init__(
        self,
        collection_name: str,
        db_path: str | Path,
        embedding_model: BaseEmbeddingModel,
        client: chromadb.ClientAPI | None = None,
        host: str | None = None,
        port: int | None = None,
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
            db_path=db_path,
            embedding_model=embedding_model,
            **kwargs,
        )

        self.client: chromadb.ClientAPI
        self.collection: chromadb.Collection
        self.is_local = client is None and not (api_key and tenant) and not (host and port)

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
            self.client = None  # Will be initialized in start()

        self.collection: chromadb.Collection | None = None

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
                metadata["score"] = 1.0 - distances[i]

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
        """Convert the universal filter format to a ChromaDB-compatible where clause.

        Supports two filter formats:
        1. Range query: {"field": [start_value, end_value]} - filters for field >= start_value AND field <= end_value
        2. Exact match: {"field": value} - filters for field == value
        """
        if not filters:
            return None

        def convert_condition(k: str, v: Any) -> dict | list | None:
            """Convert a single filter condition to ChromaDB operator format.

            Returns:
                - dict for simple conditions
                - list of dicts for range queries (which need to be wrapped in $and)
                - None for wildcard filters
            """
            if v == "*":
                return None
            # New syntax: [start, end] represents a range query
            if isinstance(v, list) and len(v) == 2:
                # Range query: field >= v[0] AND field <= v[1]
                # ChromaDB requires separate conditions combined with $and
                return [
                    {k: {"$gte": v[0]}},
                    {k: {"$lte": v[1]}},
                ]
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
            # Exact match for non-list values
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
                            if isinstance(converted, list):
                                # Range query in OR condition - need to wrap in $and
                                or_conditions.append({"$and": converted})
                            else:
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
                            if isinstance(converted, list):
                                # Range query - add each condition separately
                                processed_filters.extend(converted)
                            else:
                                processed_filters.append(converted)
            elif key == "$not":
                continue
            else:
                converted = convert_condition(key, value)
                if converted:
                    if isinstance(converted, list):
                        # Range query - add each condition separately
                        processed_filters.extend(converted)
                    else:
                        processed_filters.append(converted)

        if not processed_filters:
            return None
        return processed_filters[0] if len(processed_filters) == 1 else {"$and": processed_filters}

    async def list_collections(self) -> list[str]:
        """Retrieve a list of all existing collection names."""
        return [col.name for col in self.client.list_collections()]

    async def create_collection(self, collection_name: str, **kwargs):
        """Create a new collection with specified distance metrics and metadata."""
        distance_metric = kwargs.get("distance_metric", "cosine")
        metadata = kwargs.get("metadata", {})
        metadata["hnsw:space"] = distance_metric
        new_collection = self.client.get_or_create_collection(name=collection_name, metadata=metadata)
        if collection_name == self.collection_name:
            self.collection = new_collection
        logger.info(f"Created collection `{collection_name}`")

    async def delete_collection(self, collection_name: str, **kwargs):
        """Delete a specified collection from the database."""
        try:
            self.client.delete_collection(name=collection_name)
            deleted = True
        except Exception as _e:
            logger.warning(f"Failed to delete collection {collection_name}: {_e}")
            deleted = False
        if deleted and collection_name == self.collection_name:
            self.collection = None
        logger.info(f"Deleted collection {collection_name}")

    async def copy_collection(self, collection_name: str, **kwargs):
        """Copy all data from the current collection to a new collection."""
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

        for i in range(0, len(nodes_to_insert), batch_size):
            batch_nodes = nodes_to_insert[i : i + batch_size]
            self.collection.add(
                ids=[n.vector_id for n in batch_nodes],
                documents=[n.content for n in batch_nodes],
                embeddings=[n.vector for n in batch_nodes],
                metadatas=[n.metadata for n in batch_nodes],
            )
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

        include: list = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            where=where_clause,
            include=include,
        )
        nodes = self._parse_results(results, include_score=True)

        score_threshold = kwargs.get("score_threshold")
        if score_threshold is not None:
            nodes = [n for n in nodes if n.metadata.get("score", 0) >= score_threshold]
        return nodes

    async def delete(self, vector_ids: str | list[str], **kwargs):
        """Delete specific vector nodes by their IDs."""
        if isinstance(vector_ids, str):
            vector_ids = [vector_ids]
        if not vector_ids:
            return

        self.collection.delete(ids=vector_ids)
        logger.info(f"Deleted {len(vector_ids)} nodes from {self.collection_name}")

    async def delete_all(self, **kwargs):
        """Remove all vectors from the collection."""
        # Get all IDs in the collection
        result = self.collection.get()
        count = 0
        if result and result.get("ids"):
            ids = result["ids"]
            if ids:
                self.collection.delete(ids=ids)
                count = len(ids)
        logger.info(f"Deleted all {count} nodes from {self.collection_name}")

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

        self.collection.upsert(
            ids=[n.vector_id for n in nodes_to_update],
            documents=[n.content for n in nodes_to_update],
            embeddings=[n.vector for n in nodes_to_update],
            metadatas=[n.metadata for n in nodes_to_update],
        )
        logger.info(f"Updated {len(nodes_to_update)} nodes in {self.collection_name}")

    async def get(self, vector_ids: str | list[str]) -> VectorNode | list[VectorNode] | None:
        """Fetch vector nodes by their IDs from the collection."""
        is_single = isinstance(vector_ids, str)
        ids = [vector_ids] if is_single else vector_ids

        results = self.collection.get(ids=ids, include=["documents", "metadatas", "embeddings"])
        nodes = self._parse_results(results)
        return nodes[0] if is_single and nodes else (nodes if not is_single else None)

    async def list(
        self,
        filters: dict | None = None,
        limit: int | None = None,
        sort_key: str | None = None,
        reverse: bool = False,
    ) -> list[VectorNode]:
        """List vector nodes matching optional metadata filters.

        Args:
            filters: Dictionary of filter conditions to match vectors
            limit: Maximum number of vectors to return
            sort_key: Key to sort the results by (e.g., field name in metadata). None for no sorting
            reverse: If True, sort in descending order; if False, sort in ascending order
        """
        where_clause = self._generate_where_clause(filters)

        # If sorting is needed, fetch all records first, then apply limit after sorting
        fetch_limit = None if sort_key else limit

        results = self.collection.get(
            where=where_clause,
            limit=fetch_limit,
            include=["documents", "metadatas", "embeddings"],
        )
        nodes = self._parse_results(results)

        # Apply sorting if sort_key is provided
        if sort_key:
            # Sort with proper handling of None and missing values
            def sort_key_func(node):
                value = node.metadata.get(sort_key)
                if value is None:
                    # Return appropriate default based on reverse flag
                    return float("-inf") if not reverse else float("inf")
                return value

            nodes.sort(key=sort_key_func, reverse=reverse)

            # Apply limit after sorting
            if limit is not None:
                nodes = nodes[:limit]

        return nodes

    async def count(self) -> int:
        """Return the total number of vectors in the current collection."""
        return self.collection.count()

    async def reset(self):
        """Reset the current collection by clearing all its data."""
        logger.warning(f"Resetting collection {self.collection_name}...")
        await self.delete_collection(self.collection_name)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Collection {self.collection_name} has been reset")

    async def start(self) -> None:
        """Initialize the ChromaDB collection.

        Creates or retrieves the collection with cosine similarity metric.
        For local mode, creates the db_path directory if it doesn't exist.
        """
        if self.is_local:
            self.db_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initializing local ChromaDB at {self.db_path}")
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(anonymized_telemetry=False),
            )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB collection {self.collection_name} initialized")

    async def close(self):
        """Close the vector store and log the shutdown process."""
        logger.info(f"ChromaDB vector store for collection {self.collection_name} closed")
