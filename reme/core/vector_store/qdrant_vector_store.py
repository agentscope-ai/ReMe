"""Qdrant vector store implementation for the ReMe project."""

from pathlib import Path
from typing import Any

from loguru import logger

from .base_vector_store import BaseVectorStore
from ..embedding import BaseEmbeddingModel
from ..schema import VectorNode

_QDRANT_IMPORT_ERROR = None

try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointIdsList,
        PointStruct,
        Range,
        VectorParams,
    )
except ImportError as e:
    _QDRANT_IMPORT_ERROR = e
    AsyncQdrantClient = None
    Distance = None
    FieldCondition = None
    Filter = None
    MatchValue = None
    PointIdsList = None
    PointStruct = None
    Range = None
    VectorParams = None


class QdrantVectorStore(BaseVectorStore):
    """Vector store implementation using Qdrant for dense vector search."""

    def __init__(
        self,
        collection_name: str,
        db_path: str | Path,
        embedding_model: BaseEmbeddingModel,
        host: str | None = None,
        port: int = 6333,
        url: str | None = None,
        api_key: str | None = None,
        https: bool | None = None,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        distance: str = "cosine",
        on_disk: bool = False,
        **kwargs: Any,
    ):
        """Initialize the Qdrant client and collection configuration.

        Args:
            collection_name: Name of the collection.
            db_path: Local storage path for on-disk/in-memory mode.
            embedding_model: Model used for generating vector embeddings.
            host: Server host address.
            port: HTTP port for the server.
            url: Full connection URL.
            api_key: Authentication key for Qdrant Cloud.
            https: Use secure connection if True.
            grpc_port: gRPC interface port.
            prefer_grpc: Use gRPC instead of HTTP if True.
            distance: Metric for similarity (cosine, euclid, dot).
            on_disk: Enable persistent storage for vectors.
            **kwargs: Additional client configuration.
        """
        if _QDRANT_IMPORT_ERROR is not None:
            raise ImportError(
                "Qdrant requires extra dependencies. Install with `pip install qdrant-client`",
            ) from _QDRANT_IMPORT_ERROR

        super().__init__(
            collection_name=collection_name,
            db_path=db_path,
            embedding_model=embedding_model,
            **kwargs,
        )

        self.is_local = host is None and url is None and api_key is None
        self.client: AsyncQdrantClient

        # Store connection parameters for deferred initialization in start()
        self._host = host
        self._port = port
        self._url = url
        self._api_key = api_key
        self._https = https
        self._grpc_port = grpc_port
        self._prefer_grpc = prefer_grpc

        distance_map = {
            "cosine": Distance.COSINE,
            "euclid": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        self.distance = distance_map.get(distance.lower(), Distance.COSINE)
        self.on_disk = on_disk

    async def list_collections(self) -> list[str]:
        """Retrieve names of all existing collections in the Qdrant instance."""
        collections = await self.client.get_collections()
        return [collection.name for collection in collections.collections]

    async def create_collection(self, collection_name: str, **kwargs: Any):
        """Create a new collection with the specified vector configuration.

        Args:
            collection_name: Name of the collection to create.
            **kwargs: Overrides for dimensions, distance, or on_disk settings.
        """
        collections = await self.list_collections()
        if collection_name in collections:
            logger.info(f"Collection {collection_name} already exists")
            return

        dimensions = kwargs.get("dimensions", self.embedding_model.dimensions)
        distance = kwargs.get("distance", self.distance)
        on_disk = kwargs.get("on_disk", self.on_disk)

        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dimensions,
                distance=distance,
                on_disk=on_disk,
            ),
        )

        logger.info(f"Created collection {collection_name} with dimensions={dimensions}")

        if not self.is_local:
            await self._create_payload_indexes(collection_name)

    async def _create_payload_indexes(self, collection_name: str):
        """Create keyword indexes for common metadata fields to optimize filtering."""
        common_fields = ["user_id", "agent_id", "run_id", "actor_id", "source"]

        for field in common_fields:
            try:
                await self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema="keyword",
                )
                logger.debug(f"Created index for {field} in collection {collection_name}")
            except Exception as e:
                logger.debug(f"Index for {field} might already exist: {e}")

    async def delete_collection(self, collection_name: str, **kwargs: Any):
        """Permanently remove a collection from the Qdrant instance."""
        collections = await self.list_collections()
        if collection_name in collections:
            await self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted collection {collection_name}")
        else:
            logger.warning(f"Collection {collection_name} does not exist")

    async def copy_collection(self, collection_name: str, **kwargs: Any):
        """Duplicate an existing collection to a new one including all data."""
        collection_info = await self.client.get_collection(collection_name=self.collection_name)

        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config=collection_info.config.params.vectors,
        )

        offset = None
        batch_size = 100

        while True:
            records, next_offset = await self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )

            if not records:
                break

            points = [
                PointStruct(
                    id=record.id,
                    vector=record.vector,
                    payload=record.payload,
                )
                for record in records
            ]

            await self.client.upsert(
                collection_name=collection_name,
                points=points,
            )

            offset = next_offset
            if offset is None:
                break

        logger.info(f"Copied collection {self.collection_name} to {collection_name}")

    async def insert(self, nodes: VectorNode | list[VectorNode], **kwargs: Any):
        """Insert vector nodes into the collection, generating embeddings as needed."""
        if isinstance(nodes, VectorNode):
            nodes = [nodes]

        nodes_without_vectors = [node for node in nodes if node.vector is None]
        if nodes_without_vectors:
            nodes_with_vectors = await self.get_node_embeddings(nodes_without_vectors)
            vector_map = {n.vector_id: n for n in nodes_with_vectors}
            nodes_to_insert = [vector_map.get(n.vector_id, n) if n.vector is None else n for n in nodes]
        else:
            nodes_to_insert = nodes

        points = []
        for node in nodes_to_insert:
            try:
                point_id = int(node.vector_id)
            except ValueError:
                point_id = abs(hash(node.vector_id)) % (10**18)

            point = PointStruct(
                id=point_id,
                vector=node.vector,
                payload={
                    "vector_id": node.vector_id,
                    "content": node.content,
                    "metadata": node.metadata,
                },
            )
            points.append(point)

        wait = kwargs.get("wait", True)
        await self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=wait,
        )

        logger.info(f"Inserted {len(points)} documents into {self.collection_name}")

    @staticmethod
    def _create_filter(filters: dict) -> Filter | None:
        """Convert a dictionary of filter conditions into a Qdrant Filter object.

        Supports two filter formats:
        1. Range query: {"field": [start_value, end_value]} - filters for field >= start_value AND field <= end_value
        2. Exact match: {"field": value} - filters for field == value
        """
        if not filters:
            return None

        conditions = []
        for key, value in filters.items():
            # New syntax: [start, end] represents a range query
            if isinstance(value, list) and len(value) == 2:
                # Range query: field >= value[0] AND field <= value[1]
                # Qdrant's Range only supports numeric values
                if isinstance(value[0], (int, float)) and isinstance(value[1], (int, float)):
                    conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}",
                            range=Range(gte=value[0], lte=value[1]),
                        ),
                    )
                else:
                    # For non-numeric values (e.g., string dates), Qdrant doesn't support range queries
                    # We need to skip this filter with a warning
                    logger.warning(
                        f"Qdrant does not support range queries for non-numeric values. "
                        f"Skipping range filter for key '{key}' with values {value}. "
                        f"Consider using numeric timestamps instead.",
                    )
            elif isinstance(value, dict) and ("gte" in value or "lte" in value):
                range_params = {}
                # Check if values are numeric
                if "gte" in value:
                    if isinstance(value["gte"], (int, float)):
                        range_params["gte"] = value["gte"]
                    else:
                        logger.warning(
                            f"Qdrant range filter for key '{key}' requires numeric gte value, "
                            f"got {type(value['gte']).__name__}. Skipping.",
                        )
                        continue
                if "lte" in value:
                    if isinstance(value["lte"], (int, float)):
                        range_params["lte"] = value["lte"]
                    else:
                        logger.warning(
                            f"Qdrant range filter for key '{key}' requires numeric lte value, "
                            f"got {type(value['lte']).__name__}. Skipping.",
                        )
                        continue

                if range_params:  # Only add condition if we have valid numeric parameters
                    conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}",
                            range=Range(**range_params),
                        ),
                    )
            else:
                # Exact match
                conditions.append(
                    FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value)),
                )

        return Filter(must=conditions) if conditions else None

    async def search(
        self,
        query: str,
        limit: int = 5,
        filters: dict | None = None,
        **kwargs: Any,
    ) -> list[VectorNode]:
        """Search for the most similar vectors based on a text query."""
        query_vector = await self.get_embedding(query)
        query_filter = self._create_filter(filters) if filters else None
        score_threshold = kwargs.get("score_threshold", None)

        results = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
        )

        nodes = []
        for point in results.points:
            payload = point.payload or {}
            node = VectorNode(
                vector_id=payload.get("vector_id", str(point.id)),
                content=payload.get("content", ""),
                vector=point.vector if hasattr(point, "vector") else None,
                metadata=payload.get("metadata", {}),
            )
            node.metadata["score"] = point.score
            nodes.append(node)

        return nodes

    async def delete(self, vector_ids: str | list[str], **kwargs: Any):
        """Delete specific vectors from the collection using their IDs."""
        if isinstance(vector_ids, str):
            vector_ids = [vector_ids]

        point_ids = []
        for vector_id in vector_ids:
            try:
                point_id = int(vector_id)
            except ValueError:
                point_id = abs(hash(vector_id)) % (10**18)
            point_ids.append(point_id)

        wait = kwargs.get("wait", True)
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=point_ids),
            wait=wait,
        )

        logger.info(f"Deleted {len(point_ids)} documents from {self.collection_name}")

    async def delete_all(self, **kwargs: Any):
        """Remove all vectors from the collection."""
        wait = kwargs.get("wait", True)

        # Delete all points by using an empty filter (matches all)
        from qdrant_client.models import FilterSelector

        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(filter=Filter(must=[])),
            wait=wait,
        )

        logger.info(f"Deleted all documents from {self.collection_name}")

    async def update(self, nodes: VectorNode | list[VectorNode], **kwargs: Any):
        """Update existing vector nodes with new content or metadata."""
        if isinstance(nodes, VectorNode):
            nodes = [nodes]

        nodes_without_vectors = [node for node in nodes if node.vector is None and node.content]
        if nodes_without_vectors:
            nodes_with_vectors = await self.get_node_embeddings(nodes_without_vectors)
            vector_map = {n.vector_id: n for n in nodes_with_vectors}
            nodes_to_update = [vector_map.get(n.vector_id, n) if n.vector is None and n.content else n for n in nodes]
        else:
            nodes_to_update = nodes

        points = []
        for node in nodes_to_update:
            try:
                point_id = int(node.vector_id)
            except ValueError:
                point_id = abs(hash(node.vector_id)) % (10**18)

            point = PointStruct(
                id=point_id,
                vector=node.vector,
                payload={
                    "vector_id": node.vector_id,
                    "content": node.content,
                    "metadata": node.metadata,
                },
            )
            points.append(point)

        wait = kwargs.get("wait", True)
        await self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=wait,
        )

        logger.info(f"Updated {len(points)} documents in {self.collection_name}")

    async def get(self, vector_ids: str | list[str]) -> VectorNode | list[VectorNode]:
        """Retrieve vector nodes by their IDs from the collection."""
        single_result = isinstance(vector_ids, str)
        if single_result:
            vector_ids = [vector_ids]

        point_ids = []
        for vector_id in vector_ids:
            try:
                point_id = int(vector_id)
            except ValueError:
                point_id = abs(hash(vector_id)) % (10**18)
            point_ids.append(point_id)

        points = await self.client.retrieve(
            collection_name=self.collection_name,
            ids=point_ids,
            with_payload=True,
            with_vectors=True,
        )

        results = []
        for point in points:
            if point:
                payload = point.payload or {}
                node = VectorNode(
                    vector_id=payload.get("vector_id", str(point.id)),
                    content=payload.get("content", ""),
                    vector=point.vector,
                    metadata=payload.get("metadata", {}),
                )
                results.append(node)
            else:
                logger.warning("Point not found")

        return results[0] if single_result and results else results

    async def list(
        self,
        filters: dict | None = None,
        limit: int | None = None,
        sort_key: str | None = None,
        reverse: bool = False,
    ) -> list[VectorNode]:
        """List all vector nodes in the collection matching the filter criteria.

        Args:
            filters: Dictionary of filter conditions to match vectors
            limit: Maximum number of vectors to return
            sort_key: Key to sort the results by (e.g., field name in metadata). None for no sorting
            reverse: If True, sort in descending order; if False, sort in ascending order
        """
        scroll_filter = self._create_filter(filters) if filters else None

        # If sorting is needed, fetch more records than the limit to ensure correct sorting
        fetch_limit = 10000 if sort_key else (limit or 10000)

        records, _ = await self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            limit=fetch_limit,
            with_payload=True,
            with_vectors=True,
        )

        results = []
        for record in records:
            payload = record.payload or {}
            node = VectorNode(
                vector_id=payload.get("vector_id", str(record.id)),
                content=payload.get("content", ""),
                vector=record.vector,
                metadata=payload.get("metadata", {}),
            )
            results.append(node)

        # Apply sorting if sort_key is provided
        if sort_key:
            # Sort with proper handling of None and missing values
            def sort_key_func(node):
                value = node.metadata.get(sort_key)
                if value is None:
                    # Return appropriate default based on reverse flag
                    return float("-inf") if not reverse else float("inf")
                return value

            results.sort(key=sort_key_func, reverse=reverse)

        # Apply limit after sorting
        if limit is not None:
            results = results[:limit]

        return results

    async def start(self) -> None:
        """Initialize the Qdrant collection.

        Creates the collection if it doesn't exist with configured vector parameters.
        For local mode, creates the db_path directory if it doesn't exist.
        """
        if self.is_local:
            self.db_path.mkdir(parents=True, exist_ok=True)
            self.client = AsyncQdrantClient(path=str(self.db_path))
        else:
            self.client = AsyncQdrantClient(
                host=self._host,
                port=self._port,
                url=self._url,
                api_key=self._api_key,
                https=self._https,
                grpc_port=self._grpc_port,
                prefer_grpc=self._prefer_grpc,
                **self.kwargs,
            )
        await super().start()
        logger.info(f"Qdrant collection {self.collection_name} initialized")

    async def close(self):
        """Close the AsyncQdrantClient connection and release resources."""
        await self.client.close()
        logger.info("Qdrant client connection closed")
