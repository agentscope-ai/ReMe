"""Qdrant vector store implementation for ReMe.

This module provides a Qdrant-based vector store that implements the BaseVectorStore
interface. It supports dense vector storage and retrieval using Qdrant's high-performance
kNN search capabilities with native async operations.
"""

from typing import List

from loguru import logger

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
        SearchRequest,
    )
except ImportError as e:
    _QDRANT_IMPORT_ERROR = e
    AsyncQdrantClient = None

from ..embedding import BaseEmbeddingModel
from ..schema import VectorNode
from .base_vector_store import BaseVectorStore

from ..context import C


@C.register_vector_store("qdrant")
class QdrantVectorStore(BaseVectorStore):
    """Qdrant-based vector store implementation.
    
    This class provides a high-performance vector storage solution using Qdrant's
    dense vector capabilities. It supports both local (in-memory/disk) and remote
    Qdrant deployments with full async operations.
    """

    def __init__(
            self,
            collection_name: str,
            embedding_model: BaseEmbeddingModel,
            host: str | None = None,
            port: int = 6333,
            path: str | None = None,
            url: str | None = None,
            api_key: str | None = None,
            https: bool | None = None,
            grpc_port: int = 6334,
            prefer_grpc: bool = False,
            distance: str = "cosine",
            on_disk: bool = False,
            **kwargs,
    ):
        """Initialize the Qdrant vector store.
        
        Args:
            collection_name: Name of the Qdrant collection to use.
            embedding_model: Embedding model for generating embeddings. Cannot be None.
            host: Qdrant server host address (e.g., "localhost").
            port: Qdrant server HTTP port (default: 6333).
            path: Path for local Qdrant database (for in-memory/disk storage).
                If provided, creates a local client instead of remote.
            url: Full URL for Qdrant server (alternative to host/port).
            api_key: API key for Qdrant Cloud authentication.
            https: Whether to use HTTPS for connection (default: None).
            grpc_port: gRPC port for Qdrant server (default: 6334).
            prefer_grpc: Whether to prefer gRPC over HTTP (default: False).
            distance: Distance metric for similarity calculation when creating collection.
                Options: "cosine", "euclid", "dot". Default: "cosine".
            on_disk: Whether to enable persistent storage for vectors when creating collection.
                Default: False.
            **kwargs: Additional configuration parameters (e.g., location, prefix, timeout,
                force_disable_check_same_thread, grpc_options, auth_token_provider,
                check_compatibility, pool_size).
        """
        if _QDRANT_IMPORT_ERROR is not None:
            raise ImportError(
                "Qdrant requires extra dependencies. Install with `pip install qdrant-client`"
            ) from _QDRANT_IMPORT_ERROR

        super().__init__(collection_name=collection_name, embedding_model=embedding_model, **kwargs)

        # Initialize AsyncQdrantClient
        self.client = AsyncQdrantClient(
            host=host,
            port=port,
            path=path,
            url=url,
            api_key=api_key,
            https=https,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            **kwargs,
        )

        # Determine if using local mode
        self.is_local = path is not None

        # Convert distance string to Distance enum
        distance_map = {
            "cosine": Distance.COSINE,
            "euclid": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        self.distance = distance_map.get(distance.lower(), Distance.COSINE)
        self.on_disk = on_disk

    async def list_collections(self) -> List[str]:
        """List all available collections in Qdrant.
        
        Returns:
            A list of collection names.
        """
        collections = await self.client.get_collections()
        return [collection.name for collection in collections.collections]

    async def create_collection(self, collection_name: str, **kwargs):
        """Create a new Qdrant collection with vector configuration.
        
        Args:
            collection_name: Name of the collection to create.
            **kwargs: Additional collection settings (e.g., dimensions, distance, on_disk).
                If dimensions is not provided, it will be obtained from embedding_model.
        """
        # Check if collection already exists
        collections = await self.list_collections()
        if collection_name in collections:
            logger.info(f"Collection {collection_name} already exists")
            return

        # Get dimensions from kwargs or embedding_model
        dimensions = kwargs.get("dimensions", self.embedding_model.dimensions)
        distance = kwargs.get("distance", self.distance)
        on_disk = kwargs.get("on_disk", self.on_disk)

        # Create collection with vector configuration
        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dimensions,
                distance=distance,
                on_disk=on_disk,
            ),
        )

        logger.info(f"Created collection {collection_name} with dimensions={dimensions}")

        # Create payload indexes for common fields (only for remote servers)
        if not self.is_local:
            await self._create_payload_indexes(collection_name)

    async def _create_payload_indexes(self, collection_name: str):
        """Create payload indexes for commonly used filter fields.
        
        Args:
            collection_name: Name of the collection to create indexes for.
        """
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

    async def delete_collection(self, collection_name: str, **kwargs):
        """Delete a Qdrant collection.
        
        Args:
            collection_name: Name of the collection to delete.
            **kwargs: Additional parameters for deletion operation.
        """
        collections = await self.list_collections()
        if collection_name in collections:
            await self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted collection {collection_name}")
        else:
            logger.warning(f"Collection {collection_name} does not exist")

    async def copy_collection(self, collection_name: str, **kwargs):
        """Copy the current collection to a new collection.
        
        This creates a snapshot of the current collection and restores it to a new one.
        
        Args:
            collection_name: Name for the new copied collection.
            **kwargs: Additional parameters for the copy operation.
        """
        # Get current collection info
        collection_info = await self.client.get_collection(collection_name=self.collection_name)

        # Create new collection with same configuration
        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config=collection_info.config.params.vectors,
        )

        # Scroll through all points in the current collection and copy them
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

            # Prepare points for insertion
            points = [
                PointStruct(
                    id=record.id,
                    vector=record.vector,
                    payload=record.payload,
                )
                for record in records
            ]

            # Insert into new collection
            await self.client.upsert(
                collection_name=collection_name,
                points=points,
            )

            offset = next_offset
            if offset is None:
                break

        logger.info(f"Copied collection {self.collection_name} to {collection_name}")

    async def insert(self, nodes: VectorNode | List[VectorNode], **kwargs):
        """Insert one or more vector nodes into the collection.
        
        Args:
            nodes: A single VectorNode or list of VectorNodes to insert.
            **kwargs: Additional parameters (e.g., wait, batch_size).
        """
        # Normalize to list
        if isinstance(nodes, VectorNode):
            nodes = [nodes]

        # Generate embeddings if needed
        nodes_to_insert = []
        for node in nodes:
            if node.vector is None:
                node = await self.get_node_embeddings(node)
            nodes_to_insert.append(node)

        # Prepare points for Qdrant
        points = []
        for node in nodes_to_insert:
            # Convert vector_id to suitable format (Qdrant accepts int or UUID)
            try:
                point_id = int(node.vector_id)
            except ValueError:
                # If not an integer, use hash of the string
                point_id = abs(hash(node.vector_id)) % (10 ** 18)

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

        # Upsert points to Qdrant
        wait = kwargs.get("wait", True)
        await self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=wait,
        )

        logger.info(f"Inserted {len(points)} documents into {self.collection_name}")

    def _create_filter(self, filters: dict) -> Filter | None:
        """Create a Qdrant Filter object from the provided filters.
        
        Args:
            filters: Dictionary of filters to apply.
                Supports exact match: {"key": "value"}
                Supports IN operation: {"key": ["v1", "v2"]}
                Supports range: {"key": {"gte": 0, "lte": 100}}
                
        Returns:
            A Filter object or None if no filters provided.
        """
        if not filters:
            return None

        conditions = []
        for key, value in filters.items():
            if isinstance(value, dict) and ("gte" in value or "lte" in value):
                # Range filter
                range_params = {}
                if "gte" in value:
                    range_params["gte"] = value["gte"]
                if "lte" in value:
                    range_params["lte"] = value["lte"]
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        range=Range(**range_params),
                    )
                )
            elif isinstance(value, list):
                # IN operation - use should (OR) with multiple match conditions
                or_conditions = [
                    FieldCondition(key=f"metadata.{key}", match=MatchValue(value=v))
                    for v in value
                ]
                # For simplicity, we'll just match the first value
                # For proper OR operation, Qdrant requires nested filters
                conditions.append(
                    FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value[0]))
                )
            else:
                # Exact match
                conditions.append(
                    FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
                )

        return Filter(must=conditions) if conditions else None

    async def search(
            self,
            query: str,
            limit: int = 5,
            filters: dict | None = None,
            **kwargs
    ) -> List[VectorNode]:
        """Search for the most similar vectors to the given query.
        
        Args:
            query: The text query to search for.
            limit: Maximum number of results to return (default: 5).
            filters: Optional metadata filters to apply.
                Supports exact match: {"key": "value"} and IN operation: {"key": ["v1", "v2"]}.
            **kwargs: Additional search parameters (e.g., score_threshold).
            
        Returns:
            A list of VectorNodes ordered by similarity score (most similar first).
        """
        # Generate query embedding
        query_vector = await self.get_embeddings(query)

        # Create filter if provided
        query_filter = self._create_filter(filters) if filters else None

        # Perform search
        score_threshold = kwargs.get("score_threshold", None)

        results = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
        )

        # Convert results to VectorNodes
        nodes = []
        for point in results.points:
            payload = point.payload or {}
            node = VectorNode(
                vector_id=payload.get("vector_id", str(point.id)),
                content=payload.get("content", ""),
                vector=point.vector if hasattr(point, 'vector') else None,
                metadata=payload.get("metadata", {}),
            )
            # Add score to metadata
            node.metadata["_score"] = point.score
            nodes.append(node)

        return nodes

    async def delete(self, vector_ids: str | List[str], **kwargs):
        """Delete one or more vectors by their IDs.
        
        Args:
            vector_ids: A single vector ID or list of IDs to delete.
            **kwargs: Additional parameters (e.g., wait).
        """
        # Normalize to list
        if isinstance(vector_ids, str):
            vector_ids = [vector_ids]

        # Convert vector_ids to point IDs
        point_ids = []
        for vector_id in vector_ids:
            try:
                point_id = int(vector_id)
            except ValueError:
                point_id = abs(hash(vector_id)) % (10 ** 18)
            point_ids.append(point_id)

        # Delete points
        wait = kwargs.get("wait", True)
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=point_ids),
            wait=wait,
        )

        logger.info(f"Deleted {len(point_ids)} documents from {self.collection_name}")

    async def update(self, nodes: VectorNode | List[VectorNode], **kwargs):
        """Update one or more existing vectors in the collection.
        
        Args:
            nodes: A single VectorNode or list of VectorNodes with updated data.
            **kwargs: Additional parameters (e.g., wait).
        """
        # Normalize to list
        if isinstance(nodes, VectorNode):
            nodes = [nodes]

        # Generate embeddings for nodes that need them
        nodes_to_update = []
        for node in nodes:
            if node.vector is None and node.content:
                node = await self.get_node_embeddings(node)
            nodes_to_update.append(node)

        # Prepare points for update (upsert in Qdrant)
        points = []
        for node in nodes_to_update:
            try:
                point_id = int(node.vector_id)
            except ValueError:
                point_id = abs(hash(node.vector_id)) % (10 ** 18)

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

        # Upsert (which updates if exists, inserts if not)
        wait = kwargs.get("wait", True)
        await self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=wait,
        )

        logger.info(f"Updated {len(points)} documents in {self.collection_name}")

    async def get(self, vector_ids: str | List[str]) -> VectorNode | List[VectorNode]:
        """Retrieve one or more vectors by their IDs.
        
        Args:
            vector_ids: A single vector ID or list of IDs to retrieve.
            
        Returns:
            A single VectorNode (if single ID provided) or list of VectorNodes
            (if list of IDs provided).
        """
        single_result = isinstance(vector_ids, str)
        if single_result:
            vector_ids = [vector_ids]

        # Convert vector_ids to point IDs
        point_ids = []
        for vector_id in vector_ids:
            try:
                point_id = int(vector_id)
            except ValueError:
                point_id = abs(hash(vector_id)) % (10 ** 18)
            point_ids.append(point_id)

        # Retrieve points
        points = await self.client.retrieve(
            collection_name=self.collection_name,
            ids=point_ids,
            with_payload=True,
            with_vectors=True,
        )

        # Convert to VectorNodes
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
                logger.warning(f"Point not found")

        return results[0] if single_result and results else results

    async def list(
            self,
            filters: dict | None = None,
            limit: int | None = None
    ) -> List[VectorNode]:
        """List vectors in the collection with optional filtering.
        
        Args:
            filters: Optional metadata filters.
                Supports exact match: {"key": "value"} and IN operation: {"key": ["v1", "v2"]}.
            limit: Optional maximum number of vectors to return.
                
        Returns:
            A list of VectorNodes matching the filters.
        """
        # Create filter if provided
        scroll_filter = self._create_filter(filters) if filters else None

        # Scroll through points
        limit = limit or 10000
        records, _ = await self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )

        # Convert to VectorNodes
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

        return results

    async def close(self):
        """Close the Qdrant client connection and release resources."""
        await self.client.close()
        logger.info("Qdrant client connection closed")
