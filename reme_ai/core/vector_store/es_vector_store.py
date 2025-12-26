"""Elasticsearch vector store implementation for ReMe.

This module provides an Elasticsearch-based vector store that implements the BaseVectorStore
interface. It supports dense vector storage and retrieval using Elasticsearch's kNN search
capabilities with native async operations.
"""

from typing import Any, Dict, List

from loguru import logger

# Defer import error to instantiation time
_ELASTICSEARCH_IMPORT_ERROR = None

try:
    from elasticsearch import AsyncElasticsearch
    from elasticsearch.helpers import async_bulk
except ImportError as e:
    _ELASTICSEARCH_IMPORT_ERROR = e
    AsyncElasticsearch = None
    async_bulk = None

from ..embedding import BaseEmbeddingModel
from ..schema import VectorNode
from .base_vector_store import BaseVectorStore


class ESVectorStore(BaseVectorStore):
    """Elasticsearch-based vector store implementation.
    
    This class provides a production-ready vector storage solution using Elasticsearch's
    dense_vector field type and kNN search capabilities. It supports both cloud and
    self-hosted Elasticsearch deployments with full async operations.
    
    Attributes:
        collection_name: Name of the Elasticsearch index to operate on.
        embedding_model: The embedding model used to generate vector representations.
        client: AsyncElasticsearch client instance.
        dimensions: Dimension of the embedding vectors.
    """

    def __init__(
            self,
            collection_name: str,
            embedding_model: BaseEmbeddingModel,
            host: str = "localhost",
            port: int | None = 9200,
            user: str | None = None,
            password: str | None = None,
            cloud_id: str | None = None,
            api_key: str | None = None,
            verify_certs: bool = True,
            headers: Dict[str, str] | None = None,
            **kwargs,
    ):
        """Initialize the Elasticsearch vector store.
        
        Args:
            collection_name: Name of the Elasticsearch index to use.
            embedding_model: Embedding model for generating embeddings. Cannot be None.
            host: Elasticsearch host address (default: "localhost").
            port: Elasticsearch port (default: 9200, None for cloud).
            user: Username for basic authentication.
            password: Password for basic authentication.
            cloud_id: Elastic Cloud deployment ID (alternative to host/port).
            api_key: API key for Elastic Cloud authentication.
            verify_certs: Whether to verify SSL certificates (default: True).
            headers: Additional HTTP headers to send with requests.
            **kwargs: Additional configuration parameters.
        """
        if _ELASTICSEARCH_IMPORT_ERROR is not None:
            raise ImportError(
                "Elasticsearch requires extra dependencies. Install with `pip install elasticsearch`"
            ) from _ELASTICSEARCH_IMPORT_ERROR
        
        super().__init__(collection_name=collection_name, embedding_model=embedding_model, **kwargs)

        # Initialize AsyncElasticsearch client
        if cloud_id:
            self.client = AsyncElasticsearch(
                cloud_id=cloud_id,
                api_key=api_key,
                verify_certs=verify_certs,
                headers=headers or {},
            )
        else:
            hosts = [f"{host}" if port is None else f"{host}:{port}"]
            basic_auth = (user, password) if (user and password) else None
            self.client = AsyncElasticsearch(
                hosts=hosts,
                basic_auth=basic_auth,
                verify_certs=verify_certs,
                headers=headers or {},
            )

    async def list_collections(self) -> List[str]:
        """List all available collections (indices) in Elasticsearch.
        
        Returns:
            A list of index names.
        """
        aliases = await self.client.indices.get_alias()
        return list(aliases.keys())

    async def create_collection(self, collection_name: str, **kwargs):
        """Create a new Elasticsearch index with dense vector mappings.
        
        Args:
            collection_name: Name of the index to create.
            **kwargs: Additional index settings (e.g., dimensions, number_of_shards, number_of_replicas).
                If dimensions is not provided, it will be obtained from embedding_model.
        """
        if await self.client.indices.exists(index=self.collection_name):
            return

        # Get dimensions from kwargs or embedding_model
        dimensions = kwargs.get("dimensions", self.embedding_model.dimensions)
        similarity = kwargs.get("similarity", "cosine")
        number_of_shards = kwargs.get("number_of_shards", 5)
        number_of_replicas = kwargs.get("number_of_replicas", 1)
        refresh_interval = kwargs.get("refresh_interval", "1s")

        index_settings = {
            "settings": {
                "index": {
                    "number_of_replicas": number_of_replicas,
                    "number_of_shards": number_of_shards,
                    "refresh_interval": refresh_interval,
                }
            },
            "mappings": {
                "properties": {
                    "vector_id": {"type": "keyword"},
                    "content": {"type": "text"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": dimensions,
                        "index": True,
                        "similarity": similarity,
                    },
                    "metadata": {"type": "object", "enabled": True},
                }
            },
        }

        if not await self.client.indices.exists(index=collection_name):
            await self.client.indices.create(index=collection_name, body=index_settings)
            logger.info(f"Created index {collection_name} with dimensions={dimensions}")
        else:
            logger.info(f"Index {collection_name} already exists")

    async def delete_collection(self, collection_name: str, **kwargs):
        """Delete an Elasticsearch index.
        
        Args:
            collection_name: Name of the index to delete.
            **kwargs: Additional parameters for deletion operation.
        """
        if await self.client.indices.exists(index=collection_name):
            await self.client.indices.delete(index=collection_name)
            logger.info(f"Deleted index {collection_name}")
        else:
            logger.warning(f"Index {collection_name} does not exist")

    async def copy_collection(self, collection_name: str, **kwargs):
        """Copy the current collection to a new collection.
        
        This uses Elasticsearch's reindex API to copy documents from the current
        collection to a new one.
        
        Args:
            collection_name: Name for the new copied collection.
            **kwargs: Additional parameters for the copy operation.
        """
        # Get current index settings and mappings
        current_index = await self.client.indices.get(index=self.collection_name)
        current_settings = current_index[self.collection_name]

        # Create new index with same settings
        await self.client.indices.create(
            index=collection_name,
            body={
                "settings": current_settings.get("settings", {}),
                "mappings": current_settings.get("mappings", {}),
            },
        )

        # Reindex from current collection to new collection
        await self.client.reindex(
            body={
                "source": {"index": self.collection_name},
                "dest": {"index": collection_name},
            }
        )

        logger.info(f"Copied collection {self.collection_name} to {collection_name}")

    async def insert(self, nodes: VectorNode | List[VectorNode], **kwargs):
        """Insert one or more vector nodes into the collection.
        
        Args:
            nodes: A single VectorNode or list of VectorNodes to insert.
            **kwargs: Additional parameters (e.g., refresh policy).
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

        # Prepare bulk actions
        actions = []
        for node in nodes_to_insert:
            action = {
                "_index": self.collection_name,
                "_id": node.vector_id,
                "_source": {
                    "vector_id": node.vector_id,
                    "content": node.content,
                    "vector": node.vector,
                    "metadata": node.metadata,
                },
            }
            actions.append(action)

        success, failed = await async_bulk(self.client, actions, raise_on_error=False)

        if failed:
            logger.warning(f"Failed to insert {len(failed)} documents")

        logger.info(f"Inserted {success} documents into {self.collection_name}")

        # Optionally refresh index for immediate visibility
        if kwargs.get("refresh", False):
            await self.client.indices.refresh(index=self.collection_name)

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
            **kwargs: Additional search parameters (e.g., num_candidates, score_threshold).
            
        Returns:
            A list of VectorNodes ordered by similarity score (most similar first).
        """
        # Generate query embedding
        query_vector = await self.get_embeddings(query)

        # Build kNN search query
        num_candidates = kwargs.get("num_candidates", limit * 2)

        search_query = {
            "knn": {
                "field": "vector",
                "query_vector": query_vector,
                "k": limit,
                "num_candidates": num_candidates,
            },
            "size": limit,
        }

        # Add metadata filters if provided
        if filters:
            filter_conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    # Use terms query for list values (IN operation)
                    filter_conditions.append({"terms": {f"metadata.{key}": value}})
                else:
                    # Use term query for single values
                    filter_conditions.append({"term": {f"metadata.{key}": value}})
            search_query["knn"]["filter"] = {"bool": {"must": filter_conditions}}

        response = await self.client.search(index=self.collection_name, body=search_query)

        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            node = VectorNode(
                vector_id=source.get("vector_id", hit["_id"]),
                content=source.get("content", ""),
                vector=source.get("vector"),
                metadata=source.get("metadata", {}),
            )
            # Add score to metadata
            node.metadata["_score"] = hit["_score"]
            results.append(node)

        return results

    async def delete(self, vector_ids: str | List[str], **kwargs):
        """Delete one or more vectors by their IDs.
        
        Args:
            vector_ids: A single vector ID or list of IDs to delete.
            **kwargs: Additional parameters (e.g., refresh policy).
        """
        # Normalize to list
        if isinstance(vector_ids, str):
            vector_ids = [vector_ids]

        # Use bulk delete for efficiency
        actions = []
        for vector_id in vector_ids:
            actions.append({
                "_op_type": "delete",
                "_index": self.collection_name,
                "_id": vector_id,
            })

        success, failed = await async_bulk(
            self.client,
            actions,
            raise_on_error=False,
            raise_on_exception=False,
        )

        if failed:
            logger.warning(f"Failed to delete {len(failed)} documents")

        logger.info(f"Deleted {success} documents from {self.collection_name}")

        # Optionally refresh index for immediate visibility
        if kwargs.get("refresh", False):
            await self.client.indices.refresh(index=self.collection_name)

    async def update(self, nodes: VectorNode | List[VectorNode], **kwargs):
        """Update one or more existing vectors in the collection.
        
        Args:
            nodes: A single VectorNode or list of VectorNodes with updated data.
            **kwargs: Additional parameters (e.g., refresh policy).
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

        # Use bulk update for efficiency
        actions = []
        for node in nodes_to_update:
            doc = {
                "vector_id": node.vector_id,
                "content": node.content,
                "metadata": node.metadata,
            }
            if node.vector is not None:
                doc["vector"] = node.vector

            actions.append({
                "_op_type": "update",
                "_index": self.collection_name,
                "_id": node.vector_id,
                "doc": doc,
            })

        success, failed = await async_bulk(
            self.client,
            actions,
            raise_on_error=False,
            raise_on_exception=False,
        )

        if failed:
            logger.warning(f"Failed to update {len(failed)} documents")

        logger.info(f"Updated {success} documents in {self.collection_name}")

        # Optionally refresh index for immediate visibility
        if kwargs.get("refresh", False):
            await self.client.indices.refresh(index=self.collection_name)

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

        # Use mget for efficient batch retrieval
        response = await self.client.mget(
            index=self.collection_name,
            body={"ids": vector_ids}
        )

        results = []
        for doc in response["docs"]:
            if doc.get("found"):
                source = doc["_source"]
                node = VectorNode(
                    vector_id=source.get("vector_id", doc["_id"]),
                    content=source.get("content", ""),
                    vector=source.get("vector"),
                    metadata=source.get("metadata", {}),
                )
                results.append(node)
            else:
                logger.warning(f"Document with ID {doc['_id']} not found")

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
        # Build query
        query: Dict[str, Any] = {"query": {"match_all": {}}}

        if filters:
            filter_conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    # Use terms query for list values (IN operation)
                    filter_conditions.append({"terms": {f"metadata.{key}": value}})
                else:
                    # Use term query for single values
                    filter_conditions.append({"term": {f"metadata.{key}": value}})
            query["query"] = {"bool": {"must": filter_conditions}}

        if limit:
            query["size"] = limit
        else:
            query["size"] = 10000  # Default max size

        response = await self.client.search(index=self.collection_name, body=query)

        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            node = VectorNode(
                vector_id=source.get("vector_id", hit["_id"]),
                content=source.get("content", ""),
                vector=source.get("vector"),
                metadata=source.get("metadata", {}),
            )
            results.append(node)

        return results

    async def close(self):
        """Close the Elasticsearch client connection and release resources."""
        await self.client.close()
        logger.info("Elasticsearch client connection closed")