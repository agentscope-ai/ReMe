"""Elasticsearch vector store implementation for ReMe.

This module provides an Elasticsearch-based vector store that implements the BaseVectorStore
interface for high-performance dense vector storage and retrieval.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from loguru import logger

from .base_vector_store import BaseVectorStore
from ..embedding import BaseEmbeddingModel
from ..schema import VectorNode

_ELASTICSEARCH_IMPORT_ERROR = None

try:
    from elasticsearch import AsyncElasticsearch
    from elasticsearch.helpers import async_bulk
except ImportError as e:
    _ELASTICSEARCH_IMPORT_ERROR = e
    AsyncElasticsearch = None
    async_bulk = None


class ESVectorStore(BaseVectorStore):
    """Elasticsearch-based vector store for dense vector storage and kNN search."""

    def __init__(
        self,
        collection_name: str,
        embedding_model: BaseEmbeddingModel,
        thread_pool: ThreadPoolExecutor,
        hosts: str | list[str] | None = None,
        basic_auth: tuple[str, str] | None = None,
        cloud_id: str | None = None,
        api_key: str | None = None,
        verify_certs: bool = True,
        headers: dict[str, str] | None = None,
        **kwargs,
    ):
        """Initialize the Elasticsearch client and vector store configuration.

        Args:
            collection_name: Name of the Elasticsearch index (converted to lowercase).
            embedding_model: Model instance used to generate vector embeddings.
            thread_pool: ThreadPoolExecutor for running synchronous operations.
            hosts: Connection host(s) for the Elasticsearch cluster.
            basic_auth: Credentials for basic authentication.
            cloud_id: Deployment ID for Elastic Cloud.
            api_key: API key for authentication.
            verify_certs: Enable or disable SSL certificate verification.
            headers: Custom HTTP headers for requests.
            **kwargs: Additional configuration passed to the base class.
        """
        if _ELASTICSEARCH_IMPORT_ERROR is not None:
            raise ImportError(
                "Elasticsearch requires extra dependencies. Install with `pip install elasticsearch`",
            ) from _ELASTICSEARCH_IMPORT_ERROR

        # Elasticsearch requires lowercase index names
        collection_name = collection_name.lower()

        super().__init__(
            collection_name=collection_name,
            embedding_model=embedding_model,
            thread_pool=thread_pool,
            **kwargs,
        )

        # Initialize AsyncElasticsearch client
        self.client = AsyncElasticsearch(
            hosts=hosts,
            cloud_id=cloud_id,
            api_key=api_key,
            basic_auth=basic_auth,
            verify_certs=verify_certs,
            headers=headers or {},
        )

    async def list_collections(self) -> list[str]:
        """List all available index names in the Elasticsearch cluster."""
        aliases = await self.client.indices.get_alias()
        return list(aliases.keys())

    async def create_collection(self, collection_name: str, **kwargs):
        """Create a new index with dense vector mappings for kNN search.

        Args:
            collection_name: Name of the index to create.
            **kwargs: Settings like dimensions, similarity, shards, and replicas.
        """
        collection_name = collection_name.lower()

        if await self.client.indices.exists(index=collection_name):
            return

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
                },
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
                },
            },
        }

        if not await self.client.indices.exists(index=collection_name):
            await self.client.indices.create(index=collection_name, body=index_settings)
            logger.info(f"Created index {collection_name} with dimensions={dimensions}")
        else:
            logger.info(f"Index {collection_name} already exists")

    async def delete_collection(self, collection_name: str, **kwargs):
        """Permanently delete an Elasticsearch index.

        Args:
            collection_name: Name of the index to delete.
            **kwargs: Additional parameters for the deletion request.
        """
        collection_name = collection_name.lower()

        if await self.client.indices.exists(index=collection_name):
            await self.client.indices.delete(index=collection_name)
            logger.info(f"Deleted index {collection_name}")
        else:
            logger.warning(f"Index {collection_name} does not exist")

    async def copy_collection(self, collection_name: str, **kwargs):
        """Reindex the current collection into a new index with identical mappings.

        Args:
            collection_name: Name of the destination index.
            **kwargs: Additional parameters for the reindexing process.
        """
        collection_name = collection_name.lower()

        current_index = await self.client.indices.get(index=self.collection_name)
        current_settings = current_index[self.collection_name]

        settings_to_copy = current_settings.get("settings", {}).copy()
        if "index" in settings_to_copy:
            index_settings = settings_to_copy["index"].copy()
            internal_keys = [
                "uuid",
                "creation_date",
                "provided_name",
                "version",
                "store",
                "routing",
                "replication",
            ]
            for key in internal_keys:
                index_settings.pop(key, None)
            settings_to_copy["index"] = index_settings

        await self.client.indices.create(
            index=collection_name,
            body={
                "settings": settings_to_copy,
                "mappings": current_settings.get("mappings", {}),
            },
        )

        await self.client.reindex(
            body={
                "source": {"index": self.collection_name},
                "dest": {"index": collection_name},
            },
        )

        logger.info(f"Copied collection {self.collection_name} to {collection_name}")

    async def insert(self, nodes: VectorNode | list[VectorNode], refresh: bool = True, **kwargs):
        """Insert nodes into the index, generating embeddings if missing.

        Args:
            nodes: Single or multiple VectorNode objects to index.
            refresh: If True, makes the operation visible to search immediately.
            **kwargs: Additional insertion options.
        """
        if isinstance(nodes, VectorNode):
            nodes = [nodes]

        nodes_without_vectors = [node for node in nodes if node.vector is None]
        if nodes_without_vectors:
            nodes_with_vectors = await self.get_node_embeddings(nodes_without_vectors)
            vector_map = {n.vector_id: n for n in nodes_with_vectors}
            nodes_to_insert = [vector_map.get(n.vector_id, n) if n.vector is None else n for n in nodes]
        else:
            nodes_to_insert = nodes

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

        if refresh:
            await self.client.indices.refresh(index=self.collection_name)

    async def search(
        self,
        query: str,
        limit: int = 5,
        filters: dict | None = None,
        **kwargs,
    ) -> list[VectorNode]:
        """Perform a kNN similarity search based on a text query.

        Args:
            query: The text to search for.
            limit: Maximum number of nearest neighbors to return.
            filters: Metadata filters for exact match or 'IN' operations.
            **kwargs: Search parameters like num_candidates or score_threshold.

        Returns:
            List of VectorNode objects ordered by similarity.
        """
        query_vector = await self.get_embedding(query)
        num_candidates = kwargs.get("num_candidates", limit * 2)

        search_query: dict = {
            "knn": {
                "field": "vector",
                "query_vector": query_vector,
                "k": limit,
                "num_candidates": num_candidates,
            },
            "size": limit,
        }

        if filters:
            filter_conditions = []
            for key, value in filters.items():
                # New syntax: [start, end] represents a range query
                if isinstance(value, list) and len(value) == 2:
                    # Range query: field >= value[0] AND field <= value[1]
                    filter_conditions.append(
                        {
                            "range": {
                                f"metadata.{key}": {
                                    "gte": value[0],
                                    "lte": value[1],
                                },
                            },
                        },
                    )
                else:
                    # Exact match
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
            node.metadata["score"] = hit["_score"]
            results.append(node)

        return results

    async def delete(self, vector_ids: str | list[str], refresh: bool = True, **kwargs):
        """Delete specific vectors from the index by their IDs.

        Args:
            vector_ids: Single ID or list of IDs to remove.
            refresh: If True, refreshes the index after deletion.
            **kwargs: Additional deletion parameters.
        """
        if isinstance(vector_ids, str):
            vector_ids = [vector_ids]

        actions = []
        for vector_id in vector_ids:
            actions.append(
                {
                    "_op_type": "delete",
                    "_index": self.collection_name,
                    "_id": vector_id,
                },
            )

        success, failed = await async_bulk(
            self.client,
            actions,
            raise_on_error=False,
            raise_on_exception=False,
        )

        if failed:
            logger.warning(f"Failed to delete {len(failed)} documents")

        logger.info(f"Deleted {success} documents from {self.collection_name}")

        if refresh:
            await self.client.indices.refresh(index=self.collection_name)

    async def delete_all(self, **kwargs):
        """Remove all vectors from the collection.

        Args:
            **kwargs: Additional deletion parameters.
        """
        response = await self.client.delete_by_query(
            index=self.collection_name,
            body={"query": {"match_all": {}}},
        )

        deleted_count = response.get("deleted", 0)
        logger.info(f"Deleted all {deleted_count} documents from {self.collection_name}")

        refresh = kwargs.get("refresh", True)
        if refresh:
            await self.client.indices.refresh(index=self.collection_name)

    async def update(self, nodes: VectorNode | list[VectorNode], refresh: bool = True, **kwargs):
        """Update existing documents with new content or metadata.

        Args:
            nodes: Single or multiple VectorNode objects with updated data.
            refresh: If True, refreshes the index after update.
            **kwargs: Additional update parameters.
        """
        if isinstance(nodes, VectorNode):
            nodes = [nodes]

        nodes_without_vectors = [node for node in nodes if node.vector is None and node.content]
        if nodes_without_vectors:
            nodes_with_vectors = await self.get_node_embeddings(nodes_without_vectors)
            vector_map = {n.vector_id: n for n in nodes_with_vectors}
            nodes_to_update = [vector_map.get(n.vector_id, n) if n.vector is None and n.content else n for n in nodes]
        else:
            nodes_to_update = nodes

        actions = []
        for node in nodes_to_update:
            doc: dict = {
                "vector_id": node.vector_id,
                "content": node.content,
                "metadata": node.metadata,
            }
            if node.vector is not None:
                doc["vector"] = node.vector

            actions.append(
                {
                    "_op_type": "update",
                    "_index": self.collection_name,
                    "_id": node.vector_id,
                    "doc": doc,
                },
            )

        success, failed = await async_bulk(
            self.client,
            actions,
            raise_on_error=False,
            raise_on_exception=False,
        )

        if failed:
            logger.warning(f"Failed to update {len(failed)} documents")

        logger.info(f"Updated {success} documents in {self.collection_name}")

        if refresh:
            await self.client.indices.refresh(index=self.collection_name)

    async def get(self, vector_ids: str | list[str]) -> VectorNode | list[VectorNode]:
        """Fetch documents by their IDs from the current index.

        Args:
            vector_ids: Single ID or list of IDs to retrieve.

        Returns:
            A single VectorNode or a list of VectorNodes.
        """
        single_result = isinstance(vector_ids, str)
        if single_result:
            vector_ids = [vector_ids]

        response = await self.client.mget(
            index=self.collection_name,
            body={"ids": vector_ids},
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
        limit: int | None = None,
        sort_key: str | None = None,
        reverse: bool = False,
    ) -> list[VectorNode]:
        """Retrieve a list of nodes filtered by metadata or limit.

        Args:
            filters: Optional metadata filtering criteria.
            limit: Maximum number of nodes to return.
            sort_key: Key to sort the results by (e.g., field name in metadata). None for no sorting
            reverse: If True, sort in descending order; if False, sort in ascending order

        Returns:
            A list of matching VectorNode objects.
        """
        query: dict[str, Any] = {"query": {"match_all": {}}}

        if filters:
            filter_conditions = []
            for key, value in filters.items():
                # New syntax: [start, end] represents a range query
                if isinstance(value, list) and len(value) == 2:
                    # Range query: field >= value[0] AND field <= value[1]
                    filter_conditions.append(
                        {
                            "range": {
                                f"metadata.{key}": {
                                    "gte": value[0],
                                    "lte": value[1],
                                },
                            },
                        },
                    )
                else:
                    # Exact match
                    filter_conditions.append({"term": {f"metadata.{key}": value}})
            query["query"] = {"bool": {"must": filter_conditions}}

        # Add sorting to the Elasticsearch query if sort_key is provided
        if sort_key:
            query["sort"] = [
                {
                    f"metadata.{sort_key}": {
                        "order": "desc" if reverse else "asc",
                    },
                },
            ]

        if limit:
            query["size"] = limit
        else:
            query["size"] = 10000

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

    def set_collection_name(self, collection_name: str):
        """Set the collection name and ensure it's lowercase for Elasticsearch compatibility."""
        collection_name = collection_name.lower()
        super().set_collection_name(collection_name)
        logger.info(f"Collection name set to {collection_name} (converted to lowercase)")

    async def close(self):
        """Terminate the Elasticsearch client session and release resources."""
        await self.client.close()
        logger.info("Elasticsearch client connection closed")
