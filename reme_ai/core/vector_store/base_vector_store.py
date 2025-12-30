import asyncio
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, List

from reme_ai.core.context import C
from reme_ai.core.embedding import BaseEmbeddingModel
from reme_ai.core.schema import VectorNode


class BaseVectorStore(ABC):
    """Abstract base class for vector store implementations.

    This class defines the interface for vector storage systems that can store,
    retrieve, and search vector embeddings along with their associated metadata.
    All concrete vector store implementations should inherit from this class and
    implement the abstract methods.

    Attributes:
        collection_name: Name of the collection/index to operate on.
        embedding_model: The embedding model used to generate vector representations.
        kwargs: Additional keyword arguments for subclass-specific configuration.
    """

    def __init__(
        self,
        collection_name: str,
        embedding_model: BaseEmbeddingModel,
        **kwargs,
    ):
        """Initialize the vector store.

        Args:
            collection_name: Name of the collection to use for storing vectors.
            embedding_model: Embedding model for generating embeddings. Cannot be None.
            **kwargs: Additional configuration parameters specific to the vector store implementation.
        """
        assert embedding_model is not None, "embedding_model is required and cannot be None"
        self.collection_name: str = collection_name
        self.embedding_model: BaseEmbeddingModel = embedding_model
        self.kwargs: dict = kwargs

    @staticmethod
    async def _run_sync_in_executor(sync_func: Callable, *args, **kwargs) -> Any:
        """Execute a synchronous function in a thread pool executor.

        This utility method allows running blocking synchronous functions within
        async context without blocking the event loop.

        Args:
            sync_func: The synchronous function to execute.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result returned by the synchronous function.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(C.thread_pool, partial(sync_func, *args, **kwargs))

    async def get_node_embeddings(self, nodes: VectorNode | List[VectorNode]) -> VectorNode | List[VectorNode]:
        """Generate embeddings for the given node(s) using the embedding model.

        Args:
            nodes: A single VectorNode or a list of VectorNodes to generate embeddings for.

        Returns:
            The input node(s) with their embedding field populated.
        """
        return await self.embedding_model.get_node_embeddings(nodes)

    async def get_embeddings(self, query: str | List[str]) -> List[List[float]] | List[float]:
        """Generate embeddings for the given text query/queries.

        Args:
            query: A single text string or a list of text strings to embed.

        Returns:
            A list of floats (for single query) or a list of lists of floats (for multiple queries).
        """
        return await self.embedding_model.get_embeddings(query)

    @abstractmethod
    async def list_collections(self) -> List[str]:
        """List all available collections in the vector store.

        Returns:
            A list of collection names.
        """

    @abstractmethod
    async def create_collection(self, collection_name: str, **kwargs):
        """Create a new collection in the vector store.

        Args:
            collection_name: Name of the collection to create.
            **kwargs: Additional collection-specific configuration parameters
                (e.g., dimension, distance metric, index type).
        """

    @abstractmethod
    async def delete_collection(self, collection_name: str, **kwargs):
        """Delete a collection from the vector store.

        Args:
            collection_name: Name of the collection to delete.
            **kwargs: Additional parameters for deletion operation.
        """

    @abstractmethod
    async def copy_collection(self, collection_name: str, **kwargs):
        """Copy the current collection to a new collection.

        Creates a duplicate of self.collection_name with the name specified
        by collection_name parameter.

        Args:
            collection_name: Name for the new copied collection.
            **kwargs: Additional parameters for the copy operation.
        """

    @abstractmethod
    async def insert(self, nodes: VectorNode | List[VectorNode], **kwargs):
        """Insert one or more vector nodes into the collection.

        If the nodes don't have embeddings, they will be generated using
        the embedding model (if available).

        Args:
            nodes: A single VectorNode or list of VectorNodes to insert.
            **kwargs: Additional parameters for the insertion operation
                (e.g., batch_size, upsert mode).
        """

    @abstractmethod
    async def search(self, query: str, limit: int = 5, filters: dict | None = None, **kwargs) -> List[VectorNode]:
        """Search for the most similar vectors to the given query.

        Args:
            query: The text query to search for. Will be converted to embedding.
            limit: Maximum number of results to return. Default is 5.
            filters: Optional metadata filters to apply to the search.
                Format depends on the specific vector store implementation.
            **kwargs: Additional search parameters (e.g., score_threshold, ef_search).

        Returns:
            A list of VectorNodes ordered by similarity (most similar first).
        """

    @abstractmethod
    async def delete(self, vector_ids: str | List[str], **kwargs):
        """Delete one or more vectors by their IDs.

        Args:
            vector_ids: A single vector ID or list of IDs to delete.
            **kwargs: Additional parameters for the deletion operation.
        """

    @abstractmethod
    async def update(self, nodes: VectorNode | List[VectorNode], **kwargs):
        """Update one or more existing vectors in the collection.

        Updates the content, metadata, and/or embedding of existing nodes.

        Args:
            nodes: A single VectorNode or list of VectorNodes with updated data.
                The nodes must have valid IDs that exist in the collection.
            **kwargs: Additional parameters for the update operation.
        """

    @abstractmethod
    async def get(self, vector_ids: str | List[str]):
        """Retrieve one or more vectors by their IDs.

        Args:
            vector_ids: A single vector ID or list of IDs to retrieve.

        Returns:
            A single VectorNode (if single ID provided) or list of VectorNodes
            (if list of IDs provided).
        """

    @abstractmethod
    async def list(self, filters: dict | None = None, limit: int | None = None):
        """List vectors in the collection with optional filtering.

        Args:
            filters: Optional metadata filters to apply. Format depends on
                the specific vector store implementation.
            limit: Optional maximum number of vectors to return.
                If None, returns all matching vectors.

        Returns:
            A list of VectorNodes matching the filters.
        """

    async def close(self):
        """Close the vector store connection and release resources.

        This method should be called when the vector store is no longer needed
        to properly clean up connections and resources. Default implementation
        does nothing; subclasses should override if cleanup is needed.
        """
        pass
