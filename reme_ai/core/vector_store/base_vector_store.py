"""Base vector store interface for managing vector embeddings and similarity search."""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial

from reme_ai.core.context import C
from reme_ai.core.embedding import BaseEmbeddingModel
from reme_ai.core.schema import VectorNode


class BaseVectorStore(ABC):
    """Abstract base class defining the interface for vector storage and retrieval."""

    def __init__(
        self,
        collection_name: str,
        embedding_model: BaseEmbeddingModel,
        **kwargs,
    ):
        """Initialize the vector store with a collection name and an embedding model."""
        if embedding_model is None:
            raise ValueError("embedding_model is required")
        self.collection_name: str = collection_name
        self.embedding_model: BaseEmbeddingModel = embedding_model
        self.kwargs: dict = kwargs

    @staticmethod
    async def _run_sync_in_executor(sync_func: Callable, *args, **kwargs):
        """Run a synchronous function in the context-defined thread pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(C.thread_pool, partial(sync_func, *args, **kwargs))

    async def get_node_embedding(self, node: VectorNode) -> VectorNode:
        """Generate and assign embedding for a single vector node."""
        return await self.embedding_model.get_node_embedding(node)

    async def get_node_embeddings(self, nodes: list[VectorNode]) -> list[VectorNode]:
        """Generate and assign embeddings for multiple vector nodes."""
        return await self.embedding_model.get_node_embeddings(nodes)

    async def get_embedding(self, query: str) -> list[float]:
        """Convert a single text query into vector embedding using the configured model."""
        return await self.embedding_model.get_embedding(query)

    async def get_embeddings(self, queries: list[str]) -> list[list[float]]:
        """Convert multiple text queries into vector embeddings using the configured model."""
        return await self.embedding_model.get_embeddings(queries)

    @abstractmethod
    async def list_collections(self) -> list[str]:
        """Retrieve a list of all existing collection names in the store."""

    @abstractmethod
    async def create_collection(self, collection_name: str, **kwargs) -> None:
        """Create a new vector collection with the specified name and configuration."""

    @abstractmethod
    async def delete_collection(self, collection_name: str, **kwargs) -> None:
        """Permanently remove a collection from the vector store."""

    @abstractmethod
    async def copy_collection(self, collection_name: str, **kwargs) -> None:
        """Duplicate the current collection to a new one with the given name."""

    @abstractmethod
    async def insert(self, nodes: VectorNode | list[VectorNode], **kwargs) -> None:
        """Add one or more vector nodes into the current collection."""

    @abstractmethod
    async def search(self, query: str, limit: int = 5, filters: dict | None = None, **kwargs) -> list[VectorNode]:
        """Find the most similar vector nodes based on a text query."""

    @abstractmethod
    async def delete(self, vector_ids: str | list[str], **kwargs) -> None:
        """Remove specific vectors from the collection using their identifiers."""

    @abstractmethod
    async def update(self, nodes: VectorNode | list[VectorNode], **kwargs) -> None:
        """Update the data or metadata of existing vectors in the collection."""

    @abstractmethod
    async def get(self, vector_ids: str | list[str]) -> VectorNode | list[VectorNode]:
        """Fetch specific vector nodes from the collection by their IDs."""

    @abstractmethod
    async def list(self, filters: dict | None = None, limit: int | None = None) -> list[VectorNode]:
        """Retrieve vectors from the collection that match the given filters."""

    async def close(self) -> None:
        """Release resources and close active connections to the vector store."""
