"""Base vector store interface for managing vector embeddings and similarity search."""

from abc import ABC, abstractmethod
from pathlib import Path

from ..embedding import BaseEmbeddingModel
from ..schema import VectorNode


class BaseVectorStore(ABC):
    """Abstract base class defining the interface for vector storage and retrieval."""

    def __init__(
        self,
        collection_name: str,
        db_path: str | Path,
        embedding_model: BaseEmbeddingModel,
        **kwargs,
    ):
        """Initialize the vector store with a collection name and an embedding model."""
        self.collection_name: str = collection_name
        self.db_path: Path = Path(db_path)
        self.embedding_model: BaseEmbeddingModel = embedding_model
        self.kwargs: dict = kwargs

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

    async def reset_collection(self, collection_name: str):
        """Change the name of the current collection."""
        self.collection_name = collection_name
        await self.create_collection(collection_name)

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
    async def delete_all(self, **kwargs) -> None:
        """Remove all vectors from the collection."""

    @abstractmethod
    async def update(self, nodes: VectorNode | list[VectorNode], **kwargs) -> None:
        """Update the data or metadata of existing vectors in the collection."""

    @abstractmethod
    async def get(self, vector_ids: str | list[str]) -> VectorNode | list[VectorNode]:
        """Fetch specific vector nodes from the collection by their IDs."""

    async def dump(self) -> list[VectorNode]:
        """Dump the vector store to a list of vector nodes."""
        return await self.list()

    async def load(self, nodes: list[VectorNode]):
        """Load the vector store from a list of vector nodes."""
        await self.delete_all()
        if nodes:
            await self.insert(nodes)

    @abstractmethod
    async def list(
        self,
        filters: dict | None = None,
        limit: int | None = None,
        sort_key: str | None = None,
        reverse: bool = True,
    ) -> list[VectorNode]:
        """Retrieve vectors from the collection that match the given filters.

        Args:
            filters: Dictionary of filter conditions to match vectors
            limit: Maximum number of vectors to return
            sort_key: Key to sort the results by (e.g., field name in metadata). None for no sorting
            reverse: If True, sort in descending order; if False, sort in ascending order
        """

    async def start(self) -> None:
        """Initialize the vector store and ensure the collection exists.

        This method should be called after instantiation to perform async initialization.
        Subclasses should call super().start() to ensure collection creation.
        """
        await self.create_collection(self.collection_name)

    async def close(self) -> None:
        """Release resources and close active connections to the vector store."""
