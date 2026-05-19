"""Abstract base for file store backends."""

from abc import abstractmethod

from ..base_component import BaseComponent
from ..embedding import BaseEmbeddingModel
from ..file_graph import BaseFileGraph
from ..keyword_index import BaseKeywordIndex
from ...enumeration import ComponentEnum
from ...schema import FileChunk, FileNode, FileLink


class BaseFileStore(BaseComponent):
    """Abstract base for file store backends."""

    component_type = ComponentEnum.FILE_STORE

    def __init__(
        self,
        store_name: str,
        embedding_model: str = "default",
        keyword_index: str = "default",
        file_graph: str = "default",
        store_version: str = "v1",
        **kwargs,
    ):
        super().__init__(**kwargs)
        from ..embedding import OpenAIEmbeddingModel
        from ..file_graph import LocalFileGraph
        from ..keyword_index import BM25Index

        self.store_name = store_name or self.name
        self.store_version = store_version
        if not embedding_model and not keyword_index:
            raise ValueError("At least one of embedding_model or keyword_index must be set.")

        self.embedding_model = self.bind(embedding_model, BaseEmbeddingModel, default_factory=OpenAIEmbeddingModel)
        self.keyword_index = self.bind(keyword_index, BaseKeywordIndex, default_factory=BM25Index)
        self.file_graph = self.bind(file_graph, BaseFileGraph, default_factory=LocalFileGraph)
        self.store_path = self.working_metadata_path / self.component_type.value / store_name
        self.store_path.mkdir(parents=True, exist_ok=True)

    async def _start(self) -> None:
        """Probe embedding model; disable vector capability if it fails."""
        if self.embedding_model is None:
            return
        if not await self.embedding_model.health_check():
            self.logger.warning(f"{self.store_name}: embedding unhealthy, vector disabled")
            self.embedding_model = None

    def _disable_embedding(self, reason: str) -> None:
        """Drop embedding after a runtime failure; keyword search still works."""
        if self.embedding_model is None:
            return
        self.logger.error(f"{self.store_name}: embedding disabled, {reason}")
        self.embedding_model = None

    async def upsert_file(
        self,
        file: tuple[FileNode, list[FileChunk]] | list[tuple[FileNode, list[FileChunk]]],
    ) -> None:
        """Upsert a file and its chunks into the store."""

    async def delete_by_path(self, path: str | list[str]) -> None:
        """Delete files by their paths from the store."""

    async def clear(self):
        """Clear the store of all files and chunks."""

    @abstractmethod
    async def vector_search(self, query: str, limit: int, search_filter: dict) -> list[FileChunk]:
        """Perform vector similarity search."""

    @abstractmethod
    async def keyword_search(self, query: str, limit: int, search_filter: dict) -> list[FileChunk]:
        """Perform full-text keyword search."""

    async def rebuild_links(self) -> None:
        """Rebuild all edges from each node's link payload."""
        if not self.file_graph:
            raise RuntimeError("file_graph is required for delete_by_path")
        return await self.file_graph.rebuild_links()

    async def get_nodes(self, paths: list[str]) -> list[FileNode]:
        """Return file nodes for the given paths (missing paths are skipped)."""
        if not self.file_graph:
            raise RuntimeError("file_graph is required for get_nodes")
        return await self.file_graph.get_nodes(paths)

    async def get_outlinks(self, path: str) -> list[FileLink]:
        """Return outgoing links for *path*."""
        if not self.file_graph:
            raise RuntimeError("file_graph is required for delete_by_path")
        return await self.file_graph.get_outlinks(path)

    async def get_inlinks(self, path: str) -> list[FileLink]:
        """Return incoming links for *path*."""
        if not self.file_graph:
            raise RuntimeError("file_graph is required for delete_by_path")
        return await self.file_graph.get_inlinks(path)
