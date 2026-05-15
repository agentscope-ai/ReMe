from abc import abstractmethod

from ..base_component import BaseComponent
from ..embedding import BaseEmbeddingModel
from ..file_graph import BaseFileGraph
from ..keyword_index import BaseKeywordIndex
from ...enumeration import ComponentEnum
from ...schema import FileChunk, FileNode, FileLink


class BaseFileStore(BaseComponent):
    component_type = ComponentEnum.FILE_STORE

    def __init__(
        self,
        store_name: str,
        embedding_model: str = "default",
        keyword_index: str = "default",
        file_graph: str = "default",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.store_name = store_name or self.name
        if not embedding_model and not keyword_index:
            raise ValueError("At least one of embedding_model or keyword_index must be set.")

        self.embedding_model = self.bind(embedding_model, BaseEmbeddingModel)
        self.keyword_index = self.bind(keyword_index, BaseKeywordIndex)
        self.file_graph = self.bind(file_graph, BaseFileGraph)
        self.store_path = self.working_path / self.component_type.value / store_name
        self.store_path.mkdir(parents=True, exist_ok=True)

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
        if not self.file_graph:
            raise RuntimeError("file_graph is required for delete_by_path")
        return await self.file_graph.rebuild_links()

    async def get_outlinks(self, path: str) -> list[FileLink]:
        if not self.file_graph:
            raise RuntimeError("file_graph is required for delete_by_path")
        return await self.file_graph.get_outlinks(path)

    async def get_inlinks(self, path: str) -> list[FileLink]:
        if not self.file_graph:
            raise RuntimeError("file_graph is required for delete_by_path")
        return await self.file_graph.get_inlinks(path)
