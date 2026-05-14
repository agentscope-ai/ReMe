from abc import abstractmethod

from ..base_component import BaseComponent
from ..embedding import BaseEmbeddingModel
from ..keyword_index import BaseKeywordIndex
from ...enumeration import ComponentEnum
from ...schema import FileChunk, FileNode


class BaseFileStore(BaseComponent):
    component_type = ComponentEnum.FILE_STORE

    def __init__(
            self,
            store_name: str,
            embedding_model: str = "default",
            keyword_index: str = "default",
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.store_name = store_name or self.name
        if not embedding_model and not keyword_index:
            raise ValueError("At least one of embedding_model or keyword_index must be set.")

        self.embedding_model = self.bind(embedding_model, BaseEmbeddingModel)
        self.keyword_index = self.bind(keyword_index, BaseKeywordIndex)
        self.store_path = self.working_path / self.component_type.value / store_name
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.file_nodes: dict[str, FileNode] = {}

    async def _start(self) -> None:
        await self.load_file_nodes()

    async def _close(self) -> None:
        await self.dump_file_nodes()

    async def load_file_nodes(self):
        ...

    async def dump_file_nodes(self):
        ...

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
