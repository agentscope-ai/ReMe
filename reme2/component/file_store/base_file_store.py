"""Abstract base class for file stores — persistence + search for nodes & chunks."""

import re
from abc import abstractmethod
from pathlib import Path

from ..base_component import BaseComponent
from ..embedding import BaseEmbeddingModel
from ...enumeration import ComponentEnum
from ...schema import FileChunk, FileNode


class BaseFileStore(BaseComponent):
    """Abstract file-store engine: persistence + search for nodes & chunks."""

    component_type = ComponentEnum.FILE_STORE

    def __init__(
            self,
            store_name: str,
            embedding_model: str = "default",
            fts_enabled: bool = True,
            **kwargs,
    ):
        super().__init__(**kwargs)
        if not re.match(r"^[a-zA-Z0-9_]+$", store_name):
            raise ValueError(f"Invalid store name '{store_name}'. Only alphanumeric and underscores allowed.")
        self.store_name = store_name or self.name
        self._embedding_model_name = embedding_model
        self.fts_enabled = fts_enabled

        self.embedding_model: BaseEmbeddingModel | None = None
        self.vector_enabled = bool(embedding_model)
        self.working_dir = self.app_context.app_config.working_dir if self.app_context else ""
        self.store_path = Path(self.working_dir) / "file_store" / store_name
        self.store_path.mkdir(parents=True, exist_ok=True)
        if not self.vector_enabled and not self.fts_enabled:
            raise ValueError("At least one of embedding_model or fts_enabled must be set.")

    # Lifecycle

    async def _start(self) -> None:
        if self.vector_enabled and self.app_context is not None:
            model_dict = self.app_context.components.get(ComponentEnum.EMBEDDING_MODEL, {})
            if self._embedding_model_name not in model_dict:
                raise ValueError(f"Embedding model '{self._embedding_model_name}' not found.")
            model = model_dict[self._embedding_model_name]
            if not isinstance(model, BaseEmbeddingModel):
                raise TypeError(f"Expected BaseEmbeddingModel, got {type(model).__name__}")
            self.embedding_model = model

    async def _close(self) -> None:
        self.embedding_model = None

    # Composite operations

    async def upsert(self, node: FileNode, chunks: list[FileChunk]) -> None:
        await self.upsert_node(node)
        await self.upsert_chunks(node.path, chunks)

    async def delete(self, path: str) -> None:
        await self.delete_chunks(path)
        await self.delete_node(path)

    # Abstract: node operations

    @abstractmethod
    async def upsert_node(self, node: FileNode) -> None:
        """Persist a node, replacing any prior entry for `node.path`."""

    @abstractmethod
    async def get_node(self, path: str) -> FileNode | None:
        """Fetch a node by path, or None if absent."""

    @abstractmethod
    async def delete_node(self, path: str) -> None:
        """Delete the node entry for `path`."""

    # Abstract: chunk operations

    @abstractmethod
    async def upsert_chunks(self, path: str, chunks: list[FileChunk]) -> None:
        """Insert or replace all chunks for `path`. Handles embedding internally."""

    @abstractmethod
    async def get_chunks(self, path: str) -> list[FileChunk]:
        """All chunks for `path`."""

    @abstractmethod
    async def delete_chunks(self, path: str) -> None:
        """Delete all chunks for `path`."""

    # Abstract: search

    @abstractmethod
    async def vector_search(self, query: str, limit: int, search_filter: dict) -> list[FileChunk]:
        """Vector similarity search."""

    @abstractmethod
    async def keyword_search(self, query: str, limit: int, search_filter: dict) -> list[FileChunk]:
        """Full-text / keyword search."""

    # Internal helpers

    async def get_embeddings(self, texts: list[str]) -> list[list[float] | None] | None:
        """Embed texts. Returns None if vector search disabled or on API error."""
        if not self.vector_enabled or not texts or not self.embedding_model:
            return None
        try:
            return await self.embedding_model.get_embeddings(texts)
        except Exception as e:
            self.logger.warning(f"[{self.store_name}] Disabling vector search: {e}")
            self.vector_enabled = False
            return None
