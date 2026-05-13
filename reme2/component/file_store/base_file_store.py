"""Abstract base class for file stores."""

import re
from abc import abstractmethod
from pathlib import Path

from .bm25_lite import BM25Lite
from ..base_component import BaseComponent
from ..embedding import BaseEmbeddingModel
from ...enumeration import ComponentEnum
from ...schema import FileChunk, FileNode


class BaseFileStore(BaseComponent):
    """Abstract file-store engine."""

    component_type = ComponentEnum.FILE_STORE

    def __init__(
            self,
            store_name: str,
            embedding_model: str = "default",
            tokenizer: str = "default",
            fts_enabled: bool = True,
            **kwargs,
    ):
        """Initialize the file store."""
        super().__init__(**kwargs)
        if not re.match(r"^[a-zA-Z0-9_]+$", store_name):
            raise ValueError(f"Invalid store name '{store_name}'. Only alphanumeric and underscores allowed.")
        self.store_name = store_name or self.name
        self._embedding_model_name = embedding_model
        self.tokenizer = tokenizer
        self.fts_enabled = fts_enabled

        self.embedding_model: BaseEmbeddingModel | None = None
        self.vector_enabled = bool(embedding_model)
        self.working_dir = self.app_context.app_config.working_dir if self.app_context else ""
        self.store_path = Path(self.working_dir) / "file_store" / store_name
        self.store_path.mkdir(parents=True, exist_ok=True)
        if not self.vector_enabled and not self.fts_enabled:
            raise ValueError("At least one of embedding_model or fts_enabled must be set.")

        self.bm25: BM25Lite | None = None

    async def _start(self) -> None:
        """Start the file store."""
        if self.vector_enabled and self.app_context is not None:
            model_dict = self.app_context.components.get(ComponentEnum.EMBEDDING_MODEL, {})
            if self._embedding_model_name not in model_dict:
                raise ValueError(f"Embedding model '{self._embedding_model_name}' not found.")
            model = model_dict[self._embedding_model_name]
            if not isinstance(model, BaseEmbeddingModel):
                raise TypeError(f"Expected BaseEmbeddingModel, got {type(model).__name__}")
            self.embedding_model = model

        if self.fts_enabled:
            self.bm25 = BM25Lite(index_dir=self.store_path, tokenizer=self.tokenizer, app_context=self.app_context)
            if self.bm25 is not None:
                await self.bm25.start()

    async def _close(self) -> None:
        """Close the file store and release resources."""
        self.embedding_model = None
        if self.fts_enabled and self.bm25 is not None:
            await self.bm25.close()

    # Composite operations

    async def upsert_file(self, node: FileNode, chunks: list[FileChunk]) -> None:
        """Upsert a node and its associated chunks."""

    async def delete(self, path: str) -> None:
        """Delete a node and all its associated chunks by path."""

    async def reindex(self) -> None:
        """Re-index all nodes and chunks in the store."""

    @abstractmethod
    async def vector_search(self, query: str, limit: int, search_filter: dict) -> list[FileChunk]:
        """Perform vector similarity search."""

    @abstractmethod
    async def keyword_search(self, query: str, limit: int, search_filter: dict) -> list[FileChunk]:
        """Perform full-text keyword search."""
