"""Abstract base class for file stores — minimal engine surface.

The store owns persistence + search for the (file → chunks) graph.
Subclasses implement every read/write/search verb; the base only
resolves shared infrastructure:

  * `working_dir` pulled from `app_config` for path-relative work.
  * `embedding_model` resolved on `_start`.
  * `embed(texts)` — single async entry that wraps the model and
    auto-disables vector search on persistent failure.
"""

from __future__ import annotations

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
        store_path: str | Path,
        embedding_model: str = "default",
        fts_enabled: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not re.match(r"^[a-zA-Z0-9_]+$", store_name):
            raise ValueError(
                f"Invalid store name '{store_name}'. "
                f"Only alphanumeric characters and underscores are allowed.",
            )
        self.store_name: str = store_name
        self.store_path: Path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.working_dir: str = (
            self.app_context.app_config.working_dir
            if self.app_context is not None else ""
        )

        self._embedding_model_name: str = embedding_model
        self.embedding_model: BaseEmbeddingModel | None = None
        self.vector_enabled: bool = bool(embedding_model)
        self.fts_enabled: bool = fts_enabled
        if not self.vector_enabled and not self.fts_enabled:
            raise ValueError("At least one of embedding_model or fts_enabled must be set.")

    # -- Lifecycle ---------------------------------------------------------

    async def _start(self) -> None:
        if not self._embedding_model_name:
            return
        assert self.app_context is not None, "app_context must be provided"
        models = self.app_context.components.get(ComponentEnum.EMBEDDING_MODEL, {})
        if self._embedding_model_name not in models:
            raise ValueError(f"Embedding model '{self._embedding_model_name}' not found.")
        model = models[self._embedding_model_name]
        if not isinstance(model, BaseEmbeddingModel):
            raise TypeError(f"Expected BaseEmbeddingModel, got {type(model).__name__}")
        self.embedding_model = model

    async def _close(self) -> None:
        self.embedding_model = None

    # -- Embedding (shared helper) -----------------------------------------

    async def embed(self, texts: list[str]) -> list[list[float] | None] | None:
        """Embed a batch of texts. None on disabled / API failure.

        Returns a list parallel to `texts` whose entries may individually
        be None if the embedding model dropped them. The whole call
        returns None if vector search is disabled or the API errors out
        (which also auto-disables vector search for the rest of the
        process).
        """
        if not self.vector_enabled or not texts or self.embedding_model is None:
            return None
        try:
            return await self.embedding_model.get_embeddings(texts)
        except Exception as e:
            self._disable_vector_search(str(e))
            return None

    def _disable_vector_search(self, reason: str = "embedding API error") -> None:
        if self.vector_enabled:
            self.logger.warning(f"[{self.store_name}] Disabling vector search: {reason}")
            self.vector_enabled = False

    # -- Composite write entries (concrete, in terms of the four abstracts) -

    async def upsert(self, node: FileNode, chunks: list[FileChunk]) -> None:
        """Persist a node and its chunks together. Node first, then chunks."""
        await self.upsert_node(node)
        await self.upsert_chunks(node.path, chunks)

    async def delete(self, path: str) -> None:
        """Delete chunks then node — chunks-first avoids orphan rows."""
        await self.delete_chunks(path)
        await self.delete_node(path)

    # -- Abstract surface (subclasses implement) ---------------------------

    @abstractmethod
    async def upsert_node(self, node: FileNode) -> None:
        """Persist a single node, replacing any prior entry for `node.path`."""

    @abstractmethod
    async def delete_node(self, path: str) -> None:
        """Delete the node entry for `path`. Chunks are managed separately."""

    @abstractmethod
    async def get_node(self, path: str) -> FileNode | None:
        """Fetch a single node by path. None if absent."""

    @abstractmethod
    async def upsert_chunks(self, path: str, chunks: list[FileChunk]) -> None:
        """Insert or replace all chunks for `path`.

        The store owns the embedding pipeline: it should hash-diff
        incoming chunks against persisted ones, reuse cached embeddings
        for unchanged blocks, and only call the embedding API for new
        hashes.
        """

    @abstractmethod
    async def delete_chunks(self, path: str) -> None:
        """Delete all chunks for `path`."""

    @abstractmethod
    async def get_chunks(self, path: str) -> list[FileChunk]:
        """All chunks for `path`."""

    @abstractmethod
    async def vector_search(
        self, query: str, limit: int, search_filter: dict,
    ) -> list[FileChunk]:
        """Vector similarity search."""

    @abstractmethod
    async def keyword_search(
        self, query: str, limit: int, search_filter: dict,
    ) -> list[FileChunk]:
        """Full-text / keyword search."""
