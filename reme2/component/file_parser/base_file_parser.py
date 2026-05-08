"""Abstract base class for file parsers.

Single-pass parser: `parse(path, existing_chunks)` returns a `ParsedFile`
carrying metadata + chunks (with embeddings attached) + edges.

The parser owns the embedding pipeline. To avoid re-embedding unchanged
blocks, the watcher passes in the file's prior chunks; the parser
hash-diffs and only calls the embedding API for blocks whose hash is
new. Vanished hashes simply drop out (the file_store's upsert is a
delete-and-insert).

If `existing_chunks` is None / empty, every chunk is embedded fresh.
"""

from abc import abstractmethod

from ..base_component import BaseComponent
from ..embedding import BaseEmbeddingModel
from ...enumeration import ComponentEnum, FileSuffixEnum
from ...schema import FileChunk, ParsedFile


class BaseFileParser(BaseComponent):
    """Single-pass parser producing an embedded `ParsedFile`."""

    component_type = ComponentEnum.FILE_PARSER
    suffixes: list[FileSuffixEnum] = []

    def __init__(self, embedding_model: str = "", **kwargs):
        super().__init__(**kwargs)
        self._embedding_model_name: str = embedding_model
        self.embedding_model: BaseEmbeddingModel | None = None

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

    async def _embed_chunks(self, chunks: list[FileChunk]) -> list[FileChunk]:
        """Attach embeddings to chunks if an embedding model is configured.

        Failures (missing embeddings) leave `chunk.embedding=None`; the
        file_store can still keyword-search them.
        """
        if not chunks or self.embedding_model is None:
            return chunks
        try:
            await self.embedding_model.get_node_embeddings(chunks)
        except Exception as e:
            self.logger.warning(f"embedding chunks failed: {e}")
        return chunks

    @staticmethod
    def _hash_diff_attach(
        chunks: list[FileChunk],
        existing_chunks: list[FileChunk] | None,
    ) -> list[FileChunk]:
        """Attach cached embeddings to chunks whose hash already exists.

        Returns the dirty subset (chunks still needing embeddings). Mutates
        the input chunks in place: matched chunks get their `embedding`
        field set from the cache.
        """
        if not existing_chunks:
            return list(chunks)
        cached = {c.hash: c.embedding for c in existing_chunks if c.embedding}
        if not cached:
            return list(chunks)
        dirty: list[FileChunk] = []
        for c in chunks:
            cached_emb = cached.get(c.hash)
            if cached_emb is not None:
                c.embedding = cached_emb
            else:
                dirty.append(c)
        return dirty

    @abstractmethod
    async def parse(
        self,
        path: str,
        existing_chunks: list[FileChunk] | None = None,
    ) -> ParsedFile:
        """Parse + embed the file, returning a fully populated `ParsedFile`.

        `existing_chunks`, when provided, lets the parser reuse cached
        embeddings for blocks whose hash hasn't changed.
        """
