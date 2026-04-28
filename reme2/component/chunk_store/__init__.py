"""Chunk store module.

Storage backends for FileChunks with vector and full-text search.
File metadata is managed by FileGraph, not by ChunkStore.
"""

from .base_chunk_store import BaseChunkStore
from .chroma_chunk_store import ChromaChunkStore
from .local_chunk_store import LocalChunkStore
from .sqlite_chunk_store import SqliteChunkStore

__all__ = [
    "BaseChunkStore",
    "ChromaChunkStore",
    "LocalChunkStore",
    "SqliteChunkStore",
]
