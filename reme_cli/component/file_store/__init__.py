"""File store module for persistent memory management.

This module provides storage backends for memory chunks and file metadata,
including SQLite-based, ChromaDB-based, and pure-Python local implementations
with vector and full-text search.
"""

from .base_file_store import BaseFileStore
from .chroma_file_store import ChromaFileStore
from .local_file_store import LocalFileStore
from .sqlite_file_store import SqliteFileStore
from ..registry_factory import R

__all__ = [
    "BaseFileStore",
    "ChromaFileStore",
    "LocalFileStore",
    "SqliteFileStore",
]

R.file_stores.register("sqlite")(SqliteFileStore)
R.file_stores.register("chroma")(ChromaFileStore)
R.file_stores.register("local")(LocalFileStore)
