"""Memory store module for persistent memory management.

This module provides storage backends for memory chunks and file metadata,
including SQLite-based, ChromaDB-based, and pure-Python local implementations
with vector and full-text search.
"""

from .base_memory_store import BaseMemoryStore
from .chroma_memory_store import ChromaMemoryStore
from .local_memory_store import LocalMemoryStore
from .sqlite_memory_store import SqliteMemoryStore
from ..context import R

__all__ = [
    "BaseMemoryStore",
    "ChromaMemoryStore",
    "LocalMemoryStore",
    "SqliteMemoryStore",
]

R.memory_stores.register("sqlite")(SqliteMemoryStore)
R.memory_stores.register("chroma")(ChromaMemoryStore)
R.memory_stores.register("local")(LocalMemoryStore)
