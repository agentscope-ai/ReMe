"""File store module for persistent memory management.

Provides storage backends for memory chunks and file metadata with
vector and full-text search capabilities.
"""

from .base_file_store import BaseFileStore
from .local_file_store import LocalFileStore

__all__ = [
    "BaseFileStore",
    "LocalFileStore",
]
