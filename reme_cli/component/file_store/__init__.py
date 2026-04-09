"""File store module for persistent memory management.

This module provides storage backends for memory chunks and file metadata,
including pure-Python local implementations with vector and full-text search.
"""

from .base_file_store import BaseFileStore
from .local_file_store import LocalFileStore

__all__ = [
    "BaseFileStore",
    "LocalFileStore",
]