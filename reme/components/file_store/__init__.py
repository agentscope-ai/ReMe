"""File store module.

In-memory + JSONL backend for the (file → chunks) graph. Subclass
`BaseFileStore` to add other backends; only `LocalFileStore` is
shipped today.
"""

from .base_file_store import BaseFileStore
from .faiss_local_file_store import FaissLocalFileStore
from .local_file_store import LocalFileStore
from .zvec_local_file_store import ZvecLocalFileStore

__all__ = [
    "BaseFileStore",
    "FaissLocalFileStore",
    "LocalFileStore",
    "ZvecLocalFileStore",
]
