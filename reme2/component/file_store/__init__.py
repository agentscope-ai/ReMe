"""File store module.

Unified storage for file metadata (frontmatter + mtime), the wikilink
graph (edges per path), and chunks (text + embeddings) with vector /
keyword / hybrid search.

The store accepts a `ParsedFile` from the watcher's parsing pass via
`upsert_parsed(...)`, dispatching meta + edges + chunks into the
appropriate persistence pipelines.

Two backends:
    LocalFileStore   — pure-Python with JSONL persistence (default,
                       zero deps, fine for small vaults).
    SqliteFileStore  — SQLite + FTS5 + sqlite-vec for keyword/vector
                       at scale; meta + edges live in relational tables.
"""

from .base_file_store import BaseFileStore
from .local_file_store import LocalFileStore
from .sqlite_file_store import SqliteFileStore

__all__ = [
    "BaseFileStore",
    "LocalFileStore",
    "SqliteFileStore",
]
