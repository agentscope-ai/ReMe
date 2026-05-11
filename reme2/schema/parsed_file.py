"""ParsedFile — Parser's single output, carrying everything one file pass produces.

Inherits `FileMetadata` (file / path / st_mtime / metadata) and adds the
parsed contents:

    chunks: list[FileChunk]   — semantic blocks (text + position + hash);
                                embeddings ARE attached here — the parser
                                owns the embedding step (with hash-diff
                                cache via `existing_chunks` parameter).
    edges:  list[FileEdge]    — wikilink graph edges (regex / frontmatter /
                                LLM provenance preserved per-edge).

The Watcher passes a `ParsedFile` into `file_store.upsert_parsed(...)`,
which dispatches the three sub-payloads (meta / edges / chunks) to the
appropriate persistence pipelines while keeping the upsert atomic at
the store layer.
"""

from pydantic import Field

from .file_chunk import FileChunk
from .file_edge import FileEdge
from .file_node import FileMetadata


class ParsedFile(FileMetadata):
    chunks: list[FileChunk] = Field(default_factory=list)
    edges: list[FileEdge] = Field(default_factory=list)
