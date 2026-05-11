"""Chunk-search helpers — small pure functions used by file_store backends.

Pulled out of `BaseFileStore` so the abstract base only carries graph
mechanics. Both `LocalFileStore` and `SqliteFileStore` consume these
to score / filter candidate chunks.
"""

from __future__ import annotations

from reme2.schema import ChunkFilter, FileChunk


def keyword_score(query: str, text: str) -> float:
    """Word-overlap score with phrase bonus. Range [0, 1].

    `match_count / n_words` baseline + 0.2 if the full query phrase
    appears verbatim. Returns 0 when no query word hits the text.
    """
    words = query.split()
    if not words:
        return 0.0
    query_lower = query.lower()
    words_lower = [w.lower() for w in words]
    text_lower = text.lower()
    n_words = len(words)

    match_count = sum(1 for w in words_lower if w in text_lower)
    if match_count == 0:
        return 0.0

    base = match_count / n_words
    phrase_bonus = 0.2 if n_words > 1 and query_lower in text_lower else 0.0
    return min(1.0, base + phrase_bonus)


def filter_chunks(
    chunks: list[FileChunk], chunk_filter: ChunkFilter | None,
) -> list[FileChunk]:
    """Restrict `chunks` to those whose path passes the (compiled) filter."""
    if chunk_filter is None or chunk_filter.resolved_paths is None:
        return chunks
    return [c for c in chunks if chunk_filter.match_path(c.path)]
