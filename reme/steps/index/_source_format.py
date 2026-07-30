"""Shared helpers: render retrieved chunks and assemble search-step answers.

Raw session transcripts (``*.jsonl`` under the dialog dir) store one serialized
``Msg`` per line. :func:`render_chunk_body` turns those back into a readable
dialog; all other chunks keep their raw ``text``. Used by
``search``/``vector_search``/``bm25_search`` so every step renders session hits
identically.

:func:`format_chunks_answer` assembles a full answer string from a list of
chunks, with optional per-chunk score formatting and link expansion. Raw
session chunks from the same file whose line ranges overlap, contain one
another, or are adjacent are merged into their union before rendering (see
:func:`_merge_session_chunk_intervals`) so a passage is never shown twice.
:data:`ALL_RETURNED_MESSAGE` is the English notice shown when tool_context dedup
removes every previously-returned result. :data:`NO_RESULTS_MESSAGE` is the
English notice shown when the search returned no results at all.
"""

from typing import Callable, Final

from agentscope.message import Msg

from ..evolve._evolve import format_history
from ...schema import FileChunk
from ...utils.link_expansion import render_expansion_lines

#: English message written to ``response.answer`` when tool_context dedup
#: removed every result that was already returned in previous responses.
ALL_RETURNED_MESSAGE: Final[str] = (
    "All retrieved content has already been returned in previous responses; " "no new content was found."
)

#: English message written to ``response.answer`` when the search returned no
#: results at all (before dedup).
NO_RESULTS_MESSAGE: Final[str] = "No relevant information was found for the given query."


def is_session_chunk(chunk: FileChunk, dialog_dir: str) -> bool:
    """True if the chunk comes from a raw session transcript (a jsonl file under the dialog dir)."""
    path = (chunk.path or "").strip().strip("/")
    if not path.endswith(".jsonl"):
        return False
    dialog_dir = (dialog_dir or "").strip("/")
    return path == dialog_dir or path.startswith(f"{dialog_dir}/")


def render_chunk_body(chunk: FileChunk, dialog_dir: str) -> str:
    """Render a chunk's body, compacting raw session transcripts into a readable form.

    Session chunks are jsonl where each line is a serialized ``Msg``. Parse every line
    and render via :func:`format_history`; on any parse error (or no usable messages),
    fall back to the chunk's raw ``text``.
    """
    if not is_session_chunk(chunk, dialog_dir):
        return chunk.text
    try:
        messages: list[Msg] = []
        for line in chunk.text.splitlines():
            line = line.strip()
            if not line:
                continue
            messages.append(Msg.model_validate_json(line))
        if not messages:
            return chunk.text
        return format_history(messages)
    except Exception:
        return chunk.text


def _build_union_chunk(group: list[FileChunk]) -> FileChunk:
    """Fuse a set of same-file session chunks into one covering their union.

    Each chunk's ``text`` is line-aligned: text line ``i`` maps to file line
    ``start_line + i`` (1-based). Lines are keyed by their absolute file line
    number so overlapping regions collapse to a single copy, then emitted in
    ascending line order — this preserves the original message chronology and
    never reorders content within the merged passage. The highest-scoring chunk
    is used as the template so retrieval scores are carried through the header.

    Every entry in ``line_map`` is normalised to end with ``\n`` before joining
    so that a chunk whose text lacks a trailing newline (e.g. the last line of
    a file with no final newline) does not collide with the next line.
    """
    rep = max(group, key=lambda c: c.score)
    line_map: dict[int, str] = {}
    for c in group:
        for offset, line in enumerate(c.text.splitlines(keepends=True)):
            line_map[c.start_line + offset] = line
    parts = [line_map[k] for k in sorted(line_map)]
    union_text = "".join(p if p.endswith("\n") else f"{p}\n" for p in parts)
    return rep.model_copy(
        update={
            "start_line": min(c.start_line for c in group),
            "end_line": max(c.end_line for c in group),
            "text": union_text,
        },
    )


def _merge_session_chunk_intervals(chunks: list[FileChunk], dialog_dir: str) -> list[FileChunk]:
    """Merge raw session chunks from the same file into their line-range union.

    Only chunks recognized as raw session transcripts (see
    :func:`is_session_chunk`) are considered; every other chunk passes through
    unchanged. Within one session file, chunks are grouped by ascending line
    range and merged when the next chunk's ``start_line`` is ``<= end + 1`` of
    the group so far — covering the three overlap relations:

    * containment: one range fully inside another;
    * intersection: ranges partially overlap;
    * adjacency: ``prev.end_line + 1 == next.start_line`` (gap-free consecutive
      chunks, per :class:`~reme.components.file_chunker.JsonlFileChunker`).

    Each merged group renders once as its union. Ordering: all units belonging
    to one session file are kept adjacent and sorted by ascending ``start_line``;
    the file as a whole is placed at the rank of its earliest-appearing chunk,
    and non-session chunks keep their original rank position.
    """
    session_by_path: dict[str, list[tuple[int, FileChunk]]] = {}
    # (order_key, start_line, chunk): order_key ties all of a session file's
    # units to that file's earliest rank so they sort adjacently, while
    # non-session chunks use their own rank and thus keep their position.
    ordered: list[tuple[int, int, FileChunk]] = []
    for idx, c in enumerate(chunks):
        if is_session_chunk(c, dialog_dir):
            session_by_path.setdefault(c.path, []).append((idx, c))
        else:
            ordered.append((idx, c.start_line, c))

    for items in session_by_path.values():
        path_rank = min(idx for idx, _ in items)
        items.sort(key=lambda t: (t[1].start_line, t[1].end_line))
        group: list[FileChunk] = []
        group_end: int | None = None
        for _, c in items:
            if group and group_end is not None and c.start_line <= group_end + 1:
                group.append(c)
                group_end = max(group_end, c.end_line)
            else:
                if group:
                    ordered.append((path_rank, group[0].start_line, _finalize_group(group)))
                group = [c]
                group_end = c.end_line
        if group:
            ordered.append((path_rank, group[0].start_line, _finalize_group(group)))

    ordered.sort(key=lambda t: (t[0], t[1]))
    return [c for _, _, c in ordered]


def _finalize_group(group: list[FileChunk]) -> FileChunk:
    """Collapse a merge group into one chunk; a single member passes through unchanged."""
    if len(group) == 1:
        return group[0]
    return _build_union_chunk(group)


def format_chunks_answer(
    chunks: list[FileChunk],
    dialog_dir: str,
    *,
    include_source: bool = True,
    score_fn: Callable[[FileChunk], str] | None = None,
    link_expansion: dict[str, dict] | None = None,
) -> str:
    """Render a list of chunks into a single answer string.

    Raw session chunks from the same file that overlap, contain one another, or
    are adjacent are merged into their union first (see
    :func:`_merge_session_chunk_intervals`).

    When *include_source* is ``True`` (default), each chunk is prefixed with a
    source header showing path, line range, and score. When ``False``, only the
    rendered bodies are included, separated by blank lines.

    *score_fn* customizes the score string in the header (default
    ``"score={chunk.score:.4f}"``). *link_expansion* appends per-path expansion
    lines after each chunk's header block (used by hybrid search).
    """
    chunks = _merge_session_chunk_intervals(chunks, dialog_dir)
    if not include_source:
        return "\n\n".join(render_chunk_body(c, dialog_dir) for c in chunks)

    fmt = score_fn or (lambda c: f"score={c.score:.4f}")
    lines: list[str] = []
    for c in chunks:
        lines.append(
            f"========== {c.path}:{c.start_line}-{c.end_line} "
            f"[{fmt(c)}] ==========\n{render_chunk_body(c, dialog_dir)}",
        )
        if link_expansion:
            lines.extend(render_expansion_lines(link_expansion.get(c.path, {})))
    return "\n".join(lines)
