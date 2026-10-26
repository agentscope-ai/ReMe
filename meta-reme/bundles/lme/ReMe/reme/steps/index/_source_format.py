"""Shared helpers: render retrieved chunks and assemble search-step answers.

Raw session transcripts (``*.jsonl`` under the dialog dir) store one serialized
``Msg`` per line. :func:`render_chunk_body` renders those line-aligned — one
message per line with internal newlines flattened; all other chunks keep their
raw ``text``. Used by
``search``/``vector_search``/``bm25_search`` so every step renders session hits
identically.

:func:`render_chunk_entries` renders each chunk into a ``{"path",
"start_line", "end_line", "score", "body", "link"}`` entry dict, with optional
per-chunk score formatting and link expansion; :func:`join_chunk_entries`
assembles the source headers and joins those entries into the final answer
string. Chunks are rendered as given; callers that want raw session chunks from
the same file with overlapping, contained, or adjacent line ranges collapsed
into their union should pre-merge them via :func:`merge_session_chunk_intervals`
so a passage is never shown twice.
:data:`ALL_RETURNED_MESSAGE` is the English notice shown when tool_context dedup
removes every previously-returned result. :data:`NO_RESULTS_MESSAGE` is the
English notice shown when the search returned no results at all.
"""

import posixpath
from typing import Callable, Final

from agentscope.message import Msg

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


def normalize_posix_path(path: str) -> str:
    """Return a normalized, workspace-relative POSIX path string."""
    normalized = posixpath.normpath((path or "").strip().replace("\\", "/").strip("/"))
    return "" if normalized == "." else normalized


def is_session_path(path: str, session_dir: str) -> bool:
    """True if the path points at a raw transcript under ``{session_dir}/dialog``."""
    path = normalize_posix_path(path)
    if not path.endswith(".jsonl"):
        return False
    session_root = normalize_posix_path(session_dir)
    session_dialog_dir = posixpath.join(session_root, "dialog")
    return path == session_dialog_dir or path.startswith(f"{session_dialog_dir}/")


def is_session_chunk(chunk: FileChunk, session_dir: str) -> bool:
    """True if the chunk comes from a raw session transcript (a jsonl file under the dialog dir)."""
    return is_session_path(chunk.path, session_dir)


def render_chunk_body(chunk: FileChunk, session_dir: str) -> str:
    """Render a chunk's body; raw session transcripts render one message per line.

    Session chunks are jsonl where each line is a serialized ``Msg``. They
    render via :func:`render_session_chunk_lines` — one message per line with
    internal newlines flattened — so verbatim and compressed session bodies
    share the same single-line-per-message format. All other chunks keep
    their raw ``text``.
    """
    if not is_session_chunk(chunk, session_dir):
        return chunk.text
    return "\n".join(render_session_chunk_lines(chunk))


def render_session_chunk_lines(chunk: FileChunk) -> list[str]:
    """Render a raw session chunk line-aligned with the original jsonl file.

    Output line ``i`` maps to file line ``chunk.start_line + i`` (1-based).
    Each jsonl line (one serialized ``Msg`` per line) becomes a single rendered
    line ``[speaker @ created_at] content`` with internal whitespace and
    newlines flattened to single spaces. Blank lines stay blank and
    unparseable lines pass through stripped, so the file-line mapping is
    never broken.
    """
    rendered: list[str] = []
    for raw in chunk.text.splitlines():
        line = raw.strip()
        if not line:
            rendered.append("")
            continue
        try:
            msg = Msg.model_validate_json(line)
        except Exception:
            rendered.append(line)
            continue
        speaker = msg.name or msg.role or "?"
        content = " ".join((msg.get_text_content() or "").split())
        rendered.append(f"[{speaker} @ {msg.created_at}] {content}".rstrip())
    return rendered


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


def merge_session_chunk_intervals(chunks: list[FileChunk], session_dir: str) -> list[FileChunk]:
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
        if is_session_chunk(c, session_dir):
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


def render_chunk_entries(
    chunks: list[FileChunk],
    session_dir: str,
    *,
    include_source: bool = True,
    score_fn: Callable[[FileChunk], str] | None = None,
    link_expansion: dict[str, dict] | None = None,
) -> list[dict[str, str]]:
    """Render each chunk into an entry dict of source fields, body, and link.

    Chunks are rendered exactly as given, in input order. Callers that want
    overlapping/contained/adjacent raw session chunks collapsed into their
    union should pre-merge the list via :func:`merge_session_chunk_intervals`
    before calling.

    When *include_source* is ``True`` (default), each entry carries the source
    fields ``path`` / ``start_line`` / ``end_line`` / ``score`` (the formatted
    score string) plus ``body`` and ``link`` (the per-path expansion lines,
    empty string when there is none). When ``False``, each entry carries only
    ``body``.

    *score_fn* customizes the formatted ``score`` string (default
    ``"score={chunk.score:.4f}"``). *link_expansion* provides the per-path
    expansion lines (used by hybrid search).
    """
    if not include_source:
        return [{"body": render_chunk_body(c, session_dir)} for c in chunks]

    fmt = score_fn or (lambda c: f"score={c.score:.4f}")
    entries: list[dict[str, str]] = []
    for c in chunks:
        entries.append(
            {
                "path": c.path,
                "start_line": str(c.start_line),
                "end_line": str(c.end_line),
                "score": fmt(c),
                "body": render_chunk_body(c, session_dir),
                "link": "\n".join(render_expansion_lines((link_expansion or {}).get(c.path, {}))),
            },
        )
    return entries


def join_chunk_entries(entries: list[dict[str, str]]) -> str:
    """Assemble entries into the final answer string.

    For each entry, a source header line is rebuilt from ``path`` /
    ``start_line`` / ``end_line`` / ``score`` when ``path`` is present, then
    followed by the non-empty ``body`` and ``link`` parts, joined with ``\\n``.
    Entries are separated by a blank line; missing or empty parts are skipped.
    """
    parts: list[str] = []
    for entry in entries:
        tmp = []
        if path := entry.get("path", "").strip():
            tmp.append(
                f"========== {path}:{entry.get('start_line', '')}-{entry.get('end_line', '')} "
                f"[{entry.get('score', '')}] ==========",
            )
        for key in ("body", "link"):
            value = entry.get(key, "").strip()
            if not value:
                continue
            tmp.append(value)
        if tmp:
            parts.append("\n".join(tmp))
    return "\n\n".join(parts)
