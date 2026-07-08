"""Shared helper: render search results with each hit's originating session_id.

Plain ``vector_search``/``bm25_search`` return only ``chunk.text``. The
LongMemEval agentic-answer flow needs each hit's ``session_id`` so the agent can
pivot back to the raw session via ``extract_session_by_id``. This helper reads
that ``session_id`` from the note's frontmatter and prefixes it to the text.
"""

from pathlib import Path

import frontmatter

from ...schema import FileChunk


def render_with_source(chunks: list[FileChunk], workspace_path: Path) -> str:
    """Render each chunk as ``[session_id=<sid>]`` header + text."""
    lines: list[str] = []
    for c in chunks:
        sid = ""
        if c.path:
            try:
                post = frontmatter.loads((workspace_path / c.path).read_text(encoding="utf-8"))
                sid = str((post.metadata or {}).get("session_id", "") or "").strip()
            except Exception:
                sid = ""
        header = f"[session_id={sid}]" if sid else "[session_id: unknown]"
        lines.append(f"{header}\n{c.text}")
    return "\n\n".join(lines)
