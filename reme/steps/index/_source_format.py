"""Shared helper: render retrieved chunks, compacting raw session transcripts.

Raw session transcripts (``*.jsonl`` under the dialog dir) store one serialized
``Msg`` per line. :func:`render_chunk_body` turns those back into a readable
dialog; all other chunks keep their raw ``text``. Used by
``search``/``vector_search``/``bm25_search`` so every step renders session hits
identically.
"""

from agentscope.message import Msg

from ..evolve._evolve import format_history
from ...schema import FileChunk


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
