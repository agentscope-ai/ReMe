"""Markdown file chunking utilities.

Provides functionality to split Markdown documents into smaller chunks
while maintaining overlap between consecutive chunks for context preservation.
"""

from .common_utils import hash_text
from ..schema import FileChunk


def chunk_markdown(
        text: str,
        path: str,
        chunk_tokens: int,
        overlap: int,
) -> list[FileChunk]:
    """Split Markdown text into chunks with configurable size and overlap.

    Implements a sliding window approach to chunk Markdown content while
    preserving context through overlap between consecutive chunks. Token
    counts are approximated using a 1:4 ratio (1 token ≈ 4 characters).

    Args:
        text: Input Markdown text to be chunked.
        path: File path identifier for the source document.
        chunk_tokens: Maximum number of tokens per chunk. Will be converted
            to characters using the 1:4 ratio, with a minimum of 32 characters.
        overlap: Number of overlapping tokens between consecutive chunks.
            Helps maintain context across chunk boundaries.

    Returns:
        A list of FileChunk objects, each containing:
            - id: Unique identifier based on path, line numbers, and hash
            - path: The source file path
            - start_line: Starting line number (1-indexed)
            - end_line: Ending line number (1-indexed)
            - text: The chunk content
            - hash: SHA-256 hash of the chunk content

    Examples:
        >>> text = "# Header\\nParagraph content here.\\n\\n## Subheader"
        >>> chunks = chunk_markdown(text, "doc.md", chunk_tokens=100, overlap=20)
        >>> len(chunks)
        1
        >>> chunks[0].path
        'doc.md'
    """
    if not text.strip():
        return []

    lines = text.split("\n")

    # Convert tokens to characters (~1 token = 4 chars)
    max_chars = max(32, chunk_tokens * 4)
    overlap_chars = max(0, overlap * 4)

    chunks: list[FileChunk] = []

    # Currently building chunk
    current: list[dict] = []  # [{'line': str, 'line_no': int}]
    current_chars = 0

    def flush() -> None:
        """Add current chunk to results list."""
        if not current:
            return

        first_entry = current[0]
        last_entry = current[-1]

        if not first_entry or not last_entry:
            return

        chunk_text = "\n".join([entry["line"] for entry in current])
        start_line = first_entry["line_no"]
        end_line = last_entry["line_no"]

        chunk_hash = hash_text(chunk_text)

        chunks.append(
            FileChunk(
                id=hash_text(f"{path}:{start_line}:{end_line}:{chunk_hash}:{len(chunks)}"),
                path=path,
                start_line=start_line,
                end_line=end_line,
                text=chunk_text,
                hash=chunk_hash,
            ),
        )

    def carry_overlap() -> None:
        """Keep overlapping part and clear the rest."""
        nonlocal current, current_chars

        if overlap_chars <= 0 or not current:
            current = []
            current_chars = 0
            return

        acc = 0
        kept = []

        # Collect lines from the end until reaching overlap size
        for j in range(len(current) - 1, -1, -1):
            entry = current[j]
            if not entry:
                continue

            acc += len(entry["line"]) + 1  # +1 for newline
            kept.insert(0, entry)  # Insert at the beginning to maintain order

            if acc >= overlap_chars:
                break

        current = kept
        current_chars = sum(len(entry["line"]) + 1 for entry in kept)

    for i, line in enumerate(lines):
        line_no = i + 1

        # Split long lines into multiple segments
        segments = []
        if not line:  # Empty line
            segments.append("")
        else:
            # If line is too long, split by maximum character count
            for start in range(0, len(line), max_chars):
                segments.append(line[start: start + max_chars])

        for segment in segments:
            line_size = len(segment) + 1  # +1 for newline

            # If adding current segment would exceed the limit, flush current chunk
            if current_chars + line_size > max_chars and current:
                flush()
                carry_overlap()

            current.append({"line": segment, "line_no": line_no})
            current_chars += line_size

    # Process the final chunk
    flush()

    return [c for c in chunks if c.text.strip()]
