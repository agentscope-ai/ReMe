"""JSONL file chunker — line-aligned sliding-window chunking."""

from pathlib import Path

from .base_file_chunker import BaseFileChunker
from ..component_registry import R
from ...schema import FileChunk, FileNode


@R.register("jsonl")
class JsonlFileChunker(BaseFileChunker):
    """Chunker for JSONL (JSON Lines) files.

    Each line is treated as an atomic unit — splits are always on line
    boundaries.  A sliding window with configurable overlap produces
    consecutive chunks.

    Algorithm
    ---------
    1. From ``start``, greedily accumulate lines until adding the next line
       would exceed ``max_chars`` → emit that chunk.
    2. From the *end* of the emitted chunk, walk backwards to find the
       maximum number of trailing lines whose combined size fits within
       ``max_overlap_chars`` → those lines become the start of the next chunk.
    3. Repeat until the entire file is covered.

    The size metric can be switched between **character count** (default)
    and **byte count** via the ``mode`` parameter.
    """

    def __init__(
        self,
        encoding: str = "utf-8",
        max_chars: int = 2000,
        max_overlap_chars: int = 0,
        mode: str = "chars",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if mode not in ("chars", "bytes"):
            raise ValueError(f"mode must be 'chars' or 'bytes', got {mode!r}")
        self.encoding = encoding
        self.max_chars = max(64, max_chars)
        self.max_overlap_chars = max(0, max_overlap_chars)
        self.mode = mode

    # ------------------------------------------------------------------
    # Size helpers
    # ------------------------------------------------------------------

    def _line_size(self, line: str) -> int:
        """Return the size of *line* including its trailing newline (if any)."""
        if self.mode == "bytes":
            return len(line.encode(self.encoding))
        return len(line)

    def _text_size(self, text: str) -> int:
        """Return the size of arbitrary text."""
        if self.mode == "bytes":
            return len(text.encode(self.encoding))
        return len(text)

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def _chunk_lines(self, lines: list[str]) -> list[tuple[int, int]]:
        """Return a list of ``(start_idx, end_idx)`` half-open line ranges.

        ``start_idx`` is inclusive, ``end_idx`` is exclusive (standard Python
        slice convention).  Indices are 0-based.
        """
        if not lines:
            return []

        sizes = [self._line_size(ln) for ln in lines]
        n = len(lines)
        ranges: list[tuple[int, int]] = []
        start = 0

        while start < n:
            # -- Greedy forward: find the largest end such that
            #    sum(sizes[start:end]) <= max_chars  --
            total = 0
            end = start
            while end < n and total + sizes[end] <= self.max_chars:
                total += sizes[end]
                end += 1

            # At least one line per chunk, even if it alone exceeds max_chars
            # (we cannot split within a line).
            if end == start:
                end = start + 1

            ranges.append((start, end))

            if end >= n:
                break

            # -- Overlap: from the end of the just-emitted chunk, walk
            #    backwards to find trailing lines that fit in max_overlap_chars --
            overlap_count = 0
            overlap_size = 0
            for i in range(end - 1, start - 1, -1):
                next_size = overlap_size + sizes[i]
                if next_size > self.max_overlap_chars:
                    break
                overlap_size = next_size
                overlap_count += 1

            # Next chunk starts at (end - overlap_count).
            # Guard: if overlap somehow covers the whole chunk, advance by 1
            # to prevent an infinite loop.
            next_start = end - overlap_count
            if next_start <= start:
                next_start = end
            start = next_start

        return ranges

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def chunk(self, path: str | Path) -> tuple[FileNode, list[FileChunk]]:
        """Read and chunk a JSONL file at *path*."""
        file_path = Path(path)
        stat = file_path.stat()
        rel_path = self.to_workspace_relative(path)

        text = file_path.read_text(encoding=self.encoding)
        if not text.strip():
            return FileNode(path=rel_path, st_mtime=stat.st_mtime), []

        # readlines keeps trailing \n on each line.
        lines = text.splitlines(keepends=True)
        if not lines:
            return FileNode(path=rel_path, st_mtime=stat.st_mtime), []

        ranges = self._chunk_lines(lines)

        file_chunks: list[FileChunk] = []
        for s, e in ranges:
            chunk_text = "".join(lines[s:e])
            # 1-based inclusive line numbers.
            file_chunks.append(
                FileChunk(
                    path=rel_path,
                    start_line=s + 1,
                    end_line=e,
                    text=chunk_text,
                ).set_hash_id(),
            )

        node = FileNode(
            path=rel_path,
            st_mtime=stat.st_mtime,
            chunk_ids=[c.id for c in file_chunks],
            links=[],
        )
        return node, file_chunks
