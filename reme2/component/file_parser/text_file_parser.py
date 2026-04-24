"""Text file parser with built-in chunking."""

import hashlib
from pathlib import Path

import aiofiles

from .base_file_parser import BaseFileParser
from ..component_registry import R
from ...enumeration import FileSuffixEnum
from ...schema import FileChunk, FileMetadata


@R.register("text")
class TextFileParser(BaseFileParser):
    """Parser for text files with built-in chunking support."""

    suffixes = [FileSuffixEnum.TXT]

    def __init__(self, encoding: str = "utf-8", chunk_tokens: int = 400, chunk_overlap: int = 80, **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding
        self.chunk_size = max(32, chunk_tokens * 4)
        self.overlap_size = max(0, chunk_overlap * 4)

    async def parse(self, path: str) -> tuple[FileMetadata, list[FileChunk]]:
        file_path = Path(path)
        stat = file_path.stat()

        try:
            async with aiofiles.open(file_path, encoding=self.encoding) as f:
                content = await f.read()
        except UnicodeDecodeError:
            async with aiofiles.open(file_path, encoding=self.encoding, errors="ignore") as f:
                content = await f.read()
        except Exception:
            content = None

        file_meta = FileMetadata(
            file=file_path.stem,
            path=str(file_path.absolute()),
            st_mtime=stat.st_mtime,
        )

        chunks = self._chunk(content, file_meta.path) if content else []
        return file_meta, chunks

    def _chunk(self, text: str, path: str) -> list[FileChunk]:
        """Split text into chunks with overlap."""
        if not text.strip():
            return []

        lines = text.split("\n")
        chunks: list[FileChunk] = []
        buf: list[tuple[str, int]] = []  # (line_content, line_no)
        buf_chars = 0

        for line_no, line in enumerate(lines, 1):
            # Split long lines into segments
            for start in range(0, max(1, len(line)), self.chunk_size):
                seg = line[start:start + self.chunk_size]
                seg_chars = len(seg) + 1  # +1 for newline

                # Flush when buffer would exceed limit
                if buf and buf_chars + seg_chars > self.chunk_size:
                    self._flush_chunk(chunks, buf, path)
                    buf, buf_chars = self._carry_overlap(buf)

                buf.append((seg, line_no))
                buf_chars += seg_chars

        if buf:
            self._flush_chunk(chunks, buf, path)

        return [c for c in chunks if c.text.strip()]

    @staticmethod
    def _flush_chunk(chunks: list[FileChunk], buf: list[tuple[str, int]], path: str):
        """Create a chunk from buffer and append to chunks list."""
        chunk_text = "\n".join(content for content, _ in buf)
        start_line, end_line = buf[0][1], buf[-1][1]
        h = hashlib.sha256(chunk_text.encode()).hexdigest()
        chunk_id = hashlib.sha256(f"{path}:{start_line}:{end_line}:{h}:{len(chunks)}".encode()).hexdigest()

        chunks.append(FileChunk(
            id=chunk_id,
            path=path,
            start_line=start_line,
            end_line=end_line,
            text=chunk_text,
            hash=h,
        ))

    def _carry_overlap(self, buf: list[tuple[str, int]]) -> tuple[list[tuple[str, int]], int]:
        """Keep overlapping lines from the end of buffer."""
        if self.overlap_size <= 0 or not buf:
            return [], 0

        acc, kept = 0, []
        for content, line_no in reversed(buf):
            acc += len(content) + 1
            kept.insert(0, (content, line_no))
            if acc >= self.overlap_size:
                break

        buf_chars = sum(len(c) + 1 for c, _ in kept)
        return kept, buf_chars
