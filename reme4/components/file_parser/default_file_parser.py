"""Default file parser with byte-based overlapping chunking."""

from bisect import bisect_right
from pathlib import Path

import aiofiles
import yaml

from .base_file_parser import BaseFileParser
from ..component_registry import R
from ...schema import FileChunk, FileNode, FileFrontMatter


@R.register("default")
class DefaultFileParser(BaseFileParser):
    """Parser that splits files into byte-based overlapping chunks."""

    def __init__(self, encoding: str = "utf-8", chunk_byte_size: int = 10000, overlap_byte_size: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding
        self.chunk_byte_size = max(100, chunk_byte_size)
        self.overlap_byte_size = max(4, overlap_byte_size)

    @staticmethod
    def _parse_front_matter(text: str) -> tuple[FileFrontMatter, str]:
        """Parse YAML front matter delimited by ---, return (front_matter, remaining)."""
        if not text.startswith("---"):
            return FileFrontMatter(), text
        end_idx = text.find("\n---", 3)
        if end_idx == -1:
            return FileFrontMatter(), text
        try:
            data = yaml.safe_load(text[3:end_idx].strip()) or {}
            front_matter = FileFrontMatter(**(data if isinstance(data, dict) else {}))
        except yaml.YAMLError:
            front_matter = FileFrontMatter()
        return front_matter, text[end_idx + 4 :].lstrip("\n")

    async def parse(self, path: str | Path) -> tuple[FileNode, list[FileChunk]]:
        file_path = Path(path)
        stat = file_path.stat()
        rel_path = self._get_relative_path(path)

        async with aiofiles.open(file_path, encoding=self.encoding) as f:
            text = await f.read()

        if not text:
            return FileNode(path=rel_path, st_mtime=stat.st_mtime), []

        front_matter, content = self._parse_front_matter(text)
        if not content:
            return FileNode(path=rel_path, st_mtime=stat.st_mtime, front_matter=front_matter), []

        chunks = self._chunk_content(content, rel_path)
        chunk_ids = [c.id for c in chunks]
        return FileNode(path=rel_path, st_mtime=stat.st_mtime, front_matter=front_matter, chunk_ids=chunk_ids), chunks

    def _chunk_content(self, content: str, rel_path: str) -> list[FileChunk]:
        """Split content into overlapping byte-range chunks with line numbers."""
        content_bytes = content.encode(self.encoding)
        newline_positions = [i for i, b in enumerate(content_bytes) if b == ord("\n")]
        chunks: list[FileChunk] = []
        step = self.chunk_byte_size - self.overlap_byte_size
        start = 0

        while start < len(content_bytes):
            end = min(start + self.chunk_byte_size, len(content_bytes))
            chunk_text = content_bytes[start:end].decode(self.encoding, errors="ignore")
            start_line = bisect_right(newline_positions, start - 1) + 1
            end_line = bisect_right(newline_positions, end - 1) + 1
            if content_bytes[end - 1] == ord("\n"):
                end_line -= 1
            chunks.append(
                FileChunk(path=rel_path, start_line=start_line, end_line=end_line, text=chunk_text).set_hash_id(),
            )
            if end >= len(content_bytes):
                break
            start += step

        return chunks
