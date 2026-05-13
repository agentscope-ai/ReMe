from pathlib import Path

import aiofiles

from .base_file_parser import BaseFileParser
from ..component_registry import R
from ...schema import FileChunk, FileNode
from ...utils.common_utils import hash_text


@R.register("default")
class DefaultFileParser(BaseFileParser):
    """Parser for files using byte-based chunking."""

    def __init__(self, encoding: str = "utf-8", chunk_byte_size: int = 10000, overlap_byte_size: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding
        self.chunk_byte_size = max(100, chunk_byte_size)
        self.overlap_byte_size = max(4, overlap_byte_size)

    async def parse(self, path: str | Path) -> tuple[FileNode, list[FileChunk]]:
        file_path = Path(path)
        stat = file_path.stat()
        rel_path = self._get_relative_path(path)

        async with aiofiles.open(file_path, "rb") as f:
            data = await f.read()

        if not data:
            return FileNode(path=rel_path, st_mtime=stat.st_mtime), []

        newline_positions = [i for i, b in enumerate(data) if b == ord(b"\n")]
        chunks: list[FileChunk] = []
        step = self.chunk_byte_size - self.overlap_byte_size
        start = 0

        while start < len(data):
            end = min(start + self.chunk_byte_size, len(data))
            text = data[start:end].decode(self.encoding, errors="ignore")
            start_line = sum(1 for p in newline_positions if p < start) + 1
            end_line = sum(1 for p in newline_positions if p < end) + 1

            chunks.append(FileChunk(
                path=rel_path,
                start_line=start_line,
                end_line=end_line,
                text=text,
            ).set_hash_id())

            start += step if end < len(data) else len(data)

        return FileNode(path=rel_path, st_mtime=stat.st_mtime, chunk_ids=[c.id for c in chunks]), chunks