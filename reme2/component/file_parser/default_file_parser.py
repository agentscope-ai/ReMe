"""Default file parser with byte-based chunking."""

from pathlib import Path

import aiofiles

from .base_file_parser import BaseFileParser
from ..component_registry import R
from ...schema import FileChunk, FileNode


@R.register("default")
class DefaultFileParser(BaseFileParser):
    """Parser for files using byte-based chunking."""

    def __init__(self, encoding: str = "utf-8", chunk_byte_size: int = 10000, overlap_byte_size: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding
        self.chunk_byte_size = max(100, chunk_byte_size)
        # overlap >= 4 ensures truncated multibyte UTF-8 chars decode correctly in next chunk
        self.overlap_byte_size = max(4, overlap_byte_size)

    async def parse(self, path: str | Path) -> tuple[FileNode, list[FileChunk]]:
        file_path = Path(path)
        stat = file_path.stat()
        relative_path = self._get_relative_path(path)

        async with aiofiles.open(file_path, "rb") as f:
            data = await f.read()

        if not data:
            return FileNode(path=relative_path, st_mtime=stat.st_mtime), []

        # Find all newline byte positions
        newline_positions = [i for i, b in enumerate(data) if b == ord(b"\n")]

        chunks: list[FileChunk] = []
        step = self.chunk_byte_size - self.overlap_byte_size
        start_byte = 0

        while start_byte < len(data):
            end_byte = min(start_byte + self.chunk_byte_size, len(data))
            chunk_data = data[start_byte:end_byte]
            text = chunk_data.decode(self.encoding, errors="ignore")

            # Calculate line numbers from byte positions
            start_line = sum(1 for p in newline_positions if p < start_byte) + 1
            end_line = sum(1 for p in newline_positions if p < end_byte) + 1

            chunks.append(FileChunk(
                path=relative_path,
                start_line=start_line,
                end_line=end_line,
                text=text,
            ))

            start_byte += step if end_byte < len(data) else len(data)

        return FileNode(path=relative_path, st_mtime=stat.st_mtime), chunks
