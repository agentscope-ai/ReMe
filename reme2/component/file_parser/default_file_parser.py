"""Default file parser for unknown file types."""

import asyncio
from pathlib import Path

from .base_file_parser import BaseFileParser
from ..component_registry import R
from ...schema import FileChunk, FileMetadata
from ...utils import chunk_markdown


@R.register("default")
class DefaultFileParser(BaseFileParser):
    """Fallback parser for unknown file types.

    Attempts to read as text and chunk. If the file is binary,
    stores metadata only with no chunks.
    """

    suffixes = []

    def __init__(self, encoding: str = "utf-8", **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding

    async def parse(self, path: str) -> tuple[FileMetadata, list[FileChunk]]:
        file_path = Path(path)

        def _read_file():
            stat = file_path.stat()
            raw = file_path.read_bytes()
            try:
                return stat, raw.decode(self.encoding)
            except (UnicodeDecodeError, ValueError):
                return stat, None

        stat, content = await asyncio.to_thread(_read_file)

        file_meta = FileMetadata(
            modified_time=stat.st_mtime,
            path=str(file_path.absolute()),
        )

        chunks: list[FileChunk] = []
        if content:
            chunks = (
                    chunk_markdown(
                        content,
                        file_meta.path,
                        self.chunk_tokens,
                        self.chunk_overlap,
                    )
                    or []
            )

        return file_meta, chunks
