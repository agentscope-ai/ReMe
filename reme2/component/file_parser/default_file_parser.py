"""Default file parser for unknown file types."""

import asyncio
import hashlib
from pathlib import Path

from .base_file_parser import BaseFileParser
from ..component_registry import R
from ...schema import FileChunk, FileMetadata
from ...utils import hash_text, chunk_markdown


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
                content = raw.decode(self.encoding)
                content_hash = hash_text(content)
                return stat, content_hash, content
            except (UnicodeDecodeError, ValueError):
                binary_hash = hashlib.sha256(raw).hexdigest()
                return stat, binary_hash, None

        stat, file_hash, content = await asyncio.to_thread(_read_file)

        file_meta = FileMetadata(
            hash=file_hash,
            mtime_ms=stat.st_mtime * 1000,
            size=stat.st_size,
            path=str(file_path.absolute()),
            content=content,
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

        file_meta.chunk_count = len(chunks)
        return file_meta, chunks
