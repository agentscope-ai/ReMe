"""Markdown file parser."""

import asyncio
from pathlib import Path

import frontmatter

from .base_file_parser import BaseFileParser
from ..component_registry import R
from ...schema import FileChunk, FileMetadata
from ...utils import hash_text, chunk_markdown


@R.register("md")
class MdFileParser(BaseFileParser):
    """Parser for Markdown files with YAML frontmatter support."""

    suffixes = [".md", ".markdown"]

    def __init__(self, encoding: str = "utf-8", **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding

    async def parse(self, path: str) -> tuple[FileMetadata, list[FileChunk]]:
        file_path = Path(path)

        def _read_and_parse():
            raw = file_path.read_text(encoding=self.encoding)
            post = frontmatter.loads(raw)
            stat = file_path.stat()
            return stat, dict(post.metadata), post.content

        stat, metadata, content = await asyncio.to_thread(_read_and_parse)

        file_meta = FileMetadata(
            hash=hash_text(content),
            mtime_ms=stat.st_mtime * 1000,
            size=stat.st_size,
            path=str(file_path.absolute()),
            content=content,
            metadata=metadata,
        )

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
