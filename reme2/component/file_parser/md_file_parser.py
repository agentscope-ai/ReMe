"""Markdown file parser."""

import asyncio
from pathlib import Path

import frontmatter

from .base_file_parser import BaseFileParser
from ..component_registry import R
from ...enumeration import FileSuffixEnum
from ...schema import FileChunk, FileMetadata
from ...utils import chunk_markdown


@R.register("md")
class MdFileParser(BaseFileParser):
    """Parser for Markdown files with YAML frontmatter support."""

    suffixes = [FileSuffixEnum.MD, FileSuffixEnum.MARKDOWN]

    def __init__(self, encoding: str = "utf-8", chunk_tokens: int = 400, chunk_overlap: int = 80, **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding
        self.chunk_tokens = chunk_tokens
        self.chunk_overlap = chunk_overlap

    async def parse(self, path: str) -> tuple[FileMetadata, list[FileChunk]]:
        file_path = Path(path)

        def _read_and_parse():
            raw = file_path.read_text(encoding=self.encoding)
            post = frontmatter.loads(raw)
            stat = file_path.stat()
            return stat, dict(post.metadata), post.content

        stat, metadata, content = await asyncio.to_thread(_read_and_parse)

        file_meta = FileMetadata(
            modified_time=stat.st_mtime,
            path=str(file_path.absolute()),
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

        return file_meta, chunks
