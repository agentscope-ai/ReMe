"""Default file chunker — byte-based overlapping chunking."""

from bisect import bisect_right
from pathlib import Path

import aiofiles
import yaml

from .base_file_chunker import BaseFileChunker
from ..component_registry import R
from ...schema import FileChunk, FileFrontMatter, FileNode
from ...utils.wikilink_handler import WikilinkHandler


@R.register("default")
class DefaultFileChunker(BaseFileChunker):
    """Default chunker that splits files into byte-based overlapping chunks."""

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
        rel_path = self.to_vault_relative(path)

        async with aiofiles.open(file_path, encoding=self.encoding) as f:
            text = await f.read()

        if not text:
            return FileNode(path=rel_path, st_mtime=stat.st_mtime), []

        is_markdown = file_path.suffix.lower() == ".md"
        if is_markdown:
            front_matter, content = self._parse_front_matter(text)
            if not content:
                return FileNode(path=rel_path, st_mtime=stat.st_mtime, front_matter=front_matter), []
            links = WikilinkHandler.extract_links(content, rel_path)
        else:
            front_matter = FileFrontMatter()
            content = text
            links = []

        chunks = self._chunk_content(content, rel_path, parse_links=is_markdown)
        chunk_ids = [c.id for c in chunks]
        return (
            FileNode(
                path=rel_path,
                st_mtime=stat.st_mtime,
                front_matter=front_matter,
                links=links,
                chunk_ids=chunk_ids,
            ),
            chunks,
        )

    def _link_byte_spans(self, content: str) -> list[tuple[int, int]]:
        """Return [start, end) byte spans of every wikilink in content."""
        spans: list[tuple[int, int]] = []
        last_char, last_byte = 0, 0
        for wm in WikilinkHandler.iter_matches(content):
            last_byte += len(content[last_char : wm.start].encode(self.encoding))
            match_bytes = len(content[wm.start : wm.end].encode(self.encoding))
            spans.append((last_byte, last_byte + match_bytes))
            last_byte += match_bytes
            last_char = wm.end
        return spans

    @staticmethod
    def _span_containing(
        pos: int,
        spans: list[tuple[int, int]],
        starts: list[int],
    ) -> tuple[int, int] | None:
        """Return the span strictly containing pos (s < pos < e), or None."""
        idx = bisect_right(starts, pos) - 1
        if idx < 0:
            return None
        s, e = spans[idx]
        return (s, e) if s < pos < e else None

    def _chunk_content(self, content: str, rel_path: str, parse_links: bool = True) -> list[FileChunk]:
        """Split content into overlapping byte-range chunks, avoiding cuts inside wikilinks.

        When ``parse_links`` is False, skip wikilink span computation and boundary checks
        — used for non-markdown files where wikilink semantics don't apply.
        """
        content_bytes = content.encode(self.encoding)
        n = len(content_bytes)
        newline_positions = [i for i, b in enumerate(content_bytes) if b == ord("\n")]
        if parse_links:
            link_spans = self._link_byte_spans(content)
            link_starts = [s for s, _ in link_spans]
        else:
            link_spans: list[tuple[int, int]] = []
            link_starts: list[int] = []
        # Refuse to retreat past half of chunk_byte_size; falls back to hard cut
        # for pathologically long links so we always make forward progress.
        min_chunk = self.chunk_byte_size // 2
        chunks: list[FileChunk] = []
        start = 0

        while start < n:
            end = min(start + self.chunk_byte_size, n)
            if end < n:
                span = self._span_containing(end, link_spans, link_starts)
                if span is not None and span[0] - start >= min_chunk:
                    end = span[0]

            chunk_text = content_bytes[start:end].decode(self.encoding, errors="ignore")
            start_line = bisect_right(newline_positions, start - 1) + 1
            end_line = bisect_right(newline_positions, end - 1) + 1
            if content_bytes[end - 1] == ord("\n"):
                end_line -= 1
            chunks.append(
                FileChunk(path=rel_path, start_line=start_line, end_line=end_line, text=chunk_text).set_hash_id(),
            )
            if end >= n:
                break

            next_start = end - self.overlap_byte_size
            span = self._span_containing(next_start, link_spans, link_starts)
            if span is not None:
                next_start = span[1]
            if next_start <= start:
                next_start = end
            start = next_start

        return chunks
