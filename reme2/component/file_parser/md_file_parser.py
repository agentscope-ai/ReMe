"""Markdown file parser — frontmatter + wikilink graph + AST-aware chunking.

The chunker splits markdown into semantic blocks using:
  - ATX headings (#, ##, ..., ######) as section anchors
  - Blank lines (paragraph boundaries) as soft splits within a section
  - Code fences (``` or ~~~) preserved as a single block

Each block carries a `heading_path` breadcrumb prepended to its text — gives
the embedding model section context AND lets retrieval results show callers
where the hit lives. The hash is computed over the final text (with
breadcrumb), so renaming a heading correctly invalidates child block
embeddings.

Hash-diff cache compatibility: blocks with identical (heading_path + body)
across edits produce the same hash, so the watcher can reuse old embeddings
and only call the embedding API for dirty blocks.

Edge extraction is delegated to a `BaseEdgeExtractor` resolved from the
app context (`edge_extractor` constructor arg, defaults to "default"). If
the app context isn't bound or the named extractor isn't registered, the
parser falls back to a local `RegexEdgeExtractor` so unit tests / simple
scripts can still parse without YAML wiring.
"""

import re
from pathlib import Path

import frontmatter

from .base_file_parser import BaseFileParser
from ..component_registry import R
from ..edge_extractor import BaseEdgeExtractor, RegexEdgeExtractor
from ...enumeration import ComponentEnum, FileSuffixEnum
from ...schema import FileChunk, ParsedFile
from ...utils import hash_text


@R.register("md")
class MdFileParser(BaseFileParser):
    """Parser for Markdown files with YAML frontmatter and wikilink support."""

    suffixes = [FileSuffixEnum.MD, FileSuffixEnum.MARKDOWN]

    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
    _FENCE_RE = re.compile(r"^(```|~~~)")

    def __init__(
        self,
        encoding: str = "utf-8",
        edge_extractor: str = "default",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoding = encoding
        self._edge_extractor_name: str = edge_extractor
        self.edge_extractor: BaseEdgeExtractor | None = None

    async def _start(self) -> None:
        await super()._start()
        # Try to resolve the configured extractor; fall back to a local
        # regex extractor when no app_context / no matching component.
        resolved: BaseEdgeExtractor | None = None
        if self.app_context is not None and self._edge_extractor_name:
            extractors = self.app_context.components.get(ComponentEnum.EDGE_EXTRACTOR, {})
            candidate = extractors.get(self._edge_extractor_name)
            if isinstance(candidate, BaseEdgeExtractor):
                resolved = candidate
        if resolved is None:
            resolved = RegexEdgeExtractor()
            await resolved.start()
        self.edge_extractor = resolved

    async def _close(self) -> None:
        await super()._close()
        self.edge_extractor = None

    async def parse(
        self,
        path: str,
        existing_chunks: list[FileChunk] | None = None,
    ) -> ParsedFile:
        file_path = Path(path)
        raw = file_path.read_text(encoding=self.encoding)
        post = frontmatter.loads(raw)
        stat = file_path.stat()
        metadata = dict(post.metadata)
        content = post.content
        absolute_path = str(file_path.absolute())

        extractor = self.edge_extractor or RegexEdgeExtractor()
        edges = await extractor.extract(content, metadata, path=absolute_path)

        chunks = self.chunk_markdown(content, absolute_path)
        dirty = self._hash_diff_attach(chunks, existing_chunks)
        if dirty:
            await self._embed_chunks(dirty)

        return ParsedFile(
            file=file_path.stem,
            path=absolute_path,
            st_mtime=stat.st_mtime,
            metadata=metadata,
            edges=edges,
            chunks=chunks,
        )

    # -- Chunker ----------------------------------------------------------

    @staticmethod
    def _breadcrumb(heading_path: list[str]) -> str:
        return " > ".join(heading_path) if heading_path else ""

    @classmethod
    def _make_block_text(cls, heading_path: list[str], body: str) -> str:
        """Compose final block text: breadcrumb line (if any) + blank + body."""
        body = body.rstrip("\n")
        crumb = cls._breadcrumb(heading_path)
        if crumb:
            return f"{crumb}\n\n{body}" if body else crumb
        return body

    @classmethod
    def chunk_markdown(cls, text: str, path: str) -> list[FileChunk]:
        """Split markdown into AST-aware blocks (headings / paragraphs / fences)."""
        if not text or not text.strip():
            return []

        lines = text.split("\n")
        chunks: list[FileChunk] = []

        heading_stack: list[tuple[int, str]] = []  # [(level, title)]
        body_lines: list[str] = []
        body_start = 1
        in_fence = False
        fence_marker = ""

        def current_path() -> list[str]:
            return [t for _, t in heading_stack]

        def emit(block_text: str, start_line: int, end_line: int) -> None:
            h = hash_text(block_text)
            chunks.append(
                FileChunk(
                    id=hash_text(f"{path}::{start_line}::{end_line}::{h}::{len(chunks)}"),
                    path=path,
                    start_line=start_line,
                    end_line=end_line,
                    text=block_text,
                    hash=h,
                ),
            )

        def flush_body(end_line: int) -> None:
            nonlocal body_lines, body_start
            if not body_lines:
                return
            # Strip leading/trailing blank lines from the block (paragraph
            # boundaries eat their own newline, but whitespace can sneak in
            # via the fence path).
            while body_lines and not body_lines[0].strip():
                body_lines.pop(0)
                body_start += 1
            while body_lines and not body_lines[-1].strip():
                body_lines.pop()
                end_line -= 1
            if not body_lines:
                body_lines = []
                return
            raw_body = "\n".join(body_lines)
            block_text = cls._make_block_text(current_path(), raw_body)
            emit(block_text, body_start, end_line)
            body_lines = []

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Code fences: keep contents intact, no inner splits.
            if not in_fence and cls._FENCE_RE.match(stripped):
                if body_lines:
                    flush_body(end_line=i - 1)
                in_fence = True
                fence_marker = stripped[:3]
                body_lines = [line]
                body_start = i
                continue
            if in_fence:
                body_lines.append(line)
                if stripped.startswith(fence_marker):
                    in_fence = False
                    fence_marker = ""
                    flush_body(end_line=i)
                continue

            # ATX heading: closes prior block, opens a new section.
            m = cls._HEADING_RE.match(line)
            if m:
                if body_lines:
                    flush_body(end_line=i - 1)
                level = len(m.group(1))
                title = m.group(2).strip()
                heading_stack = [(lv, t) for lv, t in heading_stack if lv < level]
                heading_stack.append((level, title))

                # Heading line itself becomes a block (so the heading text is
                # searchable as its own unit).
                block_text = cls._make_block_text(current_path(), "")
                emit(block_text, i, i)
                body_start = i + 1
                continue

            # Blank line: paragraph boundary.
            if not stripped:
                if body_lines:
                    flush_body(end_line=i - 1)
                body_start = i + 1
                continue

            # Regular content line.
            if not body_lines:
                body_start = i
            body_lines.append(line)

        if body_lines:
            flush_body(end_line=len(lines))

        return chunks
