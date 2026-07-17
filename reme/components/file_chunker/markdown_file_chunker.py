"""Markdown file chunker — frontmatter + wikilink graph + AST tree chunks.

When ``embed_toc`` is enabled, a chunk that starts inside a section carries
only that section's ancestor heading breadcrumb. Sibling headings remain in
the document stream and are stored once, avoiding the quadratic growth caused
by repeating the complete document outline in every chunk.

Pipeline: count headings without an AST → use plain-text byte chunks when the
configured section limit is exceeded; otherwise build a mistletoe AST →
``MdNode`` tree (sections nest by heading level) → recursively chunk children
and merge adjacent small subtrees at their parent. Leaf blocks (table / code /
list / paragraph) split on internal boundaries and each piece is annotated
``[Part X/N]``. Wikilink extraction is
delegated to :class:`reme.utils.wikilink_handler.WikilinkHandler` —
the single source of truth for ``[[...]]`` syntax (including
Dataview-style typed predicates).
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError


from .base_file_chunker import BaseFileChunker
from .default_file_chunker import DefaultFileChunker
from ..component_registry import R
from ...schema import (
    FileChunk,
    FileFrontMatter,
    FileNode,
)
from ...utils.wikilink_handler import WikilinkHandler

# -- AST node + helpers ---------------------------------------------------

_ATX_HEADING_RE = re.compile(r"^ {0,3}#{1,6}(?:[ \t]+|$)")
_SETEXT_HEADING_RE = re.compile(r"^ {0,3}(?:=+|-+)[ \t]*$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


@dataclass
class MdNode:
    """``root`` / ``section`` (heading + children until equal-or-shallower
    heading) / ``body`` (one mistletoe block; ``block`` keeps the original).

    ``text`` is the rendered subtree (own heading excluded for sections).
    Line ranges span the full subtree.
    """

    kind: str  # "root" | "section" | "body"
    heading: str | None = None
    level: int = 0
    children: list["MdNode"] = field(default_factory=list)
    block: Any = None
    text: str = ""
    start_line: int = 0
    end_line: int = 0


def _heading_text(node: Any, renderer) -> str:
    """Heading text without `#` markers (for outline)."""
    rendered = renderer.render(node).rstrip("\n")
    if rendered.startswith("#"):
        return rendered.lstrip("#").strip()
    return rendered.split("\n", 1)[0].strip()


def _finalize(n: MdNode) -> None:
    """Bottom-up pass: propagate line ranges, populate ``n.text`` (rendered
    subtree, own heading excluded for sections)."""
    parts: list[str] = []
    for c in n.children:
        _finalize(c)
        if c.kind == "section":
            heading = f"{'#' * c.level} {c.heading or ''}"
            parts.append(f"{heading}\n\n{c.text}" if c.text else heading)
        elif c.text:
            parts.append(c.text)
    if n.children:
        first = n.children[0].start_line
        n.start_line = min(n.start_line, first) if n.start_line else first
        n.end_line = max(c.end_line for c in n.children)
    elif n.end_line < n.start_line:
        n.end_line = n.start_line
    if n.kind != "body":
        n.text = "\n\n".join(parts)


def _toc_join(*parts: str) -> str:
    """Concatenate TOC fragments with ``\\n\\n``, skipping empty ones."""
    return "\n\n".join(p for p in parts if p)


# -- Chunker --------------------------------------------------------------


@R.register("markdown")
class MarkdownFileChunker(BaseFileChunker):
    """Markdown chunker with breadcrumb context and adjacent-section packing."""

    def __init__(
        self,
        encoding: str = "utf-8",
        chunk_chars: int = 10000,
        embed_toc: bool = True,
        max_ast_sections: int | None = 100,
        default_chunker: str = "default",
        include_frontmatter_in_metadata: bool = False,
        include_frontmatter_keys_in_metadata: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoding = encoding
        self.chunk_chars = max(100, chunk_chars)
        self.embed_toc = embed_toc
        self.max_ast_sections = max(0, max_ast_sections) if max_ast_sections is not None else None
        self.default_chunker = self.bind(default_chunker, BaseFileChunker, optional=False)
        self.include_frontmatter_in_metadata = include_frontmatter_in_metadata
        self.include_frontmatter_keys_in_metadata = list(include_frontmatter_keys_in_metadata or [])

    async def chunk(self, path: str | Path) -> tuple[FileNode, list[FileChunk]]:
        file_path = Path(path)
        rel_path = self.to_workspace_relative(path)
        front_matter, content, line_offset = self._parse_front_matter(file_path.read_text(encoding=self.encoding))

        chunks: list[FileChunk] = []
        if content and content.strip():
            section_count = self._count_sections(content, stop_after=self.max_ast_sections)
            if self.max_ast_sections is not None and section_count > self.max_ast_sections:
                self.logger.info(
                    f"Markdown AST skipped for {rel_path}: sections>{self.max_ast_sections}; "
                    "using plain-text chunking",
                )
                chunks = self._chunk_plain_text(content, rel_path, line_offset)
            else:
                from mistletoe.markdown_renderer import MarkdownRenderer
                from mistletoe.block_token import Document

                with MarkdownRenderer() as renderer:
                    tree = self._build_tree(Document(content), renderer, line_offset=line_offset)
                    chunks = self._chunk_node(tree, (), rel_path, renderer)
            if self.include_frontmatter_in_metadata:
                chunk_metadata = self._chunk_metadata(
                    front_matter,
                    allow_keys=self.include_frontmatter_keys_in_metadata or None,
                )
                for chunk in chunks:
                    chunk.metadata = chunk_metadata.copy()

        links = WikilinkHandler.extract_links(content, rel_path) if content else []

        node = FileNode(
            path=rel_path,
            st_mtime=file_path.stat().st_mtime,
            chunk_ids=[chunk.id for chunk in chunks],
            links=links,
            front_matter=front_matter,
        )
        return node, chunks

    @staticmethod
    def _count_sections(content: str, stop_after: int | None = None) -> int:
        """Count block-level ATX/Setext headings without building a Markdown AST.

        Fenced code content is ignored. When ``stop_after`` is set, return as
        soon as the count exceeds it because the caller only needs to choose
        between AST and plain-text chunking.
        """
        count = 0
        fence_char = ""
        fence_length = 0
        previous_text = False

        line_start = 0
        while line_start <= len(content):
            newline = content.find("\n", line_start)
            if newline < 0:
                line = content[line_start:]
            else:
                line = content[line_start:newline]
            if line.endswith("\r"):
                line = line[:-1]

            fence_match = _FENCE_RE.match(line)
            if fence_match:
                marker = fence_match.group(1)
                if not fence_char:
                    fence_char = marker[0]
                    fence_length = len(marker)
                elif marker[0] == fence_char and len(marker) >= fence_length and not line[fence_match.end() :].strip():
                    fence_char = ""
                    fence_length = 0
                previous_text = False
            elif not fence_char:
                is_atx = _ATX_HEADING_RE.match(line) is not None
                is_setext = previous_text and _SETEXT_HEADING_RE.match(line) is not None
                if is_atx or is_setext:
                    count += 1
                    if stop_after is not None and count > stop_after:
                        return count
                previous_text = bool(line.strip()) and not is_atx and not is_setext

            if newline < 0:
                break
            line_start = newline + 1

        return count

    def _chunk_plain_text(self, content: str, path: str, line_offset: int) -> list[FileChunk]:
        """Use the default byte chunker while preserving original Markdown lines."""
        if not isinstance(self.default_chunker, DefaultFileChunker):
            raise RuntimeError("DefaultFileChunker dependency is unavailable; call start() before chunk().")
        chunks = self.default_chunker.chunk_content(content, path, parse_links=True)
        if line_offset:
            for chunk in chunks:
                chunk.start_line += line_offset
                chunk.end_line += line_offset
                chunk.set_hash_id()
        return chunks

    @staticmethod
    def _parse_front_matter(text: str) -> tuple[FileFrontMatter, str, int]:
        """Parse YAML frontmatter, returning 1-based line offset for body AST lines.

        Invalid YAML is ignored so a single bad frontmatter block does not
        prevent indexing the markdown body.
        """
        lines = text.splitlines(keepends=True)
        if not lines or lines[0].strip() != "---":
            return FileFrontMatter(), text, 0

        close_idx = next((i for i, line in enumerate(lines[1:], 1) if line.strip() == "---"), None)
        if close_idx is None:
            return FileFrontMatter(), text, 0

        front_matter = FileFrontMatter()
        try:
            data = yaml.safe_load("".join(lines[1:close_idx]).strip()) or {}
            if isinstance(data, dict):
                front_matter = FileFrontMatter(**data)
        except (yaml.YAMLError, TypeError, ValidationError):
            front_matter = FileFrontMatter()

        return front_matter, "".join(lines[close_idx + 1 :]), close_idx + 1

    @staticmethod
    def _chunk_metadata(
        front_matter: FileFrontMatter,
        allow_keys: list[str] | None = None,
    ) -> dict:
        """Expose frontmatter fields on chunks so search filters can match them.

        ``allow_keys`` (when non-empty) restricts the output to that subset of
        field names; absent or empty means "all fields with non-empty values".
        """
        dumped = front_matter.model_dump(mode="json")
        filtered = dumped if not allow_keys else {k: dumped.get(k) for k in allow_keys if k in dumped}
        return {key: value for key, value in filtered.items() if value not in (None, "")}

    def _build_tree(self, doc: Any, renderer, line_offset: int = 0) -> MdNode:
        """Heading-level stack folds mistletoe's flat children into nested
        sections; non-headings attach as ``body`` to the current section
        (or root before the first heading)."""
        from mistletoe.markdown_renderer import BlankLine
        from mistletoe.block_token import (
            Heading,
            SetextHeading,
        )

        root = MdNode(kind="root", start_line=line_offset + 1, end_line=line_offset + 1)
        stack: list[MdNode] = [root]
        for child in doc.children or []:
            if isinstance(child, BlankLine):
                continue
            raw_line = getattr(child, "line_number", None)
            line = raw_line + line_offset if raw_line is not None else stack[-1].start_line
            if isinstance(child, (Heading, SetextHeading)):
                level = max(1, getattr(child, "level", 1))
                while len(stack) > 1 and stack[-1].level >= level:
                    stack.pop()
                sec = MdNode(
                    kind="section",
                    heading=_heading_text(child, renderer),
                    level=level,
                    start_line=line,
                )
                stack[-1].children.append(sec)
                stack.append(sec)
                continue
            rendered = renderer.render(child).rstrip("\n")
            if not rendered:
                continue
            stack[-1].children.append(
                MdNode(
                    kind="body",
                    block=child,
                    text=rendered,
                    start_line=line,
                    end_line=line + rendered.count("\n"),
                ),
            )
        _finalize(root)
        return root

    # -- Recursive subtree chunking --------------------------------------

    def _chunk_node(
        self,
        node: MdNode,
        ancestors: tuple[str, ...],
        path: str,
        renderer,
    ) -> list[FileChunk]:
        """Greedily assemble child subtrees, recursing only on oversized ones.

        A fitting subtree is returned immediately. For an oversized container,
        fitting children are appended directly to a local ``FileChunk`` cache;
        only an oversized child calls ``_chunk_node`` recursively. A full cache
        is finalized before assembly continues in document order.
        """
        prefix = _toc_join(*ancestors)
        heading = ""
        subtree_text = node.text
        if node.kind == "section":
            heading = f"{'#' * node.level} {node.heading or ''}"
            subtree_text = _toc_join(heading, node.text)

        if subtree_text and len(subtree_text) <= self.chunk_chars:
            return [
                self._make_chunk(
                    prefix,
                    subtree_text,
                    "",
                    node.start_line,
                    node.end_line,
                    path,
                ),
            ]

        if node.kind == "body":
            return self._split_leaf(node, prefix, "", path, renderer)

        child_ancestors = ancestors
        if node.kind == "section":
            child_ancestors = (*ancestors, heading)

        chunks: list[FileChunk] = []
        cache: FileChunk | None = None
        child_prefix = _toc_join(*child_ancestors)

        def flush_cache() -> None:
            nonlocal cache
            if cache is None:
                return
            chunks.append(cache.set_hash_id())
            cache = None

        def append_to_cache(text: str, start_line: int, end_line: int, new_prefix: str) -> None:
            nonlocal cache
            if not text:
                return
            if cache is not None:
                candidate = _toc_join(cache.text, text)
                if len(candidate) <= self.chunk_chars:
                    cache.text = candidate
                    cache.end_line = end_line
                    if len(candidate) == self.chunk_chars:
                        flush_cache()
                    return
                flush_cache()

            cache_text = _toc_join(new_prefix, text) if self.embed_toc else text
            cache = FileChunk(
                path=path,
                start_line=start_line,
                end_line=end_line,
                text=cache_text,
            )
            if len(cache_text) >= self.chunk_chars:
                flush_cache()

        for child in node.children:
            child_heading = f"{'#' * child.level} {child.heading or ''}" if child.kind == "section" else ""
            child_text = _toc_join(child_heading, child.text) if child_heading else child.text
            if len(child_text) <= self.chunk_chars:
                append_to_cache(child_text, child.start_line, child.end_line, child_prefix)
                continue

            flush_cache()
            chunks.extend(self._chunk_node(child, child_ancestors, path, renderer))

        flush_cache()
        if node.kind == "section":
            if not chunks:
                return [
                    self._make_chunk(
                        prefix,
                        heading,
                        "",
                        node.start_line,
                        node.start_line,
                        path,
                    ),
                ]
            first_content = self._without_prefix(chunks[0].text, child_prefix)
            chunks[0] = self._make_chunk(
                prefix,
                _toc_join(heading, first_content),
                "",
                node.start_line,
                chunks[0].end_line,
                path,
            )
        return chunks

    def _without_prefix(self, text: str, prefix: str) -> str:
        """Remove a breadcrumb that this chunker prepended to ``text``."""
        if not self.embed_toc or not prefix:
            return text
        if text == prefix:
            return ""
        marker = f"{prefix}\n\n"
        return text[len(marker) :] if text.startswith(marker) else text

    # -- Leaf splitters: build (text, start, end) units, hand off to packer

    def _split_leaf(
        self,
        body: MdNode,
        before: str,
        after: str,
        path: str,
        renderer,
    ) -> list[FileChunk]:
        from mistletoe.block_token import (
            CodeFence,
            List,
            Table,
        )

        block = body.block
        if isinstance(block, Table):
            return self._split_table(body, before, after, path)
        if isinstance(block, CodeFence):
            return self._split_code(body, before, after, path)
        if isinstance(block, List):
            return self._split_list(body, before, after, path, renderer)
        return self._split_lines(body, before, after, path)

    def _split_table(
        self,
        body: MdNode,
        before: str,
        after: str,
        path: str,
    ) -> list[FileChunk]:
        """Repeat header + separator on every chunk."""
        from mistletoe.block_token import TableRow

        lines = body.text.split("\n")
        header, data = "\n".join(lines[:2]), lines[2:]
        rows = [r for r in (body.block.children or []) if isinstance(r, TableRow)]
        line_offset = body.start_line - (getattr(body.block, "line_number", None) or body.start_line)
        base = body.start_line + 2

        def line_of(i: int) -> int:
            return rows[i].line_number + line_offset if i < len(rows) and rows[i].line_number else base + i

        units = [(text, line_of(i), line_of(i)) for i, text in enumerate(data)]
        return self._emit_packed(
            units,
            before,
            after,
            path,
            joiner="\n",
            wrap=f"{header}\n{{inner}}",
        )

    def _split_code(
        self,
        body: MdNode,
        before: str,
        after: str,
        path: str,
    ) -> list[FileChunk]:
        """Repeat fence opener + closer on every chunk."""
        code = body.block
        indent = " " * (code.indentation or 0)
        fence = f"{indent}{code.delimiter}"
        opener = f"{fence}{code.info_string or ''}"
        raw = (code.children[0].content if code.children else "").rstrip("\n")
        if not raw:
            return []
        start = body.start_line + 1
        units = [(indent + ln, start + i, start + i) for i, ln in enumerate(raw.split("\n"))]
        return self._emit_packed(
            units,
            before,
            after,
            path,
            joiner="\n",
            wrap=f"{opener}\n{{inner}}\n{fence}",
            allow_empty=True,
        )

    def _split_list(
        self,
        body: MdNode,
        before: str,
        after: str,
        path: str,
        renderer,
    ) -> list[FileChunk]:
        """Pack list items; oversized items emit alone (overflow accepted)."""
        from mistletoe.block_token import ListItem

        items = [c for c in (body.block.children or []) if isinstance(c, ListItem)]
        if not items:
            return self._split_lines(body, before, after, path)
        units: list[tuple[str, int, int]] = []
        line_offset = body.start_line - (getattr(body.block, "line_number", None) or body.start_line)
        for it in items:
            text = renderer.render(it).rstrip("\n")
            if not text:
                continue
            line = it.line_number + line_offset if it.line_number else body.start_line
            units.append((text, line, line + text.count("\n")))
        return self._emit_packed(
            units,
            before,
            after,
            path,
            joiner="\n",
            wrap="{inner}",
        )

    def _split_lines(
        self,
        body: MdNode,
        before: str,
        after: str,
        path: str,
    ) -> list[FileChunk]:
        """Last-resort line-greedy split for paragraphs / quotes / html."""
        start = body.start_line
        units = [(line, start + i, start + i) for i, line in enumerate(body.text.split("\n"))]
        return self._emit_packed(
            units,
            before,
            after,
            path,
            joiner="\n",
            wrap="{inner}",
        )

    def _emit_packed(
        self,
        units: list[tuple[str, int, int]],
        before: str,
        after: str,
        path: str,
        joiner: str,
        wrap: str,
        allow_empty: bool = False,
    ) -> list[FileChunk]:
        """Greedy-pack units into ``wrap`` envelopes; emit each piece.

        Envelope (table header, code fence) counts against ``chunk_chars``;
        TOC (when on) is additive prefix/suffix downstream. Oversized
        units overflow rather than truncate. Multi-piece outputs get
        ``[Part X/N]`` markers; single pieces don't.
        """
        envelope = len(wrap.replace("{inner}", ""))
        budget = max(64, self.chunk_chars - envelope)
        sep_len = len(joiner)

        parts: list[tuple[str, int, int]] = []
        bucket: list[tuple[str, int, int]] = []
        bucket_chars = 0

        def flush() -> None:
            nonlocal bucket, bucket_chars
            if not bucket:
                return
            inner = joiner.join(t for t, _, _ in bucket)
            parts.append((inner, bucket[0][1], bucket[-1][2]))
            bucket = []
            bucket_chars = 0

        for text, s, e in units:
            if not text and not allow_empty:
                continue
            sep = sep_len if bucket else 0
            if bucket_chars + sep + len(text) > budget:
                flush()
                sep = 0
            bucket.append((text, s, e))
            bucket_chars += sep + len(text)
        flush()

        total = len(parts)
        return [
            self._make_chunk(
                before,
                (
                    f"[Part {idx}/{total}]\n\n{wrap.replace('{inner}', inner)}"
                    if total > 1
                    else wrap.replace("{inner}", inner)
                ),
                after,
                s,
                e,
                path,
            )
            for idx, (inner, s, e) in enumerate(parts, 1)
        ]

    # -- Emit -------------------------------------------------------------

    def _make_chunk(
        self,
        before: str,
        content: str,
        after: str,
        start_line: int,
        end_line: int,
        path: str,
    ) -> FileChunk:
        """Build one ``FileChunk`` — text is ``before + content + after``
        when ``embed_toc``, otherwise just ``content``."""
        text = _toc_join(before, content, after) if self.embed_toc else content
        return FileChunk(
            path=path,
            start_line=start_line,
            end_line=end_line,
            text=text,
        ).set_hash_id()
