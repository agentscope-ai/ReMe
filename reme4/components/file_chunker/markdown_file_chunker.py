"""Markdown file chunker — frontmatter + wikilink graph + AST tree chunks.

Each chunk carries the **complete heading skeleton** of the document
with its content inlined under the section that owns it; other sections
appear as bare headings so the reader always sees a full document map.

Pipeline: build mistletoe AST → ``MdNode`` tree (sections nest by
heading level) → recursive chunk (try whole subtree; on overflow walk
children — body siblings pack as a run, subsections recurse). Leaf
blocks (table / code / list / paragraph) split on internal boundaries
and each piece is annotated ``[Part X/N]``. Wikilink extraction is
delegated to :class:`reme.utils.wikilink_handler.WikilinkHandler` —
the single source of truth for ``[[...]]`` syntax (including
Dataview-style typed predicates).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import frontmatter


from .base_file_chunker import BaseFileChunker
from ..component_registry import R
from ...schema import (
    FileChunk,
    FileFrontMatter,
    FileNode,
)
from ...utils.wikilink_handler import WikilinkHandler


# -- AST node + helpers ---------------------------------------------------


@dataclass
class MdNode:
    """``root`` / ``section`` (heading + children until equal-or-shallower
    heading) / ``body`` (one mistletoe block; ``block`` keeps the original).

    ``text`` is the rendered subtree (own heading excluded for sections).
    ``desc_toc`` caches the section-only DFS outline of descendants
    (own heading excluded), used as the TOC suffix when emitting chunks
    inside a section. Line ranges span the full subtree.
    """

    kind: str  # "root" | "section" | "body"
    heading: str | None = None
    level: int = 0
    children: list["MdNode"] = field(default_factory=list)
    block: Any = None
    text: str = ""
    start_line: int = 0
    end_line: int = 0
    desc_toc: str = ""


def _heading_text(node: Any, renderer) -> str:
    """Heading text without `#` markers (for outline)."""
    rendered = renderer.render(node).rstrip("\n")
    if rendered.startswith("#"):
        return rendered.lstrip("#").strip()
    return rendered.split("\n", 1)[0].strip()


def _finalize(n: MdNode) -> None:
    """Bottom-up pass: propagate line ranges, populate ``n.text`` (rendered
    subtree, own heading excluded for sections) and ``n.desc_toc`` (DFS
    section outline of descendants)."""
    parts: list[str] = []
    desc_lines: list[str] = []
    for c in n.children:
        _finalize(c)
        if c.kind == "section":
            heading = f"{'#' * c.level} {c.heading or ''}"
            parts.append(f"{heading}\n\n{c.text}" if c.text else heading)
            desc_lines.append(f"{heading}\n\n{c.desc_toc}" if c.desc_toc else heading)
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
    n.desc_toc = "\n\n".join(desc_lines)


def _toc_join(*parts: str) -> str:
    """Concatenate TOC fragments with ``\\n\\n``, skipping empty ones."""
    return "\n\n".join(p for p in parts if p)


def _subtree_toc(n: MdNode) -> str:
    """Section's heading + descendants TOC — its contribution to a parent's
    ``desc_toc``. For root (no own heading) this is just ``desc_toc``."""
    if n.kind != "section" or n.heading is None:
        return n.desc_toc
    heading = f"{'#' * n.level} {n.heading}"
    return f"{heading}\n\n{n.desc_toc}" if n.desc_toc else heading


# -- Chunker --------------------------------------------------------------


@R.register("markdown")
class MarkdownFileChunker(BaseFileChunker):
    """Markdown chunker: frontmatter + wikilink edges + full-skeleton chunks."""

    def __init__(
        self,
        encoding: str = "utf-8",
        chunk_chars: int = 10000,
        embed_toc: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoding = encoding
        self.chunk_chars = max(100, chunk_chars)
        self.embed_toc = embed_toc

    async def parse(self, path: str | Path) -> tuple[FileNode, list[FileChunk]]:
        from mistletoe.markdown_renderer import MarkdownRenderer
        from mistletoe.block_token import Document

        file_path = Path(path)
        rel_path = self.to_vault_relative(path)
        post = frontmatter.loads(file_path.read_text(encoding=self.encoding))

        chunks: list[FileChunk] = []
        if post.content and post.content.strip():
            with MarkdownRenderer() as renderer:
                tree = self._build_tree(Document(post.content), renderer)
                chunks = self._chunk_node(tree, "", "", rel_path, renderer)

        links = WikilinkHandler.extract_links(post.content, rel_path) if post.content else []

        node = FileNode(
            path=rel_path,
            st_mtime=file_path.stat().st_mtime,
            chunk_ids=[chunk.id for chunk in chunks],
            links=links,
            front_matter=FileFrontMatter(**dict(post.metadata)),
        )
        return node, chunks

    def _build_tree(self, doc: Any, renderer) -> MdNode:
        """Heading-level stack folds mistletoe's flat children into nested
        sections; non-headings attach as ``body`` to the current section
        (or root before the first heading)."""
        from mistletoe.markdown_renderer import BlankLine
        from mistletoe.block_token import (
            Heading,
            SetextHeading,
        )

        root = MdNode(kind="root", start_line=1, end_line=1)
        stack: list[MdNode] = [root]
        for child in doc.children or []:
            if isinstance(child, BlankLine):
                continue
            line = getattr(child, "line_number", None) or stack[-1].start_line
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

    # -- Recursive chunker ------------------------------------------------

    def _chunk_node(
        self,
        node: MdNode,
        before: str,
        after: str,
        path: str,
        renderer,
    ) -> list[FileChunk]:
        """Try the whole subtree; on overflow split (leaf) or descend.
        ``before``/``after`` are TOC fragments that bracket each emitted
        chunk's content (chunk text = ``before + content + after``).
        As we descend, the prefix grows with section headings already
        passed and the suffix shrinks correspondingly.
        """
        if not node.text:
            return []
        if node.kind == "section":
            heading_line = f"{'#' * node.level} {node.heading or ''}"
            before_self = _toc_join(before, heading_line)
        else:
            before_self = before
        if len(node.text) <= self.chunk_chars:
            return [
                self._make_chunk(
                    before_self,
                    node.text,
                    after,
                    node.start_line,
                    node.end_line,
                    path,
                ),
            ]
        if node.kind == "body":
            return self._split_leaf(node, before, after, path, renderer)
        after_inside = _toc_join(node.desc_toc, after)
        sub_tocs = [_subtree_toc(c) for c in node.children if c.kind == "section"]
        chunks: list[FileChunk] = []
        accumulated = before_self
        sec_idx = 0
        run: list[MdNode] = []
        for c in node.children:
            if c.kind == "section":
                if run:
                    chunks.extend(
                        self._chunk_body_run(
                            run,
                            before_self,
                            after_inside,
                            path,
                            renderer,
                        ),
                    )
                    run = []
                remaining = "\n\n".join(sub_tocs[sec_idx + 1 :])
                chunks.extend(
                    self._chunk_node(
                        c,
                        accumulated,
                        _toc_join(remaining, after),
                        path,
                        renderer,
                    ),
                )
                accumulated = _toc_join(accumulated, sub_tocs[sec_idx])
                sec_idx += 1
            else:
                run.append(c)
        if run:
            chunks.extend(
                self._chunk_body_run(
                    run,
                    before_self,
                    after_inside,
                    path,
                    renderer,
                ),
            )
        return chunks

    def _chunk_body_run(
        self,
        run: list[MdNode],
        before: str,
        after: str,
        path: str,
        renderer,
    ) -> list[FileChunk]:
        """Greedy-pack consecutive body siblings under the same TOC slot.
        No ``[Part X/N]`` markers — distinct blocks, not a leaf split.
        Oversized single body recurses to ``_split_leaf``."""
        composite_size = sum(len(b.text) for b in run) + 2 * max(0, len(run) - 1)
        if composite_size <= self.chunk_chars:
            return [
                self._make_chunk(
                    before,
                    "\n\n".join(b.text for b in run),
                    after,
                    run[0].start_line,
                    run[-1].end_line,
                    path,
                ),
            ]

        chunks: list[FileChunk] = []
        bucket: list[MdNode] = []
        bucket_chars = 0

        def flush() -> None:
            nonlocal bucket, bucket_chars
            if not bucket:
                return
            chunks.append(
                self._make_chunk(
                    before,
                    "\n\n".join(b.text for b in bucket),
                    after,
                    bucket[0].start_line,
                    bucket[-1].end_line,
                    path,
                ),
            )
            bucket = []
            bucket_chars = 0

        for body in run:
            if len(body.text) > self.chunk_chars:
                flush()
                chunks.extend(self._split_leaf(body, before, after, path, renderer))
                continue
            sep = 2 if bucket else 0
            if bucket_chars + sep + len(body.text) > self.chunk_chars:
                flush()
                sep = 0
            bucket.append(body)
            bucket_chars += sep + len(body.text)
        flush()
        return chunks

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
        base = body.start_line + 2

        def line_of(i: int) -> int:
            return rows[i].line_number if i < len(rows) and rows[i].line_number else base + i

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
        for it in items:
            text = renderer.render(it).rstrip("\n")
            if not text:
                continue
            line = it.line_number or body.start_line
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
