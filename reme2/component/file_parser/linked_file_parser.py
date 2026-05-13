"""Markdown file parser — frontmatter + wikilink graph + AST tree chunks.

Chunking algorithm (full-doc skeleton + inlined content)
========================================================

Each chunk renders the **complete heading skeleton of the entire
document**, with this chunk's content inlined under the section that
owns it. Sections that don't own this chunk's content appear as bare
headings — every chunk gives the reader a complete map of the document
and shows exactly where its slice belongs.

Two phases:

1. **Build phase** — fold mistletoe's flat `Document.children` into a
   layered `MdNode` tree. A `section` node owns its heading + body
   blocks + child subsections, established by a heading-level stack.

2. **Chunk phase** — recursive `chunk(node, parent_section)`:

       if len(content) <= chunk_chars:
           emit one chunk (TOC wrapped on top, additive to size) and return
       if node is section/root:
           walk children — body siblings pack as a run inside the
           current section's TOC slot, subsections recurse
       else (leaf body):
           split by internal structure (List items / Table rows /
           code lines / paragraph lines)

Example — doc with sections A, B (containing B1), C — chunking content
of B1 produces a chunk like:

       # Doc
       ## A
       ## B
       ### B1

       <chunk content>

       ## C

The current section (here `### B1`) is the **owner**: its slot holds
the chunk's body. All other sections appear as bare headings.

**Budget rule**: ``chunk_chars`` constrains the **content** only —
the TOC skeleton is added as a free prefix on top of every chunk and
does NOT count toward the budget. This keeps content sizing predictable
even when a doc has a large heading outline.

**Toggle**: ``embed_toc=False`` disables the TOC wrap entirely; chunks
become plain content with no document-level navigation prefix. The
chunking decisions themselves are unchanged — only the final emitted
text differs.

Section integrity: a node is split only when it cannot fit as a whole.
Block-internal structure (code lines, table rows, list items) is
respected — splits land on those boundaries, never inside.

**Part markers**: when a single leaf block (table / code fence / list /
paragraph) is too large to fit and gets split into N > 1 pieces, each
piece is annotated with a ``[Part X/N]\\n\\n`` prefix so readers know
they're seeing a fragment. Single-piece outputs are unmarked.

Chunk identity: `hash_text(path::start::end::text)` — content-deterministic
so the file_store's hash-diff cache hits across re-parses of unchanged
sections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import frontmatter
from mistletoe.block_token import Document
from mistletoe.markdown_renderer import MarkdownRenderer

from .base_file_parser import BaseFileParser
from ..component_registry import R
from ...enumeration import FileSuffixEnum
from ...schema import FileChunk, FileEdge, FileNode
from ...utils import hash_text


# -- Helpers --------------------------------------------------------------


def _kind(node) -> str:
    return type(node).__name__


def _is_heading(node) -> bool:
    return _kind(node) in ("Heading", "SetextHeading")


def _line_count(text: str) -> int:
    return len(text.split("\n")) if text else 0


def _heading_text(node, renderer: MarkdownRenderer) -> str:
    """Heading text without `#` markers (for outline)."""
    rendered = renderer.render(node).rstrip("\n")
    if rendered.startswith("#"):
        return rendered.lstrip("#").strip()
    return rendered.split("\n", 1)[0].strip()


# Reserved overhead for the worst-case "[Part NNN/NNN]\n\n" prefix added
# to leaf-block split pieces. Reserved upfront in budgets so the prefix
# fits even for the chunk that just barely passed the size check.
_PART_MARKER_RESERVE = 18


# -- AST tree -------------------------------------------------------------


@dataclass
class MdNode:
    """Composed markdown AST node.

    Three synthesised kinds wrap mistletoe blocks into a layered tree:

      * ``root``    — sole top-level node; ``children`` are bodies and/or
                      sections; carries no heading.
      * ``section`` — synthesised from a heading + everything beneath it
                      until the next equal-or-shallower heading; children
                      are bodies and child sections.
      * ``body``    — wraps one mistletoe block (paragraph / list / table /
                      code / quote / html / etc.); ``block`` is the
                      original mistletoe node, ``text`` is its rendered
                      markdown.

    Ranges (`start_line`, `end_line`) span the full subtree so callers can
    record provenance on emitted chunks.
    """

    kind: str  # "root" | "section" | "body"
    heading: str | None = None
    level: int = 0
    children: list["MdNode"] = field(default_factory=list)
    block: Any = None
    text: str = ""
    start_line: int = 0
    end_line: int = 0


@R.register("md")
class LinkedFileParser(BaseFileParser):
    """Markdown parser: frontmatter + wikilink edges + full-skeleton chunks."""

    suffixes = [FileSuffixEnum.MD, FileSuffixEnum.MARKDOWN]

    def __init__(
        self,
        encoding: str = "utf-8",
        chunk_chars: int = 2000,
        embed_toc: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoding = encoding
        self.chunk_chars = max(100, chunk_chars)
        self.embed_toc = embed_toc

    async def parse(self, path: str | Path) -> tuple[FileNode, list[FileChunk]]:
        file_path = Path(path)
        raw = file_path.read_text(encoding=self.encoding)
        post = frontmatter.loads(raw)
        stat = file_path.stat()
        absolute_path = str(file_path.absolute())

        edges = self._dedup_edges(FileEdge.from_text(post.content))
        chunks = self._chunk(post.content, absolute_path)

        node = FileNode(
            path=absolute_path,
            st_mtime=stat.st_mtime,
            edges=edges,
            **dict(post.metadata),
        )
        return node, chunks

    @staticmethod
    def _dedup_edges(edges: list[FileEdge]) -> list[FileEdge]:
        seen: set[tuple] = set()
        out: list[FileEdge] = []
        for e in edges:
            key = (e.link, e.predicate)
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
        return out

    # -- Chunker entry ---------------------------------------------------

    def _chunk(self, text: str, path: str) -> list[FileChunk]:
        if not text or not text.strip():
            return []
        out: list[FileChunk] = []
        with MarkdownRenderer() as renderer:
            doc = Document(text)
            tree = self._build_tree(doc, renderer)
            self._chunk_tree(tree, tree, tree, renderer, path, out)
        return out

    # -- Build phase: mistletoe doc → MdNode tree -------------------------

    def _build_tree(self, doc, renderer: MarkdownRenderer) -> MdNode:
        """Fold mistletoe's flat children into a section tree.

        Algorithm: walk children with a stack of open sections. Each
        heading pops sections of equal-or-deeper level and pushes a new
        section. Non-heading blocks attach as ``body`` children to the
        current section (or to root before the first heading).
        """
        root = MdNode(kind="root", level=0, start_line=1, end_line=1)
        stack: list[MdNode] = [root]

        for child in doc.children or []:
            kind = _kind(child)
            if kind == "BlankLine":
                continue
            if _is_heading(child):
                level = max(1, getattr(child, "level", 1))
                # Close any sections of equal-or-greater level.
                while len(stack) > 1 and stack[-1].level >= level:
                    stack.pop()
                sec = MdNode(
                    kind="section",
                    heading=_heading_text(child, renderer),
                    level=level,
                    start_line=child.line_number or stack[-1].start_line,
                )
                stack[-1].children.append(sec)
                stack.append(sec)
            else:
                rendered = renderer.render(child).rstrip("\n")
                if not rendered:
                    continue
                start = child.line_number or stack[-1].start_line
                body = MdNode(
                    kind="body",
                    block=child,
                    text=rendered,
                    start_line=start,
                    end_line=start + _line_count(rendered) - 1,
                )
                stack[-1].children.append(body)

        # Propagate end_line bottom-up.
        def _close(n: MdNode) -> None:
            if not n.children:
                if n.end_line < n.start_line:
                    n.end_line = n.start_line
                return
            for c in n.children:
                _close(c)
            n.end_line = max(c.end_line for c in n.children)
            n.start_line = min(n.start_line or n.children[0].start_line, n.children[0].start_line)

        _close(root)
        return root

    # -- Chunk phase: recursive ------------------------------------------

    def _chunk_tree(
        self,
        tree: MdNode,
        node: MdNode,
        parent_section: MdNode,
        renderer: MarkdownRenderer,
        file_path: str,
        out: list[FileChunk],
    ) -> None:
        """Try the whole subtree first; on overflow descend into children.

        ``tree`` is the whole-document root used to render the full TOC
        skeleton (when ``embed_toc`` is on). ``parent_section`` is the
        nearest enclosing section/root — for body nodes it's the slot
        owner; for sections it becomes the slot owner when the section
        itself fits whole. The ``chunk_chars`` budget only constrains
        ``content`` — TOC overhead is excluded.
        """
        if node.kind == "body":
            owner = parent_section
            content = node.text
            owns_subtree = False
        else:  # root or section
            owner = node
            content = self._render_node_content(node)
            owns_subtree = True

        if not content.strip():
            return

        if len(content) <= self.chunk_chars:
            full = self._finalize(tree, owner, content, owns_subtree)
            self._emit(full, node.start_line, node.end_line, file_path, out)
            return

        if node.kind in ("root", "section"):
            self._chunk_children(tree, node, renderer, file_path, out)
        else:
            self._split_leaf(tree, node, parent_section, renderer, file_path, out)

    def _chunk_children(
        self,
        tree: MdNode,
        parent: MdNode,
        renderer: MarkdownRenderer,
        file_path: str,
        out: list[FileChunk],
    ) -> None:
        """Walk a section's children: body run packs together, each
        subsection recurses. Body runs use ``parent`` as TOC owner."""
        run: list[MdNode] = []

        def flush_run() -> None:
            nonlocal run
            if run:
                self._chunk_body_run(tree, run, parent, renderer, file_path, out)
                run = []

        for c in parent.children:
            if c.kind == "section":
                flush_run()
                self._chunk_tree(tree, c, parent, renderer, file_path, out)
            else:
                run.append(c)
        flush_run()

    def _chunk_body_run(
        self,
        tree: MdNode,
        run: list[MdNode],
        owner: MdNode,
        renderer: MarkdownRenderer,
        file_path: str,
        out: list[FileChunk],
    ) -> None:
        """A run of consecutive body siblings sharing ``owner``'s slot.

        Try whole run first; on overflow greedy-pack into ``chunk_chars``
        (content-only budget); an oversized single body recurses through
        ``_split_leaf``.
        """
        composite = "\n\n".join(b.text for b in run)
        if len(composite) <= self.chunk_chars:
            full = self._finalize(tree, owner, composite, owns_subtree=False)
            self._emit(full, run[0].start_line, run[-1].end_line, file_path, out)
            return

        budget = self.chunk_chars
        bucket: list[MdNode] = []
        bucket_chars = 0

        def flush() -> None:
            nonlocal bucket, bucket_chars
            if not bucket:
                return
            text = "\n\n".join(b.text for b in bucket)
            piece = self._finalize(tree, owner, text, owns_subtree=False)
            self._emit(piece, bucket[0].start_line, bucket[-1].end_line, file_path, out)
            bucket = []
            bucket_chars = 0

        for body in run:
            if len(body.text) > budget:
                flush()
                self._split_leaf(tree, body, owner, renderer, file_path, out)
                continue
            sep = 2 if bucket else 0  # "\n\n"
            if bucket and bucket_chars + sep + len(body.text) > budget:
                flush()
                sep = 0
            bucket.append(body)
            bucket_chars += sep + len(body.text)
        flush()

    # -- Leaf-internal splitters -----------------------------------------

    def _split_leaf(
        self,
        tree: MdNode,
        body: MdNode,
        owner: MdNode,
        renderer: MarkdownRenderer,
        file_path: str,
        out: list[FileChunk],
    ) -> None:
        kind = _kind(body.block) if body.block is not None else "Paragraph"
        if kind == "Table":
            self._split_table(tree, body, owner, file_path, out)
        elif kind == "CodeFence":
            self._split_code(tree, body, owner, file_path, out)
        elif kind == "List":
            self._split_list(tree, body, owner, renderer, file_path, out)
        else:
            self._split_lines(tree, body, owner, file_path, out)

    def _split_table(
        self,
        tree: MdNode,
        body: MdNode,
        owner: MdNode,
        file_path: str,
        out: list[FileChunk],
    ) -> None:
        """Split a table by data rows; repeat header + separator per piece."""
        table = body.block
        rendered = body.text
        all_lines = rendered.split("\n")
        header = "\n".join(all_lines[:2])
        data_lines = all_lines[2:]
        rows = [r for r in (table.children or []) if _kind(r) == "TableRow"]
        start = body.start_line
        if len(rows) == len(data_lines):
            units = [
                (data_lines[i],
                 rows[i].line_number or (start + 2 + i),
                 rows[i].line_number or (start + 2 + i))
                for i in range(len(data_lines))
            ]
        else:
            units = [
                (data_lines[i], start + 2 + i, start + 2 + i)
                for i in range(len(data_lines))
            ]
        self._emit_packed(
            tree, owner, units, joiner="\n",
            wrap=f"{header}\n{{inner}}", file_path=file_path, out=out,
        )

    def _split_code(
        self,
        tree: MdNode,
        body: MdNode,
        owner: MdNode,
        file_path: str,
        out: list[FileChunk],
    ) -> None:
        """Split a code fence by body lines; repeat opener / closer per piece."""
        code = body.block
        indent = " " * (code.indentation or 0)
        info = code.info_string or ""
        opener = f"{indent}{code.delimiter}{info}"
        closer = f"{indent}{code.delimiter}"
        wrap = f"{opener}\n{{inner}}\n{closer}"

        raw = (code.children[0].content if code.children else "").rstrip("\n")
        if not raw:
            return
        start = body.start_line + 1
        units = [
            (indent + ln, start + i, start + i)
            for i, ln in enumerate(raw.split("\n"))
        ]
        self._emit_packed(
            tree, owner, units, joiner="\n", wrap=wrap,
            file_path=file_path, out=out, allow_empty=True,
        )

    def _split_list(
        self,
        tree: MdNode,
        body: MdNode,
        owner: MdNode,
        renderer: MarkdownRenderer,
        file_path: str,
        out: list[FileChunk],
    ) -> None:
        """Split a list by items; greedy-pack items.

        Multi-part splits annotate each piece with ``[Part X/N]``
        (see ``_emit_parts``). An item that exceeds the budget on its
        own is emitted as one part (continuity wins over hard cap).
        """
        items = [c for c in (body.block.children or []) if _kind(c) == "ListItem"]
        if not items:
            self._split_lines(tree, body, owner, file_path, out)
            return

        budget = max(64, self.chunk_chars - _PART_MARKER_RESERVE)
        rendered_items = [
            (renderer.render(it).rstrip("\n"), it.line_number or body.start_line)
            for it in items
        ]

        parts: list[tuple[str, int, int]] = []
        bucket: list[tuple[str, int, int]] = []
        bucket_chars = 0

        def flush() -> None:
            nonlocal bucket, bucket_chars
            if not bucket:
                return
            text = "\n".join(t for t, _, _ in bucket)
            parts.append((text, bucket[0][1], bucket[-1][2]))
            bucket = []
            bucket_chars = 0

        for text, line in rendered_items:
            if not text:
                continue
            end = line + _line_count(text) - 1
            if len(text) > budget:
                # Oversized item: emit alone (overflow accepted).
                flush()
                parts.append((text, line, end))
                continue
            sep = 1 if bucket else 0  # "\n"
            if bucket and bucket_chars + sep + len(text) > budget:
                flush()
                sep = 0
            bucket.append((text, line, end))
            bucket_chars += sep + len(text)
        flush()

        self._emit_parts(tree, owner, parts, wrap="{inner}", file_path=file_path, out=out)

    def _split_lines(
        self,
        tree: MdNode,
        body: MdNode,
        owner: MdNode,
        file_path: str,
        out: list[FileChunk],
    ) -> None:
        """Last-resort split: line-greedy. Used for paragraphs / quotes /
        html / oversized list items."""
        start = body.start_line
        units = [
            (line, start + i, start + i)
            for i, line in enumerate(body.text.split("\n"))
        ]
        self._emit_packed(
            tree, owner, units, joiner="\n", wrap="{inner}",
            file_path=file_path, out=out,
        )

    def _emit_packed(
        self,
        tree: MdNode,
        owner: MdNode,
        units: list[tuple[str, int, int]],
        joiner: str,
        wrap: str,
        file_path: str,
        out: list[FileChunk],
        allow_empty: bool = False,
    ) -> None:
        """Greedy-pack units into the wrap envelope; emit each bucket
        as a chunk via ``_emit_parts`` (which adds ``[Part X/N]`` when
        more than one piece results).

        The envelope (e.g. table header, code fence) IS counted against
        ``chunk_chars`` because it's part of the chunk's content. The
        TOC skeleton, when ``embed_toc`` is on, is added as a free
        prefix downstream.

        A unit larger than the inner budget is emitted alone (continuity
        wins over hard cap — readers can still see the overflowing line).
        """
        envelope = len(wrap.replace("{inner}", ""))
        inner_budget = max(64, self.chunk_chars - envelope - _PART_MARKER_RESERVE)
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
            if bucket and bucket_chars + sep + len(text) > inner_budget:
                flush()
                sep = 0
            bucket.append((text, s, e))
            bucket_chars += sep + len(text)
        flush()

        self._emit_parts(tree, owner, parts, wrap=wrap, file_path=file_path, out=out)

    def _emit_parts(
        self,
        tree: MdNode,
        owner: MdNode,
        parts: list[tuple[str, int, int]],
        wrap: str,
        file_path: str,
        out: list[FileChunk],
    ) -> None:
        """Emit a list of leaf-block split parts.

        Each part is a ``(inner_text, start_line, end_line)`` triple.
        ``wrap`` is a format string with ``{inner}`` substituted per
        piece (e.g. ``"| header |\\n{inner}"`` for tables; ``"{inner}"``
        for plain line splits).

        When ``len(parts) > 1`` each piece is prefixed with
        ``[Part X/N]\\n\\n`` so readers know it's a fragment of a
        larger leaf block. A single part emits with no marker.
        """
        total = len(parts)
        for idx, (inner, s, e) in enumerate(parts, 1):
            piece = wrap.replace("{inner}", inner)
            if total > 1:
                piece = f"[Part {idx}/{total}]\n\n{piece}"
            full = self._finalize(tree, owner, piece, owns_subtree=False)
            self._emit(full, s, e, file_path, out)

    # -- Render helpers ---------------------------------------------------

    def _finalize(
        self,
        tree: MdNode,
        owner: MdNode,
        content: str,
        owns_subtree: bool,
    ) -> str:
        """Produce the final chunk text from raw ``content``.

        When ``embed_toc`` is on (default), wrap content with the full
        document heading skeleton (see ``_render_with_full_toc``) so the
        chunk shows where its slice belongs in the doc. When off, return
        ``content`` unchanged — the chunk is just its own text.

        ``chunk_chars`` is checked against ``content`` upstream of this
        call; the TOC skeleton is *additive* to the chunk text and does
        not consume the budget. Callers wanting to know the final chunk
        length must call ``len(self._finalize(...))`` themselves.
        """
        if not self.embed_toc:
            return content
        return self._render_with_full_toc(tree, owner, content, owns_subtree)

    @classmethod
    def _render_with_full_toc(
        cls,
        tree: MdNode,
        owner: MdNode,
        content: str,
        owns_subtree: bool,
    ) -> str:
        """Render the full document heading skeleton with ``content``
        inlined under ``owner``'s heading.

        Walks the entire tree. Every section emits its heading; only the
        ``owner`` slot also emits ``content``. If ``owns_subtree`` is
        True, recursion stops at ``owner`` (its subsection headings are
        assumed already present in ``content``); otherwise traversal
        continues so descendant headings still appear in the TOC.

        Bodies are never emitted by this walk — they're brought in only
        via ``content``. ``owner`` may be the root node, in which case
        the content sits before the first heading.
        """
        lines: list[str] = []

        def walk(node: MdNode) -> None:
            if node.kind == "section" and node.heading is not None:
                lines.append(f"{'#' * max(1, node.level)} {node.heading}")
            if node is owner:
                if content:
                    lines.append(content)
                if owns_subtree:
                    return
            for c in node.children:
                if c.kind == "section":
                    walk(c)

        walk(tree)
        return "\n\n".join(p for p in lines if p)

    @classmethod
    def _render_node_content(cls, node: MdNode) -> str:
        """Render a node's content BENEATH its own heading.

        The node's own heading is NOT included — when `_render_with_full_toc`
        emits the section's TOC entry, the slot it appends ``content`` to
        already sits below that heading. Subsection headings ARE included
        because they're deeper than the focused node and would otherwise
        be swallowed when ``owns_subtree=True``.
        """
        if node.kind == "body":
            return node.text
        parts: list[str] = []
        for c in node.children:
            if c.kind == "section":
                sub_heading = f"{'#' * max(1, c.level)} {c.heading or ''}"
                inner = cls._render_node_content(c)
                parts.append(sub_heading + ("\n\n" + inner if inner else ""))
            else:
                if c.text:
                    parts.append(c.text)
        return "\n\n".join(parts)

    # -- Emit -------------------------------------------------------------

    @staticmethod
    def _emit(
        text: str, start_line: int, end_line: int, path: str, out: list[FileChunk],
    ) -> None:
        chunk_id = hash_text(f"{path}::{start_line}::{end_line}::{text}")
        out.append(FileChunk(
            id=chunk_id,
            path=path,
            start_line=start_line,
            end_line=end_line,
            text=text,
        ))


# -- CLI: parse a markdown file and print chunks + edges ------------------


def _main() -> None:
    """Parse a markdown file and print its edges + chunks for inspection.

    Usage:
        python -m reme2.component.file_parser.linked_file_parser <path> [--chunk-chars N]
    """
    import argparse
    import asyncio

    ap = argparse.ArgumentParser(
        description="Parse a markdown file with LinkedFileParser and dump chunks + edges.",
    )
    ap.add_argument("path", help="Path to a markdown file.")
    ap.add_argument(
        "--chunk-chars", type=int, default=2000,
        help="Max characters per chunk content (default: 2000). "
             "Excludes TOC skeleton when embed_toc is on.",
    )
    ap.add_argument(
        "--no-toc", action="store_true",
        help="Disable the full-doc TOC skeleton wrap; chunks become plain content.",
    )
    ap.add_argument(
        "--show-edges", action="store_true",
        help="Print extracted FileEdges before chunks.",
    )
    ap.add_argument(
        "--preview", type=int, default=0,
        help="Truncate each chunk to N chars in output (0 = full text).",
    )
    args = ap.parse_args()

    parser = LinkedFileParser(
        chunk_chars=args.chunk_chars,
        embed_toc=not args.no_toc,
    )
    node, chunks = asyncio.run(parser.parse(args.path))

    print(f"file:        {node.path}")
    print(f"chunk_chars: {args.chunk_chars}")
    print(f"embed_toc:   {parser.embed_toc}")
    print(f"chunks:      {len(chunks)}")
    print(f"chars total: {sum(len(c.text) for c in chunks)}")
    if chunks:
        sizes = [len(c.text) for c in chunks]
        print(f"chars min/avg/max: {min(sizes)} / {sum(sizes)//len(sizes)} / {max(sizes)}")
    if args.show_edges:
        print(f"\nedges ({len(node.edges)}):")
        for e in node.edges:
            print(
                f"  → {e.link}"
                + (f"  predicate={e.predicate}" if e.predicate else "")
                + (f"  anchor={e.anchor}" if e.anchor else "")
            )

    for i, c in enumerate(chunks):
        print(f"\n{'=' * 72}")
        print(f"chunk {i}  lines {c.start_line}-{c.end_line}  {len(c.text)} chars")
        print("-" * 72)
        text = c.text if args.preview <= 0 else c.text[: args.preview]
        print(text)
        if args.preview > 0 and len(c.text) > args.preview:
            print(f"... ({len(c.text) - args.preview} more chars truncated)")


if __name__ == "__main__":
    _main()

