"""Tests for MarkdownFileChunker (markdown parser + wikilink extraction).

Wikilink convention here is strict: targets are taken literally, no
short-form basename search, no implicit ``.md``, no folder-note
expansion. ``lint:dangling`` handles validation; the parser is just
a markdown-to-FileNode transformer.
"""

# pylint: disable=protected-access

import asyncio
import os
import tempfile

from reme.components.file_chunker import MarkdownFileChunker


class temp_chdir:
    """Context manager to temporarily chdir into a path and restore on exit."""

    def __init__(self, path):
        self.path = path
        self.old = None

    def __enter__(self):
        self.old = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, *exc):
        os.chdir(self.old)


def _write_md(tmpdir: str, name: str, body: str) -> str:
    """Drop a markdown file under tmpdir, return its relative path (matches cwd)."""
    if "/" in name:
        os.makedirs(os.path.join(tmpdir, os.path.dirname(name)), exist_ok=True)
    with open(os.path.join(tmpdir, name), "w", encoding="utf-8") as f:
        f.write(body)
    return name


def test_parse_empty_file():
    """An empty .md → FileNode, no chunks, no links."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            path = _write_md(tmp, "x.md", "")
            chunker = MarkdownFileChunker()
            node, chunks = await chunker.chunk(path)
            assert node.path == "x.md"
            assert chunks == []
            assert node.links == []
        print("✓ test_parse_empty_file passed")

    asyncio.run(run())


def test_parse_frontmatter_only():
    """A file with only frontmatter (no body) → no chunks, no links."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            path = _write_md(tmp, "fm.md", "---\nname: t\n---\n")
            chunker = MarkdownFileChunker()
            node, chunks = await chunker.chunk(path)
            assert node.front_matter.name == "t"
            assert chunks == []
            assert node.links == []
        print("✓ test_parse_frontmatter_only passed")

    asyncio.run(run())


def test_parse_small_body_one_chunk():
    """A body shorter than chunk_chars produces exactly one chunk that contains the body."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            body = "# Hello\n\nthis is a small body."
            path = _write_md(tmp, "small.md", body)
            chunker = MarkdownFileChunker(chunk_chars=500)
            node, chunks = await chunker.chunk(path)
            assert len(chunks) == 1
            assert "this is a small body" in chunks[0].text
            assert node.chunk_ids == [chunks[0].id]
        print("✓ test_parse_small_body_one_chunk passed")

    asyncio.run(run())


def test_parse_oversized_body_splits():
    """A body exceeding chunk_chars triggers multiple chunks."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            paras = "\n\n".join(f"paragraph {i} with some content text here." for i in range(50))
            body = "# H\n\n" + paras
            path = _write_md(tmp, "big.md", body)
            chunker = MarkdownFileChunker(chunk_chars=200)
            _, chunks = await chunker.chunk(path)
            assert len(chunks) > 1
        print("✓ test_parse_oversized_body_splits passed")

    asyncio.run(run())


def test_parse_chunk_ids_match_node_chunk_ids():
    """node.chunk_ids is the ordered list of chunk hashes."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            paras = "\n\n".join(f"para {i} body content here." for i in range(40))
            body = "# H\n\n" + paras
            path = _write_md(tmp, "p.md", body)
            chunker = MarkdownFileChunker(chunk_chars=200)
            node, chunks = await chunker.chunk(path)
            assert node.chunk_ids == [c.id for c in chunks]
        print("✓ test_parse_chunk_ids_match_node_chunk_ids passed")

    asyncio.run(run())


def test_parse_links_literal_targets():
    """Wikilink targets are taken verbatim — full path → FileLink.target_path."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            body = "see [[topics/Alice.md]] and [[topics/Bob.md#sec]]"
            path = _write_md(tmp, "note.md", body)
            chunker = MarkdownFileChunker()
            node, _ = await chunker.chunk(path)
            triples = {(link.target_path, link.target_anchor, link.predicate) for link in node.links}
            assert ("topics/Alice.md", None, None) in triples
            assert ("topics/Bob.md", "sec", None) in triples
            # source_path always equals the node's own path
            for link in node.links:
                assert link.source_path == node.path
        print("✓ test_parse_links_literal_targets passed")

    asyncio.run(run())


def test_parse_links_short_and_no_ext_kept_literally():
    """Short and no-ext forms are NOT resolved — they're stored as-is.

    The parser does no resolution; whether the target exists is a
    ``lint:dangling`` concern. ``[[Alice]]`` becomes
    ``target_path='Alice'`` and will be flagged dangling unless a node
    with literal path 'Alice' actually exists.
    """

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            body = "see [[Alice]] and [[topics/Alice]] but also [[topics/Alice.md]]"
            path = _write_md(tmp, "note.md", body)
            chunker = MarkdownFileChunker()
            node, _ = await chunker.chunk(path)
            targets = {link.target_path for link in node.links}
            assert targets == {"Alice", "topics/Alice", "topics/Alice.md"}
        print("✓ test_parse_links_short_and_no_ext_kept_literally passed")

    asyncio.run(run())


def test_parse_links_predicate_inline_and_line():
    """Both `pred:: [[X]]` (line-level) and `[pred:: [[X]]]` (inline) propagate predicate."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            body = "extends:: [[A.md]]\n\nsome [concerns:: [[B.md]]] inline\n"
            path = _write_md(tmp, "note.md", body)
            chunker = MarkdownFileChunker()
            node, _ = await chunker.chunk(path)
            pairs = {(link.target_path, link.predicate) for link in node.links}
            assert ("A.md", "extends") in pairs
            assert ("B.md", "concerns") in pairs
        print("✓ test_parse_links_predicate_inline_and_line passed")

    asyncio.run(run())


def test_parse_links_deduped():
    """Repeated wikilinks with the same (target, predicate, anchor) emit one FileLink."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            body = "[[A.md]] again [[A.md]] and [[A.md]]"
            path = _write_md(tmp, "note.md", body)
            chunker = MarkdownFileChunker()
            node, _ = await chunker.chunk(path)
            assert len([link for link in node.links if link.target_path == "A.md"]) == 1
        print("✓ test_parse_links_deduped passed")

    asyncio.run(run())


def test_parse_min_chunk_chars_clamped():
    """chunk_chars below 100 should be clamped to 100."""
    chunker = MarkdownFileChunker(chunk_chars=10)
    assert chunker.chunk_chars == 100
    print("✓ test_parse_min_chunk_chars_clamped passed")


def test_parse_embed_toc_prefixes_chunk_text():
    """When embed_toc=True, chunks emitted inside a section are prefixed by the heading."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            body = "# Top\n\n## Sub\n\nbody-content"
            path = _write_md(tmp, "toc.md", body)
            chunker = MarkdownFileChunker(chunk_chars=200, embed_toc=True)
            _, chunks = await chunker.chunk(path)
            # Single small section fits; check that the heading appears in text.
            assert any("Top" in c.text for c in chunks)
        print("✓ test_parse_embed_toc_prefixes_chunk_text passed")

    asyncio.run(run())


def test_parse_frontmatter_preserves_original_line_numbers():
    """Chunk line ranges are 1-based and refer to the original file, including frontmatter."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            body = "---\nname: t\n---\n# H\nline 1\nline 2\n"
            path = _write_md(tmp, "front-lines.md", body)
            chunker = MarkdownFileChunker(chunk_chars=500)
            _, chunks = await chunker.chunk(path)
            assert len(chunks) == 1
            assert chunks[0].start_line == 4
            assert chunks[0].end_line == 6
        print("✓ test_parse_frontmatter_preserves_original_line_numbers passed")

    asyncio.run(run())


def test_parse_frontmatter_offsets_split_table_rows():
    """Split table row ranges include the YAML frontmatter line offset."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            rows = "".join(f"| {i} | {i} |\n" for i in range(12))
            body = "---\nname: t\n---\n| A | B |\n|---|---|\n" + rows
            path = _write_md(tmp, "front-table.md", body)
            chunker = MarkdownFileChunker(chunk_chars=100)
            _, chunks = await chunker.chunk(path)
            assert len(chunks) > 1
            assert chunks[0].start_line == 6
            assert chunks[0].end_line >= chunks[0].start_line
        print("✓ test_parse_frontmatter_offsets_split_table_rows passed")

    asyncio.run(run())


def test_parse_bad_frontmatter_does_not_abort_chunking():
    """Invalid YAML frontmatter is ignored while the markdown body still chunks."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            body = "---\nname: [\n---\n# H\nbody\n"
            path = _write_md(tmp, "bad-frontmatter.md", body)
            chunker = MarkdownFileChunker(chunk_chars=500)
            node, chunks = await chunker.chunk(path)
            assert node.front_matter.name == ""
            assert len(chunks) == 1
            assert chunks[0].start_line == 4
            assert "body" in chunks[0].text
        print("✓ test_parse_bad_frontmatter_does_not_abort_chunking passed")

    asyncio.run(run())


if __name__ == "__main__":
    print("\n=== MarkdownFileChunker tests ===")
    test_parse_empty_file()
    test_parse_frontmatter_only()
    test_parse_small_body_one_chunk()
    test_parse_oversized_body_splits()
    test_parse_chunk_ids_match_node_chunk_ids()
    test_parse_links_literal_targets()
    test_parse_links_short_and_no_ext_kept_literally()
    test_parse_links_predicate_inline_and_line()
    test_parse_links_deduped()
    test_parse_min_chunk_chars_clamped()
    test_parse_embed_toc_prefixes_chunk_text()
    test_parse_frontmatter_preserves_original_line_numbers()
    test_parse_frontmatter_offsets_split_table_rows()
    test_parse_bad_frontmatter_does_not_abort_chunking()
    print("\n所有测试通过!")
