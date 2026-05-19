"""Tests for LinkedFileParser (markdown parser + wikilink extraction)."""

# pylint: disable=protected-access

import asyncio
import os
import tempfile

from reme4.components.file_graph import LocalFileGraph
from reme4.components.file_parser import LinkedFileParser
from reme4.schema import FileNode


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
    """Drop a markdown file under tmpdir, return its path."""
    path = os.path.join(tmpdir, name)
    if "/" in name:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    return path


async def _make_graph(*nodes: FileNode) -> LocalFileGraph:
    """Build a started LocalFileGraph seeded with the given nodes."""
    graph = LocalFileGraph()
    await graph.start()
    if nodes:
        await graph.upsert_nodes(list(nodes))
    return graph


def test_parse_empty_file():
    """An empty .md → FileNode, no chunks, no links."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            path = _write_md(tmp, "empty.md", "")
            parser = LinkedFileParser()
            node, chunks = await parser.parse(path)
            assert chunks == []
            assert node.links == []
            assert node.chunk_ids == []
        print("✓ test_parse_empty_file passed")

    asyncio.run(run())


def test_parse_frontmatter_only():
    """Front-matter without body → FileNode with metadata, no chunks/links."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            path = _write_md(tmp, "fm.md", "---\ntitle: Demo\ntags: [a, b]\n---\n")
            parser = LinkedFileParser()
            node, chunks = await parser.parse(path)
            assert chunks == []
            assert node.front_matter.title == "Demo"
            assert list(node.front_matter.tags or []) == ["a", "b"]
        print("✓ test_parse_frontmatter_only passed")

    asyncio.run(run())


def test_parse_small_body_one_chunk():
    """A body shorter than chunk_chars produces exactly one chunk that contains the body."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            path = _write_md(tmp, "small.md", "# Hello\n\nworld")
            parser = LinkedFileParser(chunk_chars=2000)
            _, chunks = await parser.parse(path)
            assert len(chunks) == 1
            assert "world" in chunks[0].text
            assert chunks[0].start_line >= 1
            assert chunks[0].end_line >= chunks[0].start_line
        print("✓ test_parse_small_body_one_chunk passed")

    asyncio.run(run())


def test_parse_oversized_body_splits():
    """A body that exceeds chunk_chars produces multiple chunks."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            # 30 paragraphs of 100 chars each ≈ 3000 chars body
            body = "# Big\n\n" + "\n\n".join("p" * 100 for _ in range(30))
            path = _write_md(tmp, "big.md", body)
            parser = LinkedFileParser(chunk_chars=500, embed_toc=False)
            _, chunks = await parser.parse(path)
            assert len(chunks) > 1
        print("✓ test_parse_oversized_body_splits passed")

    asyncio.run(run())


def test_parse_chunk_ids_match_node_chunk_ids():
    """FileNode.chunk_ids should match the ids of the chunks returned."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            path = _write_md(tmp, "ids.md", "# X\n\nbody")
            parser = LinkedFileParser()
            node, chunks = await parser.parse(path)
            assert node.chunk_ids == [c.id for c in chunks]
        print("✓ test_parse_chunk_ids_match_node_chunk_ids passed")

    asyncio.run(run())


def test_parse_links_empty_when_no_graph():
    """Without an app_context / graph, links stay empty even if the body has wikilinks."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            path = _write_md(tmp, "x.md", "see [[Alice]] and [[Bob]]")
            parser = LinkedFileParser()
            node, _ = await parser.parse(path)
            assert node.links == []
        print("✓ test_parse_links_empty_when_no_graph passed")

    asyncio.run(run())


def test_parse_links_resolved_via_graph():
    """With a graph that knows the targets, wikilinks become FileLinks with resolved target_path."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            graph = await _make_graph(
                FileNode(path="topics/Alice.md", st_mtime=0.0),
                FileNode(path="topics/Bob.md", st_mtime=0.0),
            )
            path = _write_md(tmp, "note.md", "see [[Alice]] and [[Bob#sec]]")
            parser = LinkedFileParser()
            parser._resolve_file_graph = lambda: graph
            node, _ = await parser.parse(path)
            triples = {(link.target_path, link.target_anchor, link.predicate) for link in node.links}
            assert ("topics/Alice.md", None, None) in triples
            assert ("topics/Bob.md", "sec", None) in triples
            # source_path always equals the node's own path
            for link in node.links:
                assert link.source_path == node.path
            await graph.close()
        print("✓ test_parse_links_resolved_via_graph passed")

    asyncio.run(run())


def test_parse_links_predicate_inline_and_line():
    """Both `pred:: [[X]]` (line-level) and `[pred:: [[X]]]` (inline) propagate predicate."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            graph = await _make_graph(
                FileNode(path="A.md", st_mtime=0.0),
                FileNode(path="B.md", st_mtime=0.0),
            )
            body = "extends:: [[A]]\n\nsome [concerns:: [[B]]] inline\n"
            path = _write_md(tmp, "note.md", body)
            parser = LinkedFileParser()
            parser._resolve_file_graph = lambda: graph
            node, _ = await parser.parse(path)
            pairs = {(link.target_path, link.predicate) for link in node.links}
            assert ("A.md", "extends") in pairs
            assert ("B.md", "concerns") in pairs
            await graph.close()
        print("✓ test_parse_links_predicate_inline_and_line passed")

    asyncio.run(run())


def test_parse_links_short_path_ambiguity_expands():
    """A short link matching multiple nodes expands to one FileLink per candidate."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            graph = await _make_graph(
                FileNode(path="topics/Bob.md", st_mtime=0.0),
                FileNode(path="people/Bob.md", st_mtime=0.0),
            )
            path = _write_md(tmp, "note.md", "ref [[Bob]]")
            parser = LinkedFileParser()
            parser._resolve_file_graph = lambda: graph
            node, _ = await parser.parse(path)
            targets = sorted(link.target_path for link in node.links)
            assert targets == ["people/Bob.md", "topics/Bob.md"]
            await graph.close()
        print("✓ test_parse_links_short_path_ambiguity_expands passed")

    asyncio.run(run())


def test_parse_links_dangling_dropped():
    """Wikilink to a non-existent target is silently dropped (no graph node)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            graph = await _make_graph(FileNode(path="topics/Alice.md", st_mtime=0.0))
            path = _write_md(tmp, "note.md", "[[Alice]] and [[Ghost]]")
            parser = LinkedFileParser()
            parser._resolve_file_graph = lambda: graph
            node, _ = await parser.parse(path)
            targets = {link.target_path for link in node.links}
            assert targets == {"topics/Alice.md"}
            await graph.close()
        print("✓ test_parse_links_dangling_dropped passed")

    asyncio.run(run())


def test_parse_links_deduped():
    """Repeated wikilinks with the same (target, predicate, anchor) emit one FileLink."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            graph = await _make_graph(FileNode(path="A.md", st_mtime=0.0))
            path = _write_md(tmp, "note.md", "[[A]] again [[A]] and [[A]]")
            parser = LinkedFileParser()
            parser._resolve_file_graph = lambda: graph
            node, _ = await parser.parse(path)
            assert len([link for link in node.links if link.target_path == "A.md"]) == 1
            await graph.close()
        print("✓ test_parse_links_deduped passed")

    asyncio.run(run())


def test_parse_min_chunk_chars_clamped():
    """chunk_chars below 100 should be clamped to 100."""
    parser = LinkedFileParser(chunk_chars=10)
    assert parser.chunk_chars == 100
    print("✓ test_parse_min_chunk_chars_clamped passed")


def test_parse_embed_toc_prefixes_chunk_text():
    """When embed_toc=True, chunks emitted inside a section are prefixed by the heading."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            body = "# Top\n\n## Sub\n\nbody-content"
            path = _write_md(tmp, "toc.md", body)
            parser = LinkedFileParser(chunk_chars=200, embed_toc=True)
            _, chunks = await parser.parse(path)
            # Single small section fits; check that the heading appears in text.
            assert any("Top" in c.text for c in chunks)
        print("✓ test_parse_embed_toc_prefixes_chunk_text passed")

    asyncio.run(run())


if __name__ == "__main__":
    print("\n=== LinkedFileParser tests ===")
    test_parse_empty_file()
    test_parse_frontmatter_only()
    test_parse_small_body_one_chunk()
    test_parse_oversized_body_splits()
    test_parse_chunk_ids_match_node_chunk_ids()
    test_parse_links_empty_when_no_graph()
    test_parse_links_resolved_via_graph()
    test_parse_links_predicate_inline_and_line()
    test_parse_links_short_path_ambiguity_expands()
    test_parse_links_dangling_dropped()
    test_parse_links_deduped()
    test_parse_min_chunk_chars_clamped()
    test_parse_embed_toc_prefixes_chunk_text()
    print("\n所有测试通过!")
