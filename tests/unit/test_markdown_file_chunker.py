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
from unittest.mock import patch

from reme.components import ApplicationContext
from reme.components.file_chunker import DefaultFileChunker, MarkdownFileChunker
from reme.enumeration import ComponentEnum


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


def test_parse_frontmatter_metadata_is_opt_in():
    """Chunk metadata preserves the old empty default unless explicitly enabled."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            body = (
                "---\n"
                "name: locomo-event\n"
                "description: Jon lost his job\n"
                "conversation_date: 2023-01-19\n"
                "---\n"
                "Jon said he lost his job today.\n"
            )
            path = _write_md(tmp, "daily/2023-01-19/locomo-event.md", body)
            chunker = MarkdownFileChunker(chunk_chars=500)
            _, chunks = await chunker.chunk(path)

            assert len(chunks) == 1
            assert chunks[0].metadata == {}

            chunker = MarkdownFileChunker(chunk_chars=500, include_frontmatter_in_metadata=True)
            _, chunks = await chunker.chunk(path)

            assert len(chunks) == 1
            assert chunks[0].metadata == {
                "name": "locomo-event",
                "description": "Jon lost his job",
                "conversation_date": "2023-01-19",
            }
        print("✓ test_parse_frontmatter_metadata_is_opt_in passed")

    asyncio.run(run())


def test_parse_frontmatter_metadata_keys_allowlist():
    """``include_frontmatter_keys_in_metadata`` restricts copied keys to an allow-list."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            body = (
                "---\n"
                "name: locomo-event\n"
                "description: Jon lost his job\n"
                "conversation_date: 2023-01-19\n"
                "---\n"
                "Jon said he lost his job today.\n"
            )
            path = _write_md(tmp, "daily/2023-01-19/locomo-event.md", body)

            # Allow-list restricted to a single key.
            chunker = MarkdownFileChunker(
                chunk_chars=500,
                include_frontmatter_in_metadata=True,
                include_frontmatter_keys_in_metadata=["conversation_date"],
            )
            _, chunks = await chunker.chunk(path)
            assert len(chunks) == 1
            assert chunks[0].metadata == {"conversation_date": "2023-01-19"}

            # Allow-list with a key not present in frontmatter is a no-op for that key.
            chunker = MarkdownFileChunker(
                chunk_chars=500,
                include_frontmatter_in_metadata=True,
                include_frontmatter_keys_in_metadata=["conversation_date", "absent"],
            )
            _, chunks = await chunker.chunk(path)
            assert chunks[0].metadata == {"conversation_date": "2023-01-19"}

            # Empty allow-list (not None) keeps the legacy "all non-empty keys" behavior.
            chunker = MarkdownFileChunker(
                chunk_chars=500,
                include_frontmatter_in_metadata=True,
                include_frontmatter_keys_in_metadata=[],
            )
            _, chunks = await chunker.chunk(path)
            assert chunks[0].metadata == {
                "name": "locomo-event",
                "description": "Jon lost his job",
                "conversation_date": "2023-01-19",
            }

            # Allow-list is ignored when the master toggle is off (back-compat default).
            chunker = MarkdownFileChunker(
                chunk_chars=500,
                include_frontmatter_in_metadata=False,
                include_frontmatter_keys_in_metadata=["conversation_date"],
            )
            _, chunks = await chunker.chunk(path)
            assert chunks[0].metadata == {}
        print("✓ test_parse_frontmatter_metadata_keys_allowlist passed")

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


def test_parse_small_children_are_cached_without_recursive_calls():
    """An oversized parent greedily caches fitting children without recursing into them."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            sections = "\n\n".join(f"# Section-{i}\n\n{'x' * 30}" for i in range(6))
            path = _write_md(tmp, "small-children.md", sections)
            chunker = MarkdownFileChunker(chunk_chars=100, embed_toc=False)
            with patch.object(
                chunker,
                "_chunk_node",
                wraps=chunker._chunk_node,
            ) as chunk_node:
                _, chunks = await chunker.chunk(path)

            assert chunk_node.call_count == 1
            assert 1 < len(chunks) < 6
            assert all(len(chunk.text) <= chunker.chunk_chars for chunk in chunks)

    asyncio.run(run())


def test_parse_child_cache_flushes_at_exact_limit():
    """A cache reaching ``chunk_chars`` is finalized before the next child."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            path = _write_md(tmp, "exact-cache.md", f"# A\n\n{'x' * 95}\n\n# B\n\ny")
            chunker = MarkdownFileChunker(chunk_chars=100)
            _, chunks = await chunker.chunk(path)

            assert len(chunks) == 2
            assert len(chunks[0].text) == chunker.chunk_chars
            assert chunks[0].text.startswith("# A")
            assert chunks[1].text == "# B\n\ny"

    asyncio.run(run())


def test_parse_oversized_child_flushes_parent_cache():
    """Recursive child chunks do not merge across the parent cache boundary."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            leaves = "\n\n".join(f"## L{i}\n\n{'x' * 10}" for i in range(6))
            body = f"# A\n\na\n\n# Large\n\n{leaves}\n\n# C\n\nc"
            path = _write_md(tmp, "recursive-boundary.md", body)
            chunker = MarkdownFileChunker(chunk_chars=100, embed_toc=False)
            _, chunks = await chunker.chunk(path)

            assert len(chunks) == 4
            assert chunks[0].text == "# A\n\na"
            assert chunks[-2].text == f"## L5\n\n{'x' * 10}"
            assert chunks[-1].text == "# C\n\nc"

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


def test_parse_embed_toc_is_enabled_by_default():
    """The default adds bounded ancestor breadcrumbs without a full outline."""
    chunker = MarkdownFileChunker()
    assert chunker.embed_toc is True
    assert chunker.max_ast_sections == 100


def test_count_sections_ignores_fenced_headings_and_supports_setext():
    """The AST preflight counts real headings without parsing fenced examples."""
    content = "# Real\n\n```markdown\n# Fake\nFake too\n---\n```\n\nSetext\n===\n"
    assert MarkdownFileChunker._count_sections(content) == 2
    assert MarkdownFileChunker._count_sections(content, stop_after=1) == 2


def test_parse_excessive_sections_uses_plain_text_without_ast():
    """Too many sections preserve Markdown metadata while bypassing mistletoe."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            app_context = ApplicationContext(workspace_dir=tmp)
            default_chunker = DefaultFileChunker(
                name="default",
                app_context=app_context,
                chunk_byte_size=160,
                overlap_byte_size=20,
            )
            app_context.components[ComponentEnum.FILE_CHUNKER] = {"default": default_chunker}
            sections = "\n\n".join(f"# Section-{i}\n\n{'x' * 40} [[target.md]]" for i in range(4))
            body = f"---\nname: fallback\n---\n{sections}"
            _write_md(tmp, "fallback.md", body)
            path = os.path.join(tmp, "fallback.md")
            chunker = MarkdownFileChunker(
                app_context=app_context,
                chunk_chars=100,
                embed_toc=True,
                max_ast_sections=2,
                include_frontmatter_in_metadata=True,
            )
            await default_chunker.start()
            await chunker.start()
            try:
                with (
                    patch(
                        "mistletoe.block_token.Document",
                        side_effect=AssertionError("fallback must not construct an AST"),
                    ),
                    patch.object(
                        default_chunker,
                        "chunk_content",
                        wraps=default_chunker.chunk_content,
                    ) as chunk_content,
                ):
                    node, chunks = await chunker.chunk(path)
                assert chunker.default_chunker is default_chunker
                assert chunk_content.call_count == 1
            finally:
                await chunker.close()
                await default_chunker.close()

            assert len(chunks) > 1
            assert chunks[0].start_line == 4
            assert node.front_matter.name == "fallback"
            assert node.chunk_ids == [chunk.id for chunk in chunks]
            assert {(link.target_path, link.source_path) for link in node.links} == {
                ("target.md", "fallback.md"),
            }
            assert all(chunk.metadata == {"name": "fallback"} for chunk in chunks)
            for i in range(4):
                assert any(f"# Section-{i}" in chunk.text for chunk in chunks)

    asyncio.run(run())


def test_parse_section_limit_is_inclusive_for_ast():
    """A document at the configured section limit still takes the AST path."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            path = _write_md(tmp, "at-limit.md", "# A\n\na\n\n# B\n\nb")
            chunker = MarkdownFileChunker(max_ast_sections=2)
            with patch.object(chunker, "_build_tree", wraps=chunker._build_tree) as build_tree:
                _, chunks = await chunker.chunk(path)

            assert build_tree.call_count == 1
            assert len(chunks) == 1

    asyncio.run(run())


def test_parse_small_sections_are_merged_and_headings_preserved():
    """Adjacent small sections share chunks without losing their headings."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            sections = "\n\n".join(f"## Section-{i:03d}\n\nfact-{i:03d}" for i in range(40))
            path = _write_md(tmp, "sections.md", f"# Root\n\n{sections}")
            chunker = MarkdownFileChunker(chunk_chars=200, embed_toc=False)
            _, chunks = await chunker.chunk(path)

            assert 1 < len(chunks) < 40
            for i in range(40):
                heading = f"## Section-{i:03d}"
                assert sum(chunk.text.count(heading) for chunk in chunks) == 1

    asyncio.run(run())


def test_parse_embed_toc_uses_breadcrumbs_without_sibling_duplication():
    """TOC context repeats ancestors, not parallel section headings."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            sections = "\n\n".join(f"## Parallel-{i:03d}\n\nfact-{i:03d}" for i in range(40))
            path = _write_md(tmp, "breadcrumbs.md", f"# Root\n\n{sections}")
            chunker = MarkdownFileChunker(chunk_chars=200, embed_toc=True)
            _, chunks = await chunker.chunk(path)

            assert len(chunks) > 1
            assert all("# Root" in chunk.text for chunk in chunks)
            for i in range(40):
                heading = f"## Parallel-{i:03d}"
                assert sum(chunk.text.count(heading) for chunk in chunks) == 1

    asyncio.run(run())


def test_parse_heading_heavy_output_grows_linearly():
    """A thousand parallel sections remain a small, linear number of chunks."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            sections = "\n\n".join(f"## Observation-{i:04d}\n\nsynthetic fact" for i in range(1000))
            body = f"# Root\n\n{sections}"
            path = _write_md(tmp, "linear.md", body)
            chunker = MarkdownFileChunker(chunk_chars=10000, embed_toc=True, max_ast_sections=None)
            _, chunks = await chunker.chunk(path)

            assert len(chunks) < 10
            assert sum(len(chunk.text) for chunk in chunks) < 2 * len(body)

    asyncio.run(run())


def test_parse_nested_sections_merge_within_recursive_context():
    """Nested continuations retain their branch breadcrumb after recursive merging."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            branches = []
            for branch, fact in (("A", "a"), ("B", "b")):
                leaves = "\n\n".join(f"### {branch}{i}\n\n{fact * 45}" for i in range(4))
                branches.append(f"## {branch}\n\n{leaves}")
            branch_text = "\n\n".join(branches)
            path = _write_md(tmp, "nested.md", f"# Root\n\n{branch_text}")
            chunker = MarkdownFileChunker(chunk_chars=150, embed_toc=True)
            _, chunks = await chunker.chunk(path)

            assert len(chunks) == 4
            assert all(len(chunk.text) <= chunker.chunk_chars for chunk in chunks)
            assert chunks[1].text.startswith("# Root\n\n## A\n\n### A2")
            assert chunks[2].text.startswith("# Root\n\n## B\n\n### B0")
            assert "## B" not in chunks[1].text

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
    test_parse_frontmatter_metadata_is_opt_in()
    test_parse_small_body_one_chunk()
    test_parse_small_children_are_cached_without_recursive_calls()
    test_parse_child_cache_flushes_at_exact_limit()
    test_parse_oversized_child_flushes_parent_cache()
    test_parse_oversized_body_splits()
    test_parse_chunk_ids_match_node_chunk_ids()
    test_parse_links_literal_targets()
    test_parse_links_short_and_no_ext_kept_literally()
    test_parse_links_predicate_inline_and_line()
    test_parse_links_deduped()
    test_parse_min_chunk_chars_clamped()
    test_parse_embed_toc_prefixes_chunk_text()
    test_parse_embed_toc_is_enabled_by_default()
    test_count_sections_ignores_fenced_headings_and_supports_setext()
    test_parse_excessive_sections_uses_plain_text_without_ast()
    test_parse_section_limit_is_inclusive_for_ast()
    test_parse_small_sections_are_merged_and_headings_preserved()
    test_parse_embed_toc_uses_breadcrumbs_without_sibling_duplication()
    test_parse_heading_heavy_output_grows_linearly()
    test_parse_nested_sections_merge_within_recursive_context()
    test_parse_frontmatter_preserves_original_line_numbers()
    test_parse_frontmatter_offsets_split_table_rows()
    test_parse_bad_frontmatter_does_not_abort_chunking()
    print("\n所有测试通过!")
