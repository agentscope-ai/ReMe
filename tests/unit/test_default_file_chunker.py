"""Tests for DefaultFileChunker."""

import asyncio
import os
import tempfile

import pytest

from reme.components.file_chunker import DefaultFileChunker
from reme.schema import FileChunk, FileFrontMatter
from reme.utils.wikilink_handler import WikilinkHandler

# Add parent path for import


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize(
    "content",
    [
        "",
        "你好世界\n测试内容",
        "# H\nline 1\nline 2\n",
        "---\nname: t\n---\n# H\nline 1\nline 2\n",
        "---\nname: t\n---\n\n\n# H\nline 1\nline 2\n",
        "---\nname: t\n---\n",
        "\n\n",
        "---\nname: [invalid\n---\n[[target.md]]\n",
        "---\nname: unclosed\n[[target.md]]",
    ],
)
def test_file_extensions_preserve_identical_plain_text(tmp_path, content, newline):
    """The default chunker keeps the whole file without interpreting Markdown."""
    results = []
    source = content.replace("\n", newline).encode("utf-8")
    for suffix in (".md", ".MD", ".txt"):
        path = tmp_path / f"note{suffix}"
        path.write_bytes(source)
        node, chunks = asyncio.run(DefaultFileChunker().chunk(path))

        assert path.read_bytes() == source
        assert node.path == str(path)
        assert node.front_matter == FileFrontMatter()
        assert node.links == []
        assert node.chunk_ids == [chunk.id for chunk in chunks]
        if content:
            assert len(chunks) == 1
            chunk = chunks[0]
            end_line = content.count("\n") + (not content.endswith("\n"))
            assert (chunk.start_line, chunk.end_line, chunk.text) == (1, end_line, content)
            expected = FileChunk(path=chunk.path, start_line=1, end_line=end_line, text=content).set_hash_id()
            assert chunk.id == expected.id
        else:
            assert chunks == []
        results.append([(chunk.text, chunk.start_line, chunk.end_line) for chunk in chunks])
    assert results[0] == results[1] == results[2]


def test_plain_text_overlaps_preserve_source_ranges_and_ids(tmp_path):
    """Every overlapping chunk includes its unmodified source bytes and range."""
    content = "---\nname: t\n---\n\n\n" + "".join(f"line {i:02d}\n" for i in range(40))
    chunker = DefaultFileChunker(chunk_byte_size=100, overlap_byte_size=8)
    results = []
    for suffix in (".md", ".txt"):
        path = tmp_path / f"note{suffix}"
        path.write_bytes(content.encode("utf-8"))
        node, chunks = asyncio.run(chunker.chunk(path))

        starts = list(range(0, len(content) - chunker.overlap_byte_size, 92))
        assert len(chunks) == len(starts) > 1
        for start, chunk in zip(starts, chunks):
            end = min(start + 100, len(content))
            assert chunk.text == content[start:end]
            assert chunk.start_line == content[:start].count("\n") + 1
            assert chunk.end_line == content[:end].count("\n") + (not content[:end].endswith("\n"))
            expected = FileChunk(
                path=chunk.path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                text=chunk.text,
            ).set_hash_id()
            assert chunk.id == expected.id
        assert node.chunk_ids == [chunk.id for chunk in chunks]
        results.append([(chunk.text, chunk.start_line, chunk.end_line) for chunk in chunks])
    assert results[0] == results[1]


@pytest.mark.parametrize("suffix", [".md", ".txt"])
def test_wikilinks_do_not_change_plain_text_boundaries(tmp_path, suffix):
    """The default path may cut a literal wikilink just like any other text."""
    content = "x" * 90 + "[[a-very-long-target]]" + "y" * 90
    path = tmp_path / f"note{suffix}"
    path.write_bytes(content.encode("utf-8"))
    node, chunks = asyncio.run(DefaultFileChunker(chunk_byte_size=100, overlap_byte_size=10).chunk(path))

    assert [chunk.text for chunk in chunks] == [content[0:100], content[90:190], content[180:]]
    assert node.links == []
    assert node.front_matter == FileFrontMatter()


def test_invalid_encoding_policy_is_validated():
    """Reject misspelled policies instead of silently changing decoding behavior."""
    try:
        DefaultFileChunker(invalid_encoding_policy="ignore")
    except ValueError as exc:
        assert "invalid_encoding_policy" in str(exc)
    else:
        raise AssertionError("invalid encoding policy must be rejected")


def test_constructor_preserves_positional_arguments():
    """New decoding options must not reinterpret the established positional API."""
    chunker = DefaultFileChunker("utf-8", 5000, 100)

    assert chunker.encoding == "utf-8"
    assert chunker.chunk_byte_size == 5000
    assert chunker.overlap_byte_size == 100


@pytest.mark.asyncio
async def test_newlines_are_normalized_before_chunking():
    """Binary reads retain the universal-newline behavior of the old text reader."""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        f.write(b"alpha\r\nbeta\rgamma\n")
        temp_path = f.name

    try:
        chunker = DefaultFileChunker()
        _, original_chunks = await chunker.chunk(temp_path)
        with open(temp_path, "wb") as f:
            f.write(b"alpha\nbeta\ngamma\n")
        _, normalized_chunks = await chunker.chunk(temp_path)

        assert [chunk.text for chunk in original_chunks] == ["alpha\nbeta\ngamma\n"]
        assert [chunk.id for chunk in original_chunks] == [chunk.id for chunk in normalized_chunks]
    finally:
        os.unlink(temp_path)


@pytest.mark.asyncio
async def test_ascii_replacement_remains_encodable():
    """Replacement mode must survive byte-based chunking with a single-byte codec."""

    source = b"valid\xffinvalid"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        f.write(source)
        temp_path = f.name

    try:
        _, chunks = await DefaultFileChunker(encoding="ascii").chunk(temp_path)

        assert [chunk.text for chunk in chunks] == ["valid?invalid"]
        with open(temp_path, "rb") as f:
            assert f.read() == source
    finally:
        os.unlink(temp_path)


def test_parse_links_bare():
    """Bare wikilink: [[target]]."""
    links = WikilinkHandler.extract_links("see [[note]]", "src.md")
    assert len(links) == 1
    link = links[0]
    assert link.source_path == "src.md"
    assert link.target_path == "note"
    assert link.target_anchor is None


def test_parse_links_with_anchor():
    """Wikilink with anchor: [[target#anchor]]."""
    links = WikilinkHandler.extract_links("see [[note#section A]]", "src.md")
    assert len(links) == 1
    assert links[0].target_path == "note"
    assert links[0].target_anchor == "section A"


def test_parse_links_alias_dropped():
    """Alias after '|' is consumed but not captured as anchor."""
    links = WikilinkHandler.extract_links("see [[note|display text]]", "src.md")
    assert len(links) == 1
    assert links[0].target_path == "note"
    assert links[0].target_anchor is None


def test_parse_links_anchor_and_alias():
    """[[target#anchor|alias]] — anchor captured, alias dropped."""
    links = WikilinkHandler.extract_links("see [[note#sec|disp]]", "src.md")
    assert len(links) == 1
    assert links[0].target_path == "note"
    assert links[0].target_anchor == "sec"


def test_parse_links_ignores_legacy_relation_wrappers():
    """Legacy relation text remains compatible as ordinary wikilinks."""
    links = WikilinkHandler.extract_links(
        "related:: [[a]]\n- related:: [[b]]\n[related:: [[c#section]]]",
        "src.md",
    )
    assert [(link.target_path, link.target_anchor) for link in links] == [
        ("a", None),
        ("b", None),
        ("c", "section"),
    ]
    assert all(link.predicate is None for link in links)


def test_parse_links_multiple_on_one_line():
    """Multiple bare wikilinks on the same line are all captured."""
    links = WikilinkHandler.extract_links("see [[x]] and [[y#h]]", "src.md")
    assert [(link.target_path, link.target_anchor) for link in links] == [
        ("x", None),
        ("y", "h"),
    ]


def test_parse_wikilink_line_ranges():
    """Workspace paths and supported line-range forms create FileLink edges."""
    links = WikilinkHandler.extract_links(
        "[[daily/2026-06-20/session.md]] [[notes/example.md#L9]] "
        + "[[notes/example.md#L9-L10]] [[notes/example.md#L9-L10,L15-L20]]",
        "src.md",
    )
    assert [(link.target_path, link.target_anchor) for link in links] == [
        ("daily/2026-06-20/session.md", None),
        ("notes/example.md", "L9"),
        ("notes/example.md", "L9-L10"),
        ("notes/example.md", "L9-L10,L15-L20"),
    ]


def test_parse_markdown_links_are_ignored():
    """Ordinary Markdown links do not create FileLink edges."""
    links = WikilinkHandler.extract_links(
        "[plain](../wiki/a.md) [section](../wiki/a.md#section) [ranges](../wiki/b.md#L9-L10,L15-L20)",
        "daily/note.md",
    )
    assert not links


def test_parse_links_no_match():
    """Strings without [[]] yield no links, even if '::' appears."""
    assert len(WikilinkHandler.extract_links("no link here :: foo", "src.md")) == 0
    assert len(WikilinkHandler.extract_links("plain text without brackets", "src.md")) == 0
    assert len(WikilinkHandler.extract_links("", "src.md")) == 0


@pytest.mark.asyncio
async def test_parse_links_in_file():
    """Literal wikilinks and YAML remain text without populating node metadata."""

    content = (
        "---\n"
        "name: demo\n"
        "---\n"
        "\n"
        "Intro paragraph with [[alpha]] and [[beta#h2]].\n"
        "author:: [[Alice]]\n"
        "[ref:: [[paper#chapter 1]]]\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
        f.write(content)
        temp_path = f.name

    try:
        chunker = DefaultFileChunker()
        file_node, chunks = await chunker.chunk(temp_path)
        assert file_node.links == []
        assert file_node.front_matter == FileFrontMatter()
        assert len(chunks) == 1
        assert chunks[0].text == content
    finally:
        os.unlink(temp_path)


@pytest.mark.asyncio
async def test_parse_links_empty_when_no_content():
    """Empty file and front-matter-only file both yield no links."""

    # Empty file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
        empty_path = f.name
    # Front-matter-only file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
        f.write("---\nname: x\n---\n")
        fm_only_path = f.name

    try:
        chunker = DefaultFileChunker()
        node1, _ = await chunker.chunk(empty_path)
        node2, _ = await chunker.chunk(fm_only_path)
        assert node1.links == []
        assert node2.links == []
    finally:
        os.unlink(empty_path)
        os.unlink(fm_only_path)


@pytest.mark.asyncio
async def test_min_chunk_and_overlap_size():
    """Test that minimum chunk and overlap sizes are enforced."""

    # These values should be clamped to minimums
    chunker = DefaultFileChunker(chunk_byte_size=1, overlap_byte_size=0)
    assert chunker.chunk_byte_size == 100  # minimum
    assert chunker.overlap_byte_size == 4  # minimum

    content = "test"
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write(content)
        temp_path = f.name

    try:
        _, chunks = await chunker.chunk(temp_path)
        assert len(chunks) == 1
    finally:
        os.unlink(temp_path)
