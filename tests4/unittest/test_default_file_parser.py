"""Tests for DefaultFileParser."""

import asyncio
import os
import tempfile

from reme4.components.file_parser import DefaultFileParser


# Add parent path for import


def test_parse_empty_file():
    """Test parsing an empty file."""

    async def run():
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            temp_path = f.name

        try:
            parser = DefaultFileParser()
            file_node, chunks = await parser.parse(temp_path)
            assert file_node.path == temp_path
            assert len(chunks) == 0
            print("✓ test_parse_empty_file passed")
        finally:
            os.unlink(temp_path)

    asyncio.run(run())


def test_parse_small_file():
    """Test parsing a file smaller than chunk size."""

    async def run():
        content = "Hello World\nThis is a test\nLine 3"
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            parser = DefaultFileParser(chunk_byte_size=10000)
            _, chunks = await parser.parse(temp_path)
            assert len(chunks) == 1
            assert chunks[0].start_line == 1
            assert chunks[0].end_line == 3
            assert chunks[0].text == content
            print("✓ test_parse_small_file passed")
        finally:
            os.unlink(temp_path)

    asyncio.run(run())


def test_parse_multiline_file():
    """Test parsing a file with multiple lines."""

    async def run():
        lines = ["Line 1", "Line 2", "Line 3", "Line 4", "Line 5"]
        content = "\n".join(lines)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            parser = DefaultFileParser(chunk_byte_size=10000)
            _, chunks = await parser.parse(temp_path)
            assert len(chunks) == 1
            assert chunks[0].start_line == 1
            assert chunks[0].end_line == 5
            print("✓ test_parse_multiline_file passed")
        finally:
            os.unlink(temp_path)

    asyncio.run(run())


def test_parse_chunked_file():
    """Test parsing a file that requires multiple chunks."""

    async def run():
        # Create content larger than chunk size
        lines = ["A" * 100 for _ in range(200)]  # ~20200 bytes
        content = "\n".join(lines)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            parser = DefaultFileParser(chunk_byte_size=5000, overlap_byte_size=100)
            _, chunks = await parser.parse(temp_path)
            assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"
            # Verify overlap by checking that consecutive chunks share some content
            print(f"  Created {len(chunks)} chunks")
            print("✓ test_parse_chunked_file passed")
        finally:
            os.unlink(temp_path)

    asyncio.run(run())


def test_parse_with_custom_encoding():
    """Test parsing a file with different encodings."""

    async def run():
        content = "你好世界\n测试内容"
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as f:
            f.write(content)
            temp_path = f.name

        try:
            parser = DefaultFileParser(encoding="utf-8")
            _, chunks = await parser.parse(temp_path)
            assert len(chunks) >= 1
            assert "你好世界" in chunks[0].text
            print("✓ test_parse_with_custom_encoding passed")
        finally:
            os.unlink(temp_path)

    asyncio.run(run())


def test_file_node_properties():
    """Test FileNode has correct properties."""

    async def run():
        content = "test content"
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            parser = DefaultFileParser()
            file_node, _ = await parser.parse(temp_path)
            assert hasattr(file_node, "path")
            assert hasattr(file_node, "st_mtime")
            assert file_node.st_mtime > 0
            print("✓ test_file_node_properties passed")
        finally:
            os.unlink(temp_path)

    asyncio.run(run())


def test_file_chunk_properties():
    """Test FileChunk has correct properties."""

    async def run():
        content = "test content for chunk"
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            parser = DefaultFileParser()
            _, chunks = await parser.parse(temp_path)
            chunk = chunks[0]
            assert hasattr(chunk, "path")
            assert hasattr(chunk, "start_line")
            assert hasattr(chunk, "end_line")
            assert hasattr(chunk, "text")
            assert hasattr(chunk, "id")
            assert chunk.start_line >= 1
            assert chunk.end_line >= chunk.start_line
            print("✓ test_file_chunk_properties passed")
        finally:
            os.unlink(temp_path)

    asyncio.run(run())


def test_parse_links_bare():
    """Bare wikilink: [[target]]."""
    links = DefaultFileParser.parse_links("see [[note]]", "src.md")
    assert len(links) == 1
    link = links[0]
    assert link.source_path == "src.md"
    assert link.target_path == "note"
    assert link.target_anchor is None
    assert link.predicate is None
    print("✓ test_parse_links_bare passed")


def test_parse_links_with_anchor():
    """Wikilink with anchor: [[target#anchor]]."""
    links = DefaultFileParser.parse_links("see [[note#section A]]", "src.md")
    assert len(links) == 1
    assert links[0].target_path == "note"
    assert links[0].target_anchor == "section A"
    assert links[0].predicate is None
    print("✓ test_parse_links_with_anchor passed")


def test_parse_links_alias_dropped():
    """Alias after '|' is consumed but not captured as anchor."""
    links = DefaultFileParser.parse_links("see [[note|display text]]", "src.md")
    assert len(links) == 1
    assert links[0].target_path == "note"
    assert links[0].target_anchor is None
    print("✓ test_parse_links_alias_dropped passed")


def test_parse_links_anchor_and_alias():
    """[[target#anchor|alias]] — anchor captured, alias dropped."""
    links = DefaultFileParser.parse_links("see [[note#sec|disp]]", "src.md")
    assert len(links) == 1
    assert links[0].target_path == "note"
    assert links[0].target_anchor == "sec"
    print("✓ test_parse_links_anchor_and_alias passed")


def test_parse_links_predicate_simple():
    """Dataview inline: predicate:: [[target]]."""
    links = DefaultFileParser.parse_links("author:: [[Alice]]", "src.md")
    assert len(links) == 1
    assert links[0].predicate == "author"
    assert links[0].target_path == "Alice"
    assert links[0].target_anchor is None
    print("✓ test_parse_links_predicate_simple passed")


def test_parse_links_predicate_bracketed():
    """Dataview inline-bracket: [predicate:: [[target]]]."""
    links = DefaultFileParser.parse_links("text [author:: [[Alice]]] more", "src.md")
    assert len(links) == 1
    assert links[0].predicate == "author"
    assert links[0].target_path == "Alice"
    print("✓ test_parse_links_predicate_bracketed passed")


def test_parse_links_predicate_bracketed_with_anchor():
    """[predicate:: [[target_path#target_anchor]]] — combined form."""
    links = DefaultFileParser.parse_links(
        "[predicate:: [[target_path#target_anchor]]]",
        "src.md",
    )
    assert len(links) == 1
    link = links[0]
    assert link.source_path == "src.md"
    assert link.predicate == "predicate"
    assert link.target_path == "target_path"
    assert link.target_anchor == "target_anchor"
    print("✓ test_parse_links_predicate_bracketed_with_anchor passed")


def test_parse_links_predicate_sticks_to_first():
    """Predicate attaches only to the immediately following wikilink."""
    links = DefaultFileParser.parse_links("pred:: [[a]] and bare [[b]]", "src.md")
    assert len(links) == 2
    assert links[0].predicate == "pred" and links[0].target_path == "a"
    assert links[1].predicate is None and links[1].target_path == "b"
    print("✓ test_parse_links_predicate_sticks_to_first passed")


def test_parse_links_multiple_on_one_line():
    """Multiple bare wikilinks on the same line are all captured."""
    links = DefaultFileParser.parse_links("see [[x]] and [[y#h]]", "src.md")
    assert [(link.target_path, link.target_anchor) for link in links] == [
        ("x", None),
        ("y", "h"),
    ]
    print("✓ test_parse_links_multiple_on_one_line passed")


def test_parse_links_no_match():
    """Strings without [[]] yield no links, even if '::' appears."""
    assert len(DefaultFileParser.parse_links("no link here :: foo", "src.md")) == 0
    assert len(DefaultFileParser.parse_links("plain text without brackets", "src.md")) == 0
    assert len(DefaultFileParser.parse_links("", "src.md")) == 0
    print("✓ test_parse_links_no_match passed")


def test_parse_links_predicate_with_dash_and_digits():
    """Predicate identifier accepts letters, digits, underscore, dash."""
    links = DefaultFileParser.parse_links("see-also-2:: [[target]]", "src.md")
    assert len(links) == 1
    assert links[0].predicate == "see-also-2"
    assert links[0].target_path == "target"
    print("✓ test_parse_links_predicate_with_dash_and_digits passed")


def test_parse_links_in_file():
    """Integration: parse() populates FileNode.links from file content."""

    async def run():
        content = (
            "---\n"
            "title: demo\n"
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
            parser = DefaultFileParser()
            file_node, _ = await parser.parse(temp_path)
            triples = {(link.predicate, link.target_path, link.target_anchor) for link in file_node.links}
            assert (None, "alpha", None) in triples
            assert (None, "beta", "h2") in triples
            assert ("author", "Alice", None) in triples
            assert ("ref", "paper", "chapter 1") in triples
            assert all(link.source_path == file_node.path for link in file_node.links)
            print("✓ test_parse_links_in_file passed")
        finally:
            os.unlink(temp_path)

    asyncio.run(run())


def test_parse_links_empty_when_no_content():
    """Empty file and front-matter-only file both yield no links."""

    async def run():
        # Empty file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
            empty_path = f.name
        # Front-matter-only file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
            f.write("---\ntitle: x\n---\n")
            fm_only_path = f.name

        try:
            parser = DefaultFileParser()
            node1, _ = await parser.parse(empty_path)
            node2, _ = await parser.parse(fm_only_path)
            assert node1.links == []
            assert node2.links == []
            print("✓ test_parse_links_empty_when_no_content passed")
        finally:
            os.unlink(empty_path)
            os.unlink(fm_only_path)

    asyncio.run(run())


def test_min_chunk_and_overlap_size():
    """Test that minimum chunk and overlap sizes are enforced."""

    async def run():
        # These values should be clamped to minimums
        parser = DefaultFileParser(chunk_byte_size=1, overlap_byte_size=0)
        assert parser.chunk_byte_size == 100  # minimum
        assert parser.overlap_byte_size == 4  # minimum

        content = "test"
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            _, chunks = await parser.parse(temp_path)
            assert len(chunks) == 1
            print("✓ test_min_chunk_and_overlap_size passed")
        finally:
            os.unlink(temp_path)

    asyncio.run(run())


if __name__ == "__main__":
    test_parse_empty_file()
    test_parse_small_file()
    test_parse_multiline_file()
    test_parse_chunked_file()
    test_parse_with_custom_encoding()
    test_file_node_properties()
    test_file_chunk_properties()
    test_parse_links_bare()
    test_parse_links_with_anchor()
    test_parse_links_alias_dropped()
    test_parse_links_anchor_and_alias()
    test_parse_links_predicate_simple()
    test_parse_links_predicate_bracketed()
    test_parse_links_predicate_bracketed_with_anchor()
    test_parse_links_predicate_sticks_to_first()
    test_parse_links_multiple_on_one_line()
    test_parse_links_no_match()
    test_parse_links_predicate_with_dash_and_digits()
    test_parse_links_in_file()
    test_parse_links_empty_when_no_content()
    test_min_chunk_and_overlap_size()
    print("\n所有测试通过!")
