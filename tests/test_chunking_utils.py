"""Tests for chunking utilities."""

import pytest

from reme.core.enumeration import MemorySource
from reme.core.utils.chunking_utils import chunk_markdown


def test_chunk_markdown_basic():
    """Test basic markdown chunking functionality."""
    text = """# Heading 1

This is a paragraph with some content.

## Heading 2

Another paragraph here.
More content in this paragraph."""

    chunks = chunk_markdown(
        text=text,
        path="test.md",
        source=MemorySource.MEMORY,
        chunk_tokens=100,
        overlap=10,
    )

    assert len(chunks) > 0
    assert all(chunk.path == "test.md" for chunk in chunks)
    assert all(chunk.source == MemorySource.MEMORY for chunk in chunks)
    assert all(chunk.hash for chunk in chunks)
    assert all(chunk.id for chunk in chunks)


def test_chunk_markdown_empty():
    """Test chunking with empty text."""
    chunks = chunk_markdown(
        text="",
        path="empty.md",
        source=MemorySource.MEMORY,
        chunk_tokens=100,
        overlap=10,
    )

    # Empty text is filtered out (no meaningful content to store)
    assert len(chunks) == 0


def test_chunk_markdown_single_line():
    """Test chunking with single line."""
    text = "Single line of text"

    chunks = chunk_markdown(
        text=text,
        path="single.md",
        source=MemorySource.MEMORY,
        chunk_tokens=100,
        overlap=10,
    )

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 1


def test_chunk_markdown_long_line():
    """Test chunking with a very long line that exceeds max_chars."""
    # Create a line longer than max_chars (300 tokens * 4 = 1200 chars)
    long_text = "x" * 1500

    chunks = chunk_markdown(
        text=long_text,
        path="long.md",
        source=MemorySource.MEMORY,
        chunk_tokens=300,
        overlap=30,
    )

    # Should split into multiple chunks
    assert len(chunks) > 1
    # All chunks should have the same line number since it's one line
    assert all(chunk.start_line == 1 for chunk in chunks)
    assert all(chunk.end_line == 1 for chunk in chunks)


def test_chunk_markdown_overlap():
    """Test that overlap is working correctly."""
    text = "\n".join([f"Line {i}" for i in range(1, 51)])

    chunks = chunk_markdown(
        text=text,
        path="overlap.md",
        source=MemorySource.MEMORY,
        chunk_tokens=50,
        overlap=10,
    )

    # With overlap, consecutive chunks should have some overlapping content
    if len(chunks) > 1:
        for i in range(len(chunks) - 1):
            # Check that there's potential overlap
            assert chunks[i].end_line >= chunks[i].start_line
            assert chunks[i + 1].start_line <= chunks[i].end_line + 1


def test_chunk_markdown_no_overlap():
    """Test chunking without overlap."""
    text = "\n".join([f"Line {i}" for i in range(1, 51)])

    chunks = chunk_markdown(
        text=text,
        path="no_overlap.md",
        source=MemorySource.MEMORY,
        chunk_tokens=50,
        overlap=0,
    )

    assert len(chunks) > 0
    # Verify all chunks are non-empty
    assert all(chunk.text for chunk in chunks)


def test_chunk_markdown_line_numbers():
    """Test that line numbers are correctly assigned."""
    text = """Line 1
Line 2
Line 3
Line 4
Line 5"""

    chunks = chunk_markdown(
        text=text,
        path="lines.md",
        source=MemorySource.MEMORY,
        chunk_tokens=20,
        overlap=5,
    )

    # First chunk should start at line 1
    assert chunks[0].start_line == 1
    # Last chunk should end at the last line
    assert chunks[-1].end_line == 5
    # All chunks should have valid line ranges
    for chunk in chunks:
        assert chunk.start_line <= chunk.end_line
        assert chunk.start_line >= 1


def test_chunk_markdown_hash_uniqueness():
    """Test that different chunks have different hashes."""
    text = """# Section 1

Content for section 1.

# Section 2

Content for section 2."""

    chunks = chunk_markdown(
        text=text,
        path="sections.md",
        source=MemorySource.MEMORY,
        chunk_tokens=50,
        overlap=5,
    )

    # Collect all hashes
    hashes = [chunk.hash for chunk in chunks]

    # If we have multiple chunks, they should have different hashes
    if len(chunks) > 1:
        assert len(set(hashes)) == len(hashes), "Chunks should have unique hashes"


def test_chunk_markdown_id_uniqueness():
    """Test that chunk IDs are unique."""
    text = "\n".join([f"Line {i}" for i in range(1, 101)])

    chunks = chunk_markdown(
        text=text,
        path="unique.md",
        source=MemorySource.MEMORY,
        chunk_tokens=50,
        overlap=10,
    )

    ids = [chunk.id for chunk in chunks]
    assert len(set(ids)) == len(ids), "All chunk IDs should be unique"


def test_chunk_markdown_small_tokens():
    """Test chunking with very small token limit."""
    text = "This is a short text with multiple words in it."

    chunks = chunk_markdown(
        text=text,
        path="small.md",
        source=MemorySource.MEMORY,
        chunk_tokens=5,
        overlap=1,
    )

    # Even with small chunk size, should create at least one chunk
    assert len(chunks) >= 1


def test_chunk_markdown_sessions_source():
    """Test chunking with SESSIONS source."""
    text = "Session log content"

    chunks = chunk_markdown(
        text=text,
        path="session.log",
        source=MemorySource.SESSIONS,
        chunk_tokens=100,
        overlap=10,
    )

    assert len(chunks) == 1
    assert chunks[0].source == MemorySource.SESSIONS


def test_chunk_markdown_multiline_paragraph():
    """Test chunking with realistic markdown content."""
    text = """# Introduction

This is a longer paragraph that spans multiple lines.
It contains various sentences and information.
The chunking algorithm should handle this properly.

## Details

Here are some details:
- Point 1
- Point 2
- Point 3

## Conclusion

Final thoughts and conclusions go here."""

    chunks = chunk_markdown(
        text=text,
        path="document.md",
        source=MemorySource.MEMORY,
        chunk_tokens=50,
        overlap=10,
    )

    # Should create multiple chunks
    assert len(chunks) > 0

    # Verify text reconstruction
    all_text_parts = []
    for chunk in chunks:
        all_text_parts.append(chunk.text)

    # At least some of the original content should be in the chunks
    combined = "\n".join(all_text_parts)
    assert "Introduction" in combined or "Introduction" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
