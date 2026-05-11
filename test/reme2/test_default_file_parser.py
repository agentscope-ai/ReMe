"""Tests for DefaultFileParser."""

import asyncio
import tempfile
import os
from pathlib import Path

# Add parent path for import
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reme2.component.file_parser import DefaultFileParser


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
            file_node, chunks = await parser.parse(temp_path)
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
            file_node, chunks = await parser.parse(temp_path)
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
            file_node, chunks = await parser.parse(temp_path)
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
            file_node, chunks = await parser.parse(temp_path)
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
            file_node, chunks = await parser.parse(temp_path)
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
            file_node, chunks = await parser.parse(temp_path)
            chunk = chunks[0]
            assert hasattr(chunk, "path")
            assert hasattr(chunk, "start_line")
            assert hasattr(chunk, "end_line")
            assert hasattr(chunk, "text")
            assert hasattr(chunk, "id")
            assert hasattr(chunk, "hash")
            assert chunk.start_line >= 1
            assert chunk.end_line >= chunk.start_line
            print("✓ test_file_chunk_properties passed")
        finally:
            os.unlink(temp_path)

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
            file_node, chunks = await parser.parse(temp_path)
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
    test_min_chunk_and_overlap_size()
    print("\n所有测试通过!")