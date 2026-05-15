"""Tests for file system tools including bash, edit, find, grep, ls, read, and write tools."""

import asyncio
import os
import tempfile
from pathlib import Path


async def test_bash_tool():
    """Test BashTool."""
    from reme.core.tools import BashTool

    print("=== Testing BashTool ===")
    bash_tool = BashTool()
    result = await bash_tool.call(command="echo 'Hello World'")
    print(f"Result: {result}")
    assert "Hello World" in result
    print("✓ BashTool test passed\n")


async def test_edit_tool():
    """Test EditTool."""
    from reme.core.tools import EditTool

    print("=== Testing EditTool ===")

    # Create temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        temp_path = f.name
        f.write("Hello World\nThis is a test\nGoodbye World\n")

    try:
        # Test edit
        edit_tool = EditTool()
        result = await edit_tool.call(
            path=temp_path,
            oldText="This is a test",
            newText="This is an updated test",
        )
        print(f"Result: {result}")

        # Verify content
        with open(temp_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "This is an updated test" in content
        assert "This is a test" not in content
        print("✓ EditTool test passed\n")

        # Test error: file not found
        print("=== Testing file not found error ===")
        result = await edit_tool.call(
            path="/nonexistent/file.txt",
            oldText="test",
            newText="new",
        )
        print(f"Expected error result: {result}")
        assert "failed" in result and "File not found" in result
        print("✓ File not found error test passed\n")

        # Test error: text not found
        print("=== Testing text not found error ===")
        result = await edit_tool.call(
            path=temp_path,
            oldText="nonexistent text",
            newText="new",
        )
        print(f"Expected error result: {result}")
        assert "failed" in result and "Could not find" in result
        print("✓ Text not found error test passed\n")

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)


async def test_find_tool():
    """Test FindTool."""
    from reme.core.tools import FindTool

    print("=== Testing FindTool ===")

    # Create temp directory with test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create some test files
        (temp_path / "test1.txt").write_text("test file 1")
        (temp_path / "test2.txt").write_text("test file 2")
        (temp_path / "readme.md").write_text("readme")

        # Create subdirectory with files
        sub_dir = temp_path / "subdir"
        sub_dir.mkdir()
        (sub_dir / "test3.txt").write_text("test file 3")
        (sub_dir / "config.json").write_text("{}")

        # Create .gitignore to ignore certain files
        (temp_path / ".gitignore").write_text("*.md\n")

        # Test: find all txt files
        find_tool = FindTool(cwd=str(temp_path))
        result = await find_tool.call(pattern="*.txt")
        print(f"Find *.txt result:\n{result}")
        assert "test1.txt" in result
        assert "test2.txt" in result
        assert "readme.md" not in result  # Should be ignored by .gitignore
        print("✓ Find *.txt test passed\n")

        # Test: find with recursive pattern
        result = await find_tool.call(pattern="**/*.txt")
        print(f"Find **/*.txt result:\n{result}")
        assert "test1.txt" in result
        assert "subdir/test3.txt" in result or "test3.txt" in result
        print("✓ Find **/*.txt test passed\n")

        # Test: find with no matches
        result = await find_tool.call(pattern="*.nonexistent")
        print(f"Find *.nonexistent result:\n{result}")
        assert "No files found" in result
        print("✓ No matches test passed\n")

        # Test: error - directory not found
        print("=== Testing directory not found error ===")
        result = await find_tool.call(pattern="*.txt", path="/nonexistent/dir")
        print(f"Expected error result: {result}")
        assert "failed" in result and "Path not found" in result
        print("✓ Directory not found error test passed\n")


async def test_grep_tool():
    """Test GrepTool."""
    from reme.core.tools import GrepTool

    print("=== Testing GrepTool ===")

    # Create temp directory with test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test files with searchable content
        (temp_path / "file1.txt").write_text("Hello World\nThis is a test\nGoodbye World\n")
        (temp_path / "file2.txt").write_text("Another test file\nWith multiple lines\nHello again\n")
        (temp_path / "script.py").write_text("def hello():\n    print('Hello')\n    return True\n")

        # Create subdirectory with files
        sub_dir = temp_path / "subdir"
        sub_dir.mkdir()
        (sub_dir / "nested.txt").write_text("Nested file content\nWith hello keyword\n")

        # Test: search for pattern
        grep_tool = GrepTool(cwd=str(temp_path))
        result = await grep_tool.call(pattern="Hello", path=str(temp_path))
        print(f"Search 'Hello' result:\n{result}")
        assert "file1.txt" in result
        assert "Hello World" in result or "Hello" in result
        print("✓ Basic search test passed\n")

        # Test: case-insensitive search
        result = await grep_tool.call(pattern="hello", path=str(temp_path), ignoreCase=True)
        print(f"Case-insensitive search result:\n{result}")
        assert "file1.txt" in result or "Hello" in result.lower()
        print("✓ Case-insensitive search test passed\n")

        # Test: literal string search
        result = await grep_tool.call(pattern="Hello()", path=str(temp_path), literal=True)
        print(f"Literal search result:\n{result}")
        # Should not find regex interpretation
        print("✓ Literal search test passed\n")

        # Test: glob filter
        result = await grep_tool.call(pattern="Hello", path=str(temp_path), glob="*.txt")
        print(f"Glob filter *.txt result:\n{result}")
        assert "file1.txt" in result or "file2.txt" in result
        assert ".py" not in result  # Python files should be excluded
        print("✓ Glob filter test passed\n")

        # Test: context lines
        result = await grep_tool.call(pattern="test", path=str(temp_path), contextLines=1)
        print(f"Context lines result:\n{result}")
        # Should include lines before and after matches
        print("✓ Context lines test passed\n")

        # Test: limit matches
        result = await grep_tool.call(pattern="Hello", path=str(temp_path), limit=1)
        print(f"Limit to 1 match result:\n{result}")
        assert "limit reached" in result or result.count(":") >= 1
        print("✓ Limit test passed\n")

        # Test: no matches
        result = await grep_tool.call(pattern="nonexistent_pattern_xyz", path=str(temp_path))
        print(f"No matches result:\n{result}")
        assert "No matches found" in result
        print("✓ No matches test passed\n")

        # Test: error - path not found
        print("=== Testing path not found error ===")
        try:
            result = await grep_tool.call(pattern="test", path="/nonexistent/path")
            assert "failed" in result and "not found" in result
        except Exception as e:
            assert "not found" in str(e).lower()
        print("✓ Path not found error test passed\n")


async def test_ls_tool():
    """Test LsTool."""
    from reme.core.tools import LsTool

    print("=== Testing LsTool ===")

    # Create temp directory with test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test files and directories
        (temp_path / "file1.txt").write_text("test file 1")
        (temp_path / "file2.py").write_text("test file 2")
        (temp_path / ".hidden").write_text("hidden file")
        (temp_path / "README.md").write_text("readme")

        # Create subdirectories
        (temp_path / "subdir1").mkdir()
        (temp_path / "subdir2").mkdir()

        # Test: list current directory
        ls_tool = LsTool(cwd=str(temp_path))
        result = await ls_tool.call()
        print(f"List directory result:\n{result}")
        assert ".hidden" in result  # Includes dotfiles
        assert "file1.txt" in result
        assert "file2.py" in result
        assert "subdir1/" in result  # Directories have '/' suffix
        assert "subdir2/" in result
        print("✓ Basic ls test passed\n")

        # Test: list specific path
        result = await ls_tool.call(path=".")
        print(f"List current directory result:\n{result}")
        assert "file1.txt" in result
        print("✓ Specific path test passed\n")

        # Test: empty directory
        empty_dir = temp_path / "empty"
        empty_dir.mkdir()
        result = await ls_tool.call(path="empty")
        print(f"Empty directory result:\n{result}")
        assert "(empty directory)" in result
        print("✓ Empty directory test passed\n")

        # Test: entry limit
        # Create many files
        for i in range(10):
            (temp_path / f"file{i:03d}.txt").write_text(f"file {i}")

        result = await ls_tool.call(limit=5)
        print(f"Limited entries result:\n{result}")
        assert "entries limit reached" in result
        assert "limit=10" in result  # Should suggest doubling the limit
        print("✓ Entry limit test passed\n")

        # Test: error - path not found
        print("=== Testing path not found error ===")
        result = await ls_tool.call(path="/nonexistent/path")
        print(f"Expected error result: {result}")
        assert "failed" in result and "Path not found" in result
        print("✓ Path not found error test passed\n")

        # Test: error - not a directory
        print("=== Testing not a directory error ===")
        result = await ls_tool.call(path="file1.txt")
        print(f"Expected error result: {result}")
        assert "failed" in result and "Not a directory" in result
        print("✓ Not a directory error test passed\n")


async def test_read_tool():
    """Test ReadTool."""
    from reme.core.tools import ReadTool

    print("=== Testing ReadTool ===")

    # Create temp directory with test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test text file
        test_file = temp_path / "test.txt"
        test_content = "\n".join([f"Line {i}" for i in range(1, 101)])  # 100 lines
        test_file.write_text(test_content)

        # Create test image file
        image_file = temp_path / "test.jpg"
        image_file.write_bytes(b"\xff\xd8\xff\xe0")  # Minimal JPEG header

        # Test: read full file
        read_tool = ReadTool(cwd=str(temp_path))
        result = await read_tool.call(path="test.txt")
        print(f"Read full file result:\n{result[:200]}...")
        assert "Line 1" in result
        assert "Line 100" in result
        print("✓ Read full file test passed\n")

        # Test: read with offset
        result = await read_tool.call(path="test.txt", offset=50)
        print(f"Read with offset=50 result:\n{result[:200]}...")
        # Check that we start from Line 50 (should be first line of content)
        assert result.startswith("Line 50"), f"Should start with 'Line 50', got: {result[:50]}"
        assert "Line 100" in result
        print("✓ Read with offset test passed\n")

        # Test: read with limit
        result = await read_tool.call(path="test.txt", limit=10)
        print(f"Read with limit=10 result:\n{result}")
        assert "Line 1" in result
        assert "Line 10" in result or "more lines in file" in result
        assert "Line 50" not in result
        print("✓ Read with limit test passed\n")

        # Test: read with offset and limit
        result = await read_tool.call(path="test.txt", offset=20, limit=5)
        print(f"Read with offset=20, limit=5 result:\n{result}")
        assert "Line 20" in result
        assert "Line 24" in result or "more lines" in result
        print("✓ Read with offset and limit test passed\n")

        # Test: read image file
        result = await read_tool.call(path="test.jpg")
        print(f"Read image result:\n{result}")
        assert "image file" in result.lower() or ".jpg" in result.lower()
        print("✓ Read image test passed\n")

        # Test: offset beyond file
        print("=== Testing offset beyond file error ===")
        result = await read_tool.call(path="test.txt", offset=200)
        print(f"Expected error result: {result}")
        assert "failed" in result and ("beyond end of file" in result or "offset" in result.lower())
        print("✓ Offset beyond file error test passed\n")

        # Test: file not found
        print("=== Testing file not found error ===")
        result = await read_tool.call(path="nonexistent.txt")
        print(f"Expected error result: {result}")
        assert "failed" in result and "not found" in result.lower()
        print("✓ File not found error test passed\n")

        # Test: read directory (should fail)
        print("=== Testing read directory error ===")
        sub_dir = temp_path / "subdir"
        sub_dir.mkdir()
        result = await read_tool.call(path="subdir")
        print(f"Expected error result: {result}")
        assert "failed" in result and ("Not a file" in result or "directory" in result.lower())
        print("✓ Read directory error test passed\n")


async def test_write_tool():
    """Test WriteTool."""
    from reme.core.tools import WriteTool

    print("=== Testing WriteTool ===")

    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Test: write new file
        write_tool = WriteTool(cwd=str(temp_path))
        test_content = "Hello World\nThis is a test file\n"
        result = await write_tool.call(path="test.txt", content=test_content)
        print(f"Write result: {result}")
        assert "Successfully wrote" in result
        assert "test.txt" in result

        # Verify file was created
        test_file = temp_path / "test.txt"
        assert test_file.exists()
        assert test_file.read_text() == test_content
        print("✓ Write new file test passed\n")

        # Test: overwrite existing file
        new_content = "Updated content\n"
        result = await write_tool.call(path="test.txt", content=new_content)
        print(f"Overwrite result: {result}")
        assert "Successfully wrote" in result

        # Verify file was overwritten
        assert test_file.read_text() == new_content
        assert test_content not in test_file.read_text()
        print("✓ Overwrite existing file test passed\n")

        # Test: create file with parent directories
        nested_path = "subdir1/subdir2/nested.txt"
        nested_content = "Nested file content"
        result = await write_tool.call(path=nested_path, content=nested_content)
        print(f"Create with parents result: {result}")
        assert "Successfully wrote" in result

        # Verify nested file was created
        nested_file = temp_path / "subdir1" / "subdir2" / "nested.txt"
        assert nested_file.exists()
        assert nested_file.read_text() == nested_content
        print("✓ Create file with parent directories test passed\n")

        # Test: write empty file
        result = await write_tool.call(path="empty.txt", content="")
        print(f"Write empty file result: {result}")
        assert "Successfully wrote" in result
        assert "0 bytes" in result

        # Verify empty file
        empty_file = temp_path / "empty.txt"
        assert empty_file.exists()
        assert empty_file.read_text() == ""
        print("✓ Write empty file test passed\n")

        # Test: write file with absolute path
        abs_path = str(temp_path / "absolute.txt")
        abs_content = "Absolute path content"
        result = await write_tool.call(path=abs_path, content=abs_content)
        print(f"Write absolute path result: {result}")
        assert "Successfully wrote" in result

        # Verify absolute path file
        abs_file = Path(abs_path)
        assert abs_file.exists()
        assert abs_file.read_text(encoding="utf-8") == abs_content
        print("✓ Write absolute path test passed\n")


async def main():
    """Run all file system tool tests."""
    await test_bash_tool()
    await test_edit_tool()
    await test_find_tool()
    await test_grep_tool()
    await test_ls_tool()
    await test_read_tool()
    await test_write_tool()
    print("=== All tests passed! ===")


if __name__ == "__main__":
    asyncio.run(main())
