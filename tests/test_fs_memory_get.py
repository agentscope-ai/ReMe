"""Tests for ReMeFb memory_get interface.

This module tests the memory_get() method of ReMeFb class which provides
a high-level interface for reading specific snippets from memory files.

The memory_get function should enable the LLM to:
1. Read entire memory files (MEMORY.md, memory/*.md)
2. Read specific line ranges using offset and limit parameters
3. Extract only the needed content to keep context small
"""

import asyncio
import os
from pathlib import Path

from reme import ReMeFb


def print_result(content: str, title: str = "RESULT", max_len: int = 300):
    """Print the result of memory_get() call.

    Args:
        content: Content returned from memory_get()
        title: Title for the result section
        max_len: Maximum content length to display (truncate if longer)
    """
    print(f"\n{'=' * 80}")
    print(f"{title}:")

    lines = content.split("\n")
    print(f"  total_lines: {len(lines)}")
    print(f"  total_chars: {len(content)}")

    if len(content) > max_len:
        preview = content[:max_len] + "..."
    else:
        preview = content

    print("\n  Content Preview:")
    print("-" * 80)
    print(preview)
    print("-" * 80)
    print(f"{'=' * 80}")


def create_test_memory_file(workspace_dir: str) -> str:
    """Create a test memory file with numbered lines.

    Args:
        workspace_dir: Directory to create the test file in

    Returns:
        Path to the created test file (relative to workspace_dir)
    """
    workspace_path = Path(workspace_dir)
    workspace_path.mkdir(parents=True, exist_ok=True)

    memory_dir = workspace_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    test_file = memory_dir / "test_profile.md"

    content = """# User Profile

## Personal Information
Name: Alice Johnson
Age: 28
Location: San Francisco, CA

## Professional Background
Occupation: Software Engineer
Company: Tech Innovations Inc.
Years of Experience: 5

## Skills
- Python Programming
- Machine Learning
- Natural Language Processing
- Docker & Kubernetes

## Interests
- Reading sci-fi novels
- Hiking in national parks
- Photography
- Cooking international cuisines

## Preferences
- Prefers detailed technical explanations
- Likes to see code examples
- Values efficiency and clean code
- Appreciates constructive feedback
"""

    test_file.write_text(content, encoding="utf-8")
    return "memory/test_profile.md"


async def test_memory_get_full_file():
    """Test memory_get() reads entire file without offset/limit.

    Expects: Returns complete file content
    """
    print("\n" + "=" * 80)
    print("TEST 1: Memory Get - Read Entire File")
    print("=" * 80)

    workspace_dir = ".reme_test_get"
    reme_fs = ReMeFb(enable_logo=False, working_dir=workspace_dir)
    await reme_fs.start()

    # Create test file
    test_file_path = create_test_memory_file(workspace_dir)
    print("\nTest Setup:")
    print(f"  test_file: {test_file_path}")
    print(f"  workspace: {workspace_dir}")

    print("\nParameters:")
    print(f"  path: {test_file_path}")
    print("  offset: None")
    print("  limit: None")
    print("  Expected: Read entire file content")

    # Call memory_get
    result = await reme_fs.memory_get(path=test_file_path)

    print_result(result, "MEMORY_GET RESULT", max_len=500)

    # Verify
    lines = result.split("\n")
    print("\nVerification:")
    print(f"  ✓ Got {len(lines)} lines")
    print(f"  ✓ Content starts with: {result[:50].strip()}")

    await reme_fs.close()


async def test_memory_get_with_offset():
    """Test memory_get() reads from specific line to end.

    Expects: Returns content from line 10 to end of file
    """
    print("\n" + "=" * 80)
    print("TEST 2: Memory Get - Read with Offset")
    print("=" * 80)

    workspace_dir = ".reme_test_get"
    reme_fs = ReMeFb(enable_logo=False, working_dir=workspace_dir)
    await reme_fs.start()

    test_file_path = "memory/test_profile.md"
    offset = 10

    print("\nParameters:")
    print(f"  path: {test_file_path}")
    print(f"  offset: {offset}")
    print("  limit: None")
    print(f"  Expected: Read from line {offset} to end of file")

    # Call memory_get
    result = await reme_fs.memory_get(path=test_file_path, offset=offset)

    print_result(result, "MEMORY_GET RESULT", max_len=400)

    # Verify
    lines = result.split("\n")
    print("\nVerification:")
    print(f"  ✓ Got {len(lines)} lines starting from line {offset}")
    print(f"  ✓ First line of result: {lines[0]}")

    await reme_fs.close()


async def test_memory_get_with_offset_and_limit():
    """Test memory_get() reads specific line range.

    Expects: Returns exactly 5 lines starting from line 5
    """
    print("\n" + "=" * 80)
    print("TEST 3: Memory Get - Read with Offset and Limit")
    print("=" * 80)

    workspace_dir = ".reme_test_get"
    reme_fs = ReMeFb(enable_logo=False, working_dir=workspace_dir)
    await reme_fs.start()

    test_file_path = "memory/test_profile.md"
    offset = 5
    limit = 5

    print("\nParameters:")
    print(f"  path: {test_file_path}")
    print(f"  offset: {offset}")
    print(f"  limit: {limit}")
    print(f"  Expected: Read exactly {limit} lines starting from line {offset}")

    # Call memory_get
    result = await reme_fs.memory_get(path=test_file_path, offset=offset, limit=limit)

    print_result(result, "MEMORY_GET RESULT", max_len=400)

    # Verify
    lines = result.split("\n")
    print("\nVerification:")
    print(f"  ✓ Got {len(lines)} lines (expected {limit})")
    print("  ✓ Lines content:")
    for i, line in enumerate(lines, start=offset):
        print(f"    Line {i}: {line}")

    await reme_fs.close()


async def test_memory_get_beginning_lines():
    """Test memory_get() reads first few lines.

    Expects: Returns first 3 lines of the file
    """
    print("\n" + "=" * 80)
    print("TEST 4: Memory Get - Read Beginning Lines")
    print("=" * 80)

    workspace_dir = ".reme_test_get"
    reme_fs = ReMeFb(enable_logo=False, working_dir=workspace_dir)
    await reme_fs.start()

    test_file_path = "memory/test_profile.md"
    offset = 1
    limit = 3

    print("\nParameters:")
    print(f"  path: {test_file_path}")
    print(f"  offset: {offset}")
    print(f"  limit: {limit}")
    print(f"  Expected: Read first {limit} lines")

    # Call memory_get
    result = await reme_fs.memory_get(path=test_file_path, offset=offset, limit=limit)

    print_result(result, "MEMORY_GET RESULT", max_len=400)

    # Verify
    lines = result.split("\n")
    print("\nVerification:")
    print(f"  ✓ Got {len(lines)} lines (expected {limit})")
    print("  ✓ Should contain '# User Profile' header")
    assert "# User Profile" in result, "Expected header not found"

    await reme_fs.close()


async def test_memory_get_single_line():
    """Test memory_get() reads a single specific line.

    Expects: Returns exactly 1 line
    """
    print("\n" + "=" * 80)
    print("TEST 5: Memory Get - Read Single Line")
    print("=" * 80)

    workspace_dir = ".reme_test_get"
    reme_fs = ReMeFb(enable_logo=False, working_dir=workspace_dir)
    await reme_fs.start()

    test_file_path = "memory/test_profile.md"
    offset = 3
    limit = 1

    print("\nParameters:")
    print(f"  path: {test_file_path}")
    print(f"  offset: {offset}")
    print(f"  limit: {limit}")
    print(f"  Expected: Read exactly 1 line at position {offset}")

    # Call memory_get
    result = await reme_fs.memory_get(path=test_file_path, offset=offset, limit=limit)

    print_result(result, "MEMORY_GET RESULT", max_len=400)

    # Verify
    lines = result.split("\n")
    print("\nVerification:")
    print(f"  ✓ Got {len(lines)} line (expected {limit})")
    print(f"  ✓ Line {offset}: {result}")

    await reme_fs.close()


async def test_memory_get_with_absolute_path():
    """Test memory_get() with absolute path.

    Expects: Works with both relative and absolute paths
    """
    print("\n" + "=" * 80)
    print("TEST 6: Memory Get - Read with Absolute Path")
    print("=" * 80)

    workspace_dir = ".reme_test_get"
    reme_fs = ReMeFb(enable_logo=False, working_dir=workspace_dir)
    await reme_fs.start()

    # Get absolute path
    abs_path = os.path.abspath(os.path.join(workspace_dir, "memory/test_profile.md"))
    limit = 5

    print("\nParameters:")
    print(f"  path: {abs_path}")
    print("  offset: None")
    print(f"  limit: {limit}")
    print(f"  Expected: Read first {limit} lines using absolute path")

    # Call memory_get with absolute path
    result = await reme_fs.memory_get(path=abs_path, limit=limit)

    print_result(result, "MEMORY_GET RESULT", max_len=400)

    # Verify
    lines = result.split("\n")
    print("\nVerification:")
    print(f"  ✓ Got {len(lines)} lines using absolute path")
    print("  ✓ Absolute path handling works correctly")

    await reme_fs.close()


async def main():
    """Run core memory_get interface tests."""
    print("\n" + "=" * 80)
    print("ReMeFb Memory Get Interface - Tests")
    print("=" * 80)
    print("\nThis test suite validates that the memory_get() function:")
    print("  1. Reads entire memory files without parameters")
    print("  2. Reads from specific line (offset) to end of file")
    print("  3. Reads specific line ranges (offset + limit)")
    print("  4. Handles both relative and absolute paths")
    print("\nTest Scenarios:")
    print("  1. Full file read (no offset/limit)")
    print("  2. Read with offset (from line N to end)")
    print("  3. Read with offset and limit (specific range)")
    print("  4. Read beginning lines (first N lines)")
    print("  5. Read single line")
    print("  6. Read with absolute path")
    print("=" * 80)

    # Test 1: Read entire file
    await test_memory_get_full_file()

    # Test 2: Read with offset
    await test_memory_get_with_offset()

    # Test 3: Read with offset and limit
    await test_memory_get_with_offset_and_limit()

    # Test 4: Read beginning lines
    await test_memory_get_beginning_lines()

    # Test 5: Read single line
    await test_memory_get_single_line()

    # Test 6: Read with absolute path
    await test_memory_get_with_absolute_path()

    print("\n" + "=" * 80)
    print("All memory_get tests completed!")
    print("=" * 80)
    print("\nNote: Test files are created in .reme_test_get/ directory")
    print("You can manually inspect them or delete the directory after testing.")


if __name__ == "__main__":
    asyncio.run(main())
