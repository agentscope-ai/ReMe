# -*- coding: utf-8 -*-
# pylint: disable=redefined-outer-name
"""Unit tests for Shell and FileIO tools."""

import asyncio
import os
import re
import shutil
import tempfile

import pytest

from reme.memory.file_based.tools.file_io import FileIO
from reme.memory.file_based.tools.shell import Shell
from reme.memory.file_based.utils import DEFAULT_MAX_BYTES


# ============ Shell Tests ============


@pytest.fixture(scope="module")
def shell_env():
    """Create temporary directory and Shell instance."""
    test_dir = tempfile.mkdtemp(prefix="test_shell_")
    shell = Shell(working_dir=test_dir)
    yield {"dir": test_dir, "shell": shell}
    shutil.rmtree(test_dir, ignore_errors=True)


def test_shell_echo_success(shell_env):
    """Test successful echo command execution."""
    result = asyncio.run(shell_env["shell"].execute_shell_command("echo hello"))
    assert result.content
    text = result.content[0].get("text", "")
    assert "hello" in text


def test_shell_pwd_in_working_dir(shell_env):
    """Test command executes in correct working directory."""
    result = asyncio.run(shell_env["shell"].execute_shell_command("pwd"))
    text = result.content[0].get("text", "")
    assert shell_env["dir"] in text


def test_shell_command_failure(shell_env):
    """Test failed command returns error information."""
    result = asyncio.run(shell_env["shell"].execute_shell_command("exit 1"))
    text = result.content[0].get("text", "")
    assert "failed" in text.lower()
    assert "exit code" in text.lower()


def test_shell_command_with_stderr(shell_env):
    """Test command with stderr output."""
    result = asyncio.run(
        shell_env["shell"].execute_shell_command("echo error >&2 && exit 1"),
    )
    text = result.content[0].get("text", "")
    assert "error" in text


def test_shell_no_output(shell_env):
    """Test successful command with no output."""
    result = asyncio.run(shell_env["shell"].execute_shell_command("true"))
    text = result.content[0].get("text", "")
    assert "successfully" in text.lower()


def test_shell_multiline_output(shell_env):
    """Test command with multiline output."""
    result = asyncio.run(
        shell_env["shell"].execute_shell_command("echo -e 'line1\nline2\nline3'"),
    )
    text = result.content[0].get("text", "")
    assert "line1" in text
    assert "line2" in text
    assert "line3" in text


def test_shell_timeout(shell_env):
    """Test command timeout handling."""
    result = asyncio.run(
        shell_env["shell"].execute_shell_command("sleep 10", timeout=1),
    )
    text = result.content[0].get("text", "")
    assert "timeout" in text.lower()


# ============ FileIO Read Tests ============


@pytest.fixture(scope="module")
def fileio_env():
    """Create temporary directory with test files."""
    test_dir = tempfile.mkdtemp(prefix="test_fileio_")
    file_io = FileIO(working_dir=test_dir)

    # Create simple test file
    simple_file = os.path.join(test_dir, "simple.txt")
    with open(simple_file, "w", encoding="utf-8") as f:
        f.write("line1\nline2\nline3\nline4\nline5")

    # Create large file (exceeds DEFAULT_MAX_BYTES)
    large_file = os.path.join(test_dir, "large.txt")
    with open(large_file, "w", encoding="utf-8") as f:
        # Each line is ~7-10 bytes ("line N\n"); generate enough to exceed limit.
        # Line 1 is literally "line 1" so the head-kept assertion can match it.
        line_count = (DEFAULT_MAX_BYTES // 7) + 1000
        for i in range(1, line_count + 1):
            f.write(f"line {i}\n")

    # Create large bytes file (exceeds DEFAULT_MAX_BYTES)
    # Lines are 101 bytes each; at DEFAULT_MAX_BYTES the cut lands mid-line → else branch
    large_bytes_file = os.path.join(test_dir, "large_bytes.txt")
    with open(large_bytes_file, "w", encoding="utf-8") as f:
        content = "x" * 100 + "\n"
        lines_needed = (DEFAULT_MAX_BYTES // 101) + 100
        for _ in range(lines_needed):
            f.write(content)

    # Single line larger than DEFAULT_MAX_BYTES → newline_count==0 branch in truncate
    huge_line_file = os.path.join(test_dir, "huge_line.txt")
    with open(huge_line_file, "w", encoding="utf-8") as f:
        f.write("A" * (DEFAULT_MAX_BYTES + 1000) + "\nline2\n")

    # Empty file
    empty_file = os.path.join(test_dir, "empty.txt")
    with open(empty_file, "w", encoding="utf-8") as f:
        f.write("")

    yield {
        "dir": test_dir,
        "file_io": file_io,
        "simple_file": simple_file,
        "large_file": large_file,
        "large_bytes_file": large_bytes_file,
        "huge_line_file": huge_line_file,
        "empty_file": empty_file,
    }
    shutil.rmtree(test_dir, ignore_errors=True)


def test_read_file_success(fileio_env):
    """Test successful file reading."""
    result = asyncio.run(fileio_env["file_io"].read_file(fileio_env["simple_file"]))
    text = result.content[0].get("text", "")
    assert "line1" in text
    assert "line5" in text


def test_read_file_relative_path(fileio_env):
    """Test reading file with relative path."""
    result = asyncio.run(fileio_env["file_io"].read_file("simple.txt"))
    text = result.content[0].get("text", "")
    assert "line1" in text


def test_read_file_not_exists(fileio_env):
    """Test reading non-existent file."""
    result = asyncio.run(fileio_env["file_io"].read_file("nonexistent.txt"))
    text = result.content[0].get("text", "")
    assert "Error" in text
    assert "does not exist" in text


def test_read_file_with_line_range(fileio_env):
    """Test reading specific line range."""
    result = asyncio.run(
        fileio_env["file_io"].read_file(fileio_env["simple_file"], start_line=2, end_line=4),
    )
    text = result.content[0].get("text", "")
    assert "line2" in text
    assert "line4" in text
    assert "lines 2-4" in text.lower()


def test_read_file_start_line_exceeds(fileio_env):
    """Test start_line exceeding file length."""
    result = asyncio.run(
        fileio_env["file_io"].read_file(fileio_env["simple_file"], start_line=100),
    )
    text = result.content[0].get("text", "")
    assert "Error" in text
    assert "exceeds" in text


def test_read_file_invalid_range(fileio_env):
    """Test invalid line range (start > end)."""
    result = asyncio.run(
        fileio_env["file_io"].read_file(fileio_env["simple_file"], start_line=4, end_line=2),
    )
    text = result.content[0].get("text", "")
    assert "Error" in text


def test_read_file_truncated(fileio_env):
    """Test file truncation by byte limit."""
    result = asyncio.run(fileio_env["file_io"].read_file(fileio_env["large_file"]))
    text = result.content[0].get("text", "")
    assert "line 1" in text  # Head is kept
    assert "continue" in text.lower()


def test_read_file_truncated_by_bytes(fileio_env):
    """Test file truncation by byte limit."""
    result = asyncio.run(fileio_env["file_io"].read_file(fileio_env["large_bytes_file"]))
    text = result.content[0].get("text", "")
    assert "continue" in text.lower()
    assert "KB limit" in text


def test_read_directory_error(fileio_env):
    """Test reading a directory returns error."""
    result = asyncio.run(fileio_env["file_io"].read_file(fileio_env["dir"]))
    text = result.content[0].get("text", "")
    assert "Error" in text
    assert "not a file" in text


def test_read_file_single_line_range(fileio_env):
    """Test reading exactly one line (start_line == end_line)."""
    result = asyncio.run(
        fileio_env["file_io"].read_file(fileio_env["simple_file"], start_line=3, end_line=3),
    )
    text = result.content[0].get("text", "")
    assert "line3" in text
    assert "line2" not in text
    assert "line4" not in text


def test_read_file_only_start_line(fileio_env):
    """Test reading from start_line to end of file (no end_line)."""
    result = asyncio.run(
        fileio_env["file_io"].read_file(fileio_env["simple_file"], start_line=4),
    )
    text = result.content[0].get("text", "")
    assert "line4" in text
    assert "line5" in text
    assert "line1" not in text
    assert "line3" not in text


def test_read_file_only_end_line(fileio_env):
    """Test reading from beginning to end_line (no start_line)."""
    result = asyncio.run(
        fileio_env["file_io"].read_file(fileio_env["simple_file"], end_line=2),
    )
    text = result.content[0].get("text", "")
    assert "line1" in text
    assert "line2" in text
    assert "line4" not in text
    assert "line5" not in text


def test_read_file_end_line_clamped(fileio_env):
    """Test end_line beyond total lines is silently clamped to file end."""
    result = asyncio.run(
        fileio_env["file_io"].read_file(fileio_env["simple_file"], start_line=1, end_line=999),
    )
    text = result.content[0].get("text", "")
    assert "Error" not in text
    assert "line1" in text
    assert "line5" in text


def test_read_file_continuation_hint(fileio_env):
    """Partial range read without truncation shows remaining-lines continuation hint."""
    # simple.txt has 5 lines; reading 1-3 leaves 2 more
    result = asyncio.run(
        fileio_env["file_io"].read_file(fileio_env["simple_file"], start_line=1, end_line=3),
    )
    text = result.content[0].get("text", "")
    assert "more lines" in text
    assert "start_line=4" in text


def test_read_file_truncated_next_line_hint(fileio_env):
    """Truncated large file provides a valid start_line > 1 to continue."""
    result = asyncio.run(fileio_env["file_io"].read_file(fileio_env["large_file"]))
    text = result.content[0].get("text", "")
    match = re.search(r"start_line=(\d+)", text)
    assert match is not None, "Expected start_line hint in truncated output"
    assert int(match.group(1)) > 1


def test_read_file_truncated_mid_line_message(fileio_env):
    """Truncation mid-line reports which line is truncated (else branch)."""
    # large_bytes_file lines are 101 bytes; truncation lands mid-line
    result = asyncio.run(fileio_env["file_io"].read_file(fileio_env["large_bytes_file"]))
    text = result.content[0].get("text", "")
    assert "is truncated" in text.lower()


def test_read_file_huge_single_line(fileio_env):
    """Single line exceeding byte limit triggers 'partially shown' notice (newline_count==0 branch)."""
    result = asyncio.run(fileio_env["file_io"].read_file(fileio_env["huge_line_file"]))
    text = result.content[0].get("text", "")
    assert "partially shown" in text.lower()
    assert "start_line=2" in text


def test_read_file_invalid_start_line_type(fileio_env):
    """Non-integer start_line returns a descriptive error."""
    result = asyncio.run(
        fileio_env["file_io"].read_file(fileio_env["simple_file"], start_line="abc"),
    )
    text = result.content[0].get("text", "")
    assert "Error" in text
    assert "start_line" in text


def test_read_file_invalid_end_line_type(fileio_env):
    """Non-integer end_line returns a descriptive error."""
    result = asyncio.run(
        fileio_env["file_io"].read_file(fileio_env["simple_file"], end_line="xyz"),
    )
    text = result.content[0].get("text", "")
    assert "Error" in text
    assert "end_line" in text


def test_read_file_start_line_as_string(fileio_env):
    """Numeric-string start_line/end_line are coerced to int successfully."""
    result = asyncio.run(
        fileio_env["file_io"].read_file(fileio_env["simple_file"], start_line="2", end_line="4"),
    )
    text = result.content[0].get("text", "")
    assert "Error" not in text
    assert "line2" in text
    assert "line4" in text


def test_read_file_empty(fileio_env):
    """Reading an empty file returns without error."""
    result = asyncio.run(fileio_env["file_io"].read_file(fileio_env["empty_file"]))
    text = result.content[0].get("text", "")
    assert "Error" not in text


# ============ FileIO Write Tests ============


@pytest.fixture
def write_env():
    """Create temporary directory for write tests."""
    test_dir = tempfile.mkdtemp(prefix="test_fileio_write_")
    file_io = FileIO(working_dir=test_dir)
    yield {"dir": test_dir, "file_io": file_io}
    shutil.rmtree(test_dir, ignore_errors=True)


def test_write_new_file(write_env):
    """Test writing a new file."""
    file_path = os.path.join(write_env["dir"], "new_file.txt")
    result = asyncio.run(write_env["file_io"].write_file(file_path, "test content"))
    text = result.content[0].get("text", "")
    assert "Wrote" in text

    with open(file_path, "r", encoding="utf-8") as f:
        assert f.read() == "test content"


def test_write_overwrite_file(write_env):
    """Test overwriting existing file."""
    file_path = os.path.join(write_env["dir"], "overwrite.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("old content")

    result = asyncio.run(write_env["file_io"].write_file(file_path, "new content"))
    text = result.content[0].get("text", "")
    assert "Wrote" in text

    with open(file_path, "r", encoding="utf-8") as f:
        assert f.read() == "new content"


def test_write_empty_path(write_env):
    """Test writing with empty path."""
    result = asyncio.run(write_env["file_io"].write_file("", "content"))
    text = result.content[0].get("text", "")
    assert "Error" in text


def test_write_relative_path(write_env):
    """Test writing file with relative path."""
    result = asyncio.run(write_env["file_io"].write_file("relative.txt", "relative content"))
    text = result.content[0].get("text", "")
    assert "Wrote" in text

    file_path = os.path.join(write_env["dir"], "relative.txt")
    assert os.path.exists(file_path)


# ============ FileIO Edit Tests ============


@pytest.fixture
def edit_env():
    """Create temporary directory with edit test file."""
    test_dir = tempfile.mkdtemp(prefix="test_fileio_edit_")
    file_io = FileIO(working_dir=test_dir)

    edit_file = os.path.join(test_dir, "edit_test.txt")
    with open(edit_file, "w", encoding="utf-8") as f:
        f.write("Hello World\nThis is a test\nHello Again")

    yield {"dir": test_dir, "file_io": file_io, "edit_file": edit_file}
    shutil.rmtree(test_dir, ignore_errors=True)


def test_edit_replace_text(edit_env):
    """Test replacing text in file."""
    result = asyncio.run(
        edit_env["file_io"].edit_file(edit_env["edit_file"], "Hello", "Hi"),
    )
    text = result.content[0].get("text", "")
    assert "Successfully" in text

    with open(edit_env["edit_file"], "r", encoding="utf-8") as f:
        content = f.read()
    assert "Hello" not in content
    assert "Hi World" in content
    assert "Hi Again" in content


def test_edit_text_not_found(edit_env):
    """Test editing when text not found."""
    result = asyncio.run(
        edit_env["file_io"].edit_file(edit_env["edit_file"], "NotExists", "Replacement"),
    )
    text = result.content[0].get("text", "")
    assert "Error" in text
    assert "not found" in text


def test_edit_nonexistent_file(edit_env):
    """Test editing non-existent file."""
    result = asyncio.run(
        edit_env["file_io"].edit_file("nonexistent.txt", "old", "new"),
    )
    text = result.content[0].get("text", "")
    assert "Error" in text
