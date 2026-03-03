"""Tests for ToolResultCompactor."""

import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from agentscope.message import Msg

from reme.memory.file_based_copaw.tool_result_compactor import ToolResultCompactor
from reme.memory.file_based_copaw.utils import TRUNCATION_MARKER_START


def create_tool_result_msg(output: str | list, tool_name: str = "test_tool") -> Msg:
    """Create a Msg with tool_result content block."""
    return Msg(
        name="tool",
        role="user",
        content=[
            {
                "type": "tool_result",
                "id": "call_123",
                "name": tool_name,
                "output": output,
            },
        ],
    )


class TestToolResultCompactor:
    """Tests for ToolResultCompactor."""

    def test_no_truncation_when_under_threshold(self):
        """Test that short content is not truncated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            op = ToolResultCompactor(tool_result_dir=tmpdir, tool_result_threshold=1000)
            messages = [create_tool_result_msg("short content")]

            result = asyncio.run(op.call(messages=messages))

            assert result == messages
            assert messages[0].content[0]["output"] == "short content"
            assert len(list(Path(tmpdir).glob("*.txt"))) == 0

    def test_truncation_when_over_threshold(self):
        """Test that long content is truncated and saved to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            op = ToolResultCompactor(tool_result_dir=tmpdir, tool_result_threshold=100)
            long_content = "x" * 500
            messages = [create_tool_result_msg(long_content)]

            _ = asyncio.run(op.call(messages=messages))

            output = messages[0].content[0]["output"]
            assert TRUNCATION_MARKER_START in output
            assert "[Full content saved to:" in output

            # Verify file was created
            files = list(Path(tmpdir).glob("*.txt"))
            assert len(files) == 1

            # Verify file content
            content = files[0].read_text()
            assert "# tool_name: test_tool" in content
            assert "# created_at:" in content
            assert long_content in content

    def test_skip_already_truncated(self):
        """Test that already truncated content is not re-truncated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            op = ToolResultCompactor(tool_result_dir=tmpdir, tool_result_threshold=100)
            truncated_content = f"head{TRUNCATION_MARKER_START}(100 chars omitted)<<<END_TRUNCATED>>>tail"
            messages = [create_tool_result_msg(truncated_content)]

            asyncio.run(op.call(messages=messages))

            assert messages[0].content[0]["output"] == truncated_content
            assert len(list(Path(tmpdir).glob("*.txt"))) == 0

    def test_truncation_list_output(self):
        """Test truncation of list output with text blocks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            op = ToolResultCompactor(tool_result_dir=tmpdir, tool_result_threshold=100)
            list_output = [{"type": "text", "text": "y" * 500}]
            messages = [create_tool_result_msg(list_output)]

            asyncio.run(op.call(messages=messages))

            text_block = messages[0].content[0]["output"][0]
            assert TRUNCATION_MARKER_START in text_block["text"]
            assert len(list(Path(tmpdir).glob("*.txt"))) == 1

    def test_list_output_no_truncation_when_short(self):
        """Test that short list output is not truncated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            op = ToolResultCompactor(tool_result_dir=tmpdir, tool_result_threshold=1000)
            list_output = [{"type": "text", "text": "short"}]
            messages = [create_tool_result_msg(list_output)]

            asyncio.run(op.call(messages=messages))

            assert messages[0].content[0]["output"][0]["text"] == "short"
            assert len(list(Path(tmpdir).glob("*.txt"))) == 0

    def test_list_output_multiple_text_blocks(self):
        """Test truncation of multiple text blocks in list output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            op = ToolResultCompactor(tool_result_dir=tmpdir, tool_result_threshold=100)
            list_output = [
                {"type": "text", "text": "a" * 500},
                {"type": "text", "text": "short"},
                {"type": "text", "text": "b" * 500},
            ]
            messages = [create_tool_result_msg(list_output)]

            asyncio.run(op.call(messages=messages))

            output = messages[0].content[0]["output"]
            assert TRUNCATION_MARKER_START in output[0]["text"]
            assert output[1]["text"] == "short"  # unchanged
            assert TRUNCATION_MARKER_START in output[2]["text"]
            assert len(list(Path(tmpdir).glob("*.txt"))) == 2

    def test_list_output_mixed_block_types(self):
        """Test that non-text blocks in list output are unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            op = ToolResultCompactor(tool_result_dir=tmpdir, tool_result_threshold=100)
            list_output = [
                {"type": "text", "text": "c" * 500},
                {"type": "image", "source": {"type": "url", "url": "http://example.com/img.png"}},
            ]
            messages = [create_tool_result_msg(list_output)]

            asyncio.run(op.call(messages=messages))

            output = messages[0].content[0]["output"]
            assert TRUNCATION_MARKER_START in output[0]["text"]
            assert output[1] == {"type": "image", "source": {"type": "url", "url": "http://example.com/img.png"}}
            assert len(list(Path(tmpdir).glob("*.txt"))) == 1

    def test_cleanup_expired_files(self):
        """Test cleanup of expired files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            op = ToolResultCompactor(tool_result_dir=tmpdir, tool_result_threshold=100, retention_days=1)

            # Create an old file
            old_time = (datetime.now() - timedelta(days=2)).isoformat()
            old_file = Path(tmpdir) / "old_file.txt"
            old_file.write_text(f"# tool_name: test\n# created_at: {old_time}\n# ---\ncontent")

            # Create a new file
            new_time = datetime.now().isoformat()
            new_file = Path(tmpdir) / "new_file.txt"
            new_file.write_text(f"# tool_name: test\n# created_at: {new_time}\n# ---\ncontent")

            deleted = op.cleanup_expired_files()

            assert deleted == 1
            assert not old_file.exists()
            assert new_file.exists()

    def test_string_content_msg_unchanged(self):
        """Test that messages with string content are unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            op = ToolResultCompactor(tool_result_dir=tmpdir, tool_result_threshold=100)
            messages = [Msg(name="user", role="user", content="hello world")]

            asyncio.run(op.call(messages=messages))

            assert messages[0].content == "hello world"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
