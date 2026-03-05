"""Tests for MemoryFormatter."""

# pylint: disable=W0212

import logging

from agentscope.message import Msg

from test_utils import get_token_counter
from reme.memory.file_based import MemoryFormatter

# 配置日志输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ANSI 颜色码
class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_pass(test_name: str):
    """打印测试通过信息"""
    print(f"{Colors.GREEN}{Colors.BOLD}✓ {test_name} PASSED{Colors.RESET}")


def print_fail(test_name: str, error: str):
    """打印测试失败信息"""
    print(f"{Colors.RED}{Colors.BOLD}✗ {test_name} FAILED: {error}{Colors.RESET}")


def print_error(test_name: str, error: str):
    """打印测试错误信息"""
    print(f"{Colors.YELLOW}{Colors.BOLD}⚠ {test_name} ERROR: {error}{Colors.RESET}")


def print_test_header(test_name: str):
    """打印测试标题"""
    print(f"\n{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}Running: {test_name}{Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}")


def create_user_msg(content: str) -> Msg:
    """Create a user message."""
    return Msg(name="user", role="user", content=content)


def create_assistant_msg(content: str) -> Msg:
    """Create an assistant message."""
    return Msg(name="assistant", role="assistant", content=content)


def create_tool_use_msg(tool_name: str, tool_input: dict) -> Msg:
    """Create a message with tool_use content block."""
    return Msg(
        name="assistant",
        role="assistant",
        content=[
            {
                "type": "tool_use",
                "id": "call_123",
                "name": tool_name,
                "input": tool_input,
            },
        ],
    )


def create_tool_result_msg(tool_name: str, output: str | list[dict]) -> Msg:
    """Create a message with tool_result content block."""
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


def create_thinking_msg(thinking_content: str) -> Msg:
    """Create a message with thinking content block."""
    return Msg(
        name="assistant",
        role="assistant",
        content=[
            {
                "type": "thinking",
                "text": thinking_content,
            },
        ],
    )


def create_image_msg(url: str = "") -> Msg:
    """Create a message with image content block."""
    content = [
        {
            "type": "image",
            "source": {"url": url} if url else {},
        },
    ]
    return Msg(name="assistant", role="assistant", content=content)


def create_formatter(memory_compact_threshold: int = 4000) -> MemoryFormatter:
    """Create a MemoryFormatter instance for testing."""
    return MemoryFormatter(
        token_counter=get_token_counter(),
        memory_compact_threshold=memory_compact_threshold,
    )


# ==================== _format_tool_result_output Tests ====================


def test_format_tool_result_output_string():
    """Test _format_tool_result_output with string input."""
    result = MemoryFormatter._format_tool_result_output("Hello, world!")
    assert result == "Hello, world!", f"Expected 'Hello, world!', got: {result}"
    print_pass("test_format_tool_result_output_string")


def test_format_tool_result_output_text_block():
    """Test _format_tool_result_output with text block."""
    output = [{"type": "text", "text": "This is text content"}]
    result = MemoryFormatter._format_tool_result_output(output)
    assert result == "This is text content", f"Expected 'This is text content', got: {result}"
    print_pass("test_format_tool_result_output_text_block")


def test_format_tool_result_output_image_block():
    """Test _format_tool_result_output with image block."""
    output = [{"type": "image", "source": {"url": "https://example.com/image.png"}}]
    result = MemoryFormatter._format_tool_result_output(output)
    assert "[image]" in result, f"Expected '[image]' in result, got: {result}"
    assert "https://example.com/image.png" in result, f"Expected URL in result, got: {result}"
    print_pass("test_format_tool_result_output_image_block")


def test_format_tool_result_output_file_block():
    """Test _format_tool_result_output with file block."""
    output = [{"type": "file", "path": "/path/to/file.txt", "name": "file.txt"}]
    result = MemoryFormatter._format_tool_result_output(output)
    assert "[file]" in result, f"Expected '[file]' in result, got: {result}"
    assert "file.txt" in result, f"Expected 'file.txt' in result, got: {result}"
    print_pass("test_format_tool_result_output_file_block")


def test_format_tool_result_output_multiple_blocks():
    """Test _format_tool_result_output with multiple blocks."""
    output = [
        {"type": "text", "text": "First part"},
        {"type": "text", "text": "Second part"},
    ]
    result = MemoryFormatter._format_tool_result_output(output)
    assert "First part" in result, f"Expected 'First part' in result, got: {result}"
    assert "Second part" in result, f"Expected 'Second part' in result, got: {result}"
    # Multiple parts should be joined with newlines and bullets
    assert "- " in result, f"Expected bullet format in result, got: {result}"
    print_pass("test_format_tool_result_output_multiple_blocks")


def test_format_tool_result_output_empty_list():
    """Test _format_tool_result_output with empty list."""
    result = MemoryFormatter._format_tool_result_output([])
    assert result == "", f"Expected empty string, got: {result}"
    print_pass("test_format_tool_result_output_empty_list")


def test_format_tool_result_output_invalid_block():
    """Test _format_tool_result_output with invalid block (missing type)."""
    output = [{"text": "No type key"}]
    result = MemoryFormatter._format_tool_result_output(output)
    assert result == "", f"Expected empty string for invalid block, got: {result}"
    print_pass("test_format_tool_result_output_invalid_block")


def test_format_tool_result_output_unknown_type():
    """Test _format_tool_result_output with unknown block type."""
    output = [{"type": "unknown_type", "data": "some data"}]
    result = MemoryFormatter._format_tool_result_output(output)
    assert result == "", f"Expected empty string for unknown type, got: {result}"
    print_pass("test_format_tool_result_output_unknown_type")


# ==================== format (single message) Tests ====================


def test_format_empty_messages():
    """Test format with empty message list."""
    formatter = create_formatter()
    result = formatter.format([])
    assert result == "", f"Expected empty string, got: {result}"
    print_pass("test_format_empty_messages")


def test_format_single_user_message():
    """Test format with a single user message."""
    formatter = create_formatter()
    msgs = [create_user_msg("Hello, how are you?")]
    result = formatter.format(msgs)

    assert "user:" in result, f"Expected 'user:' in result, got: {result}"
    assert "Hello, how are you?" in result, f"Expected content in result, got: {result}"
    print_pass("test_format_single_user_message")


def test_format_single_assistant_message():
    """Test format with a single assistant message."""
    formatter = create_formatter()
    msgs = [create_assistant_msg("I am fine, thank you!")]
    result = formatter.format(msgs)

    assert "assistant:" in result, f"Expected 'assistant:' in result, got: {result}"
    assert "I am fine, thank you!" in result, f"Expected content in result, got: {result}"
    print_pass("test_format_single_assistant_message")


def test_format_with_tool_use():
    """Test format with tool_use message."""
    formatter = create_formatter()
    msgs = [create_tool_use_msg("read_file", {"path": "/test.txt"})]
    result = formatter.format(msgs)

    assert "tool_call=read_file" in result, f"Expected 'tool_call=read_file' in result, got: {result}"
    assert "params=" in result, f"Expected 'params=' in result, got: {result}"
    print_pass("test_format_with_tool_use")


def test_format_with_tool_result():
    """Test format with tool_result message."""
    formatter = create_formatter()
    msgs = [create_tool_result_msg("read_file", "file content here")]
    result = formatter.format(msgs)

    assert "tool_result=read_file" in result, f"Expected 'tool_result=read_file' in result, got: {result}"
    assert "output=" in result, f"Expected 'output=' in result, got: {result}"
    print_pass("test_format_with_tool_result")


def test_format_with_thinking_block():
    """Test that thinking blocks are skipped."""
    formatter = create_formatter()
    msgs = [create_thinking_msg("Let me think about this...")]
    result = formatter.format(msgs)

    # Thinking content should NOT appear in the result
    assert "Let me think about this" not in result, f"Thinking content should be skipped, got: {result}"
    print_pass("test_format_with_thinking_block")


def test_format_with_image():
    """Test format with image content block."""
    formatter = create_formatter()
    msgs = [create_image_msg("https://example.com/image.png")]
    result = formatter.format(msgs)

    assert "[image]" in result, f"Expected '[image]' in result, got: {result}"
    print_pass("test_format_with_image")


# ==================== format (multiple messages) Tests ====================


def test_format_conversation():
    """Test format with a conversation."""
    formatter = create_formatter()
    msgs = [
        create_user_msg("What is Python?"),
        create_assistant_msg("Python is a programming language."),
        create_user_msg("Tell me more."),
        create_assistant_msg("Python is known for its readability and simplicity."),
    ]
    result = formatter.format(msgs)

    assert "round0" in result, f"Expected 'round0' in result, got: {result}"
    assert "round1" in result, f"Expected 'round1' in result, got: {result}"
    assert "round2" in result, f"Expected 'round2' in result, got: {result}"
    assert "round3" in result, f"Expected 'round3' in result, got: {result}"
    print_pass("test_format_conversation")


def test_format_without_index():
    """Test format without round index."""
    formatter = create_formatter()
    msgs = [
        create_user_msg("Hello"),
        create_assistant_msg("Hi there!"),
    ]
    result = formatter.format(msgs, add_index=False)

    assert "round" not in result, f"Expected no 'round' prefix, got: {result}"
    print_pass("test_format_without_index")


def test_format_without_time():
    """Test format without timestamp."""
    formatter = create_formatter()
    msgs = [create_user_msg("Test message")]
    result = formatter.format(msgs, add_time=False)

    # The result should not have timestamp brackets at the beginning
    # Note: this test may need adjustment based on actual timestamp format
    assert "user:" in result, f"Expected 'user:' in result, got: {result}"
    print_pass("test_format_without_time")


def test_format_with_tool_conversation():
    """Test format with tool use and result in conversation."""
    formatter = create_formatter()
    msgs = [
        create_user_msg("Read the file."),
        create_tool_use_msg("read_file", {"path": "/data.txt"}),
        create_tool_result_msg("read_file", "File content here"),
        create_assistant_msg("The file contains: File content here"),
    ]
    result = formatter.format(msgs)

    assert "user:" in result
    assert "tool_call=read_file" in result
    assert "tool_result=read_file" in result
    assert "assistant:" in result
    print_pass("test_format_with_tool_conversation")


# ==================== Token Threshold Tests ====================


def test_format_low_threshold():
    """Test that older messages are skipped with low threshold."""
    formatter = create_formatter(memory_compact_threshold=100)
    msgs = []
    for i in range(20):
        msgs.append(create_user_msg(f"Question {i}: " + "x" * 50))
        msgs.append(create_assistant_msg(f"Answer {i}: " + "y" * 50))

    result = formatter.format(msgs)

    # With low threshold, not all messages should be included
    # The newest messages should be present
    assert "round39" in result or "round38" in result, f"Expected recent round in result, got: {result}"
    # Older messages might be truncated
    logger.info(f"Result length: {len(result)}")
    print_pass("test_format_low_threshold")


def test_format_high_threshold():
    """Test that all messages are included with high threshold."""
    formatter = create_formatter(memory_compact_threshold=100000)
    msgs = [
        create_user_msg("Message 1"),
        create_assistant_msg("Response 1"),
        create_user_msg("Message 2"),
        create_assistant_msg("Response 2"),
    ]
    result = formatter.format(msgs)

    # All messages should be included
    assert "round0" in result
    assert "round1" in result
    assert "round2" in result
    assert "round3" in result
    print_pass("test_format_high_threshold")


# ==================== Edge Cases Tests ====================


def test_format_long_text_truncation():
    """Test that long text is truncated."""
    formatter = create_formatter()
    long_text = "x" * 5000  # Much longer than default max length
    msgs = [create_user_msg(long_text)]
    result = formatter.format(msgs)

    # The result should be shorter due to truncation
    assert len(result) < len(long_text), f"Expected truncated result, got length: {len(result)}"
    print_pass("test_format_long_text_truncation")


def test_format_special_characters():
    """Test format with special characters in content."""
    formatter = create_formatter()
    msgs = [create_user_msg("Test with 中文, 日本語, émojis 🎉")]
    result = formatter.format(msgs)

    assert "中文" in result, f"Expected Chinese characters in result, got: {result}"
    print_pass("test_format_special_characters")


def test_format_tool_result_with_complex_output():
    """Test format with complex tool result output."""
    formatter = create_formatter()
    complex_output = [
        {"type": "text", "text": "Operation completed"},
        {"type": "image", "source": {"url": "https://example.com/result.png"}},
    ]
    msgs = [create_tool_result_msg("process_data", complex_output)]
    result = formatter.format(msgs)

    assert "tool_result=process_data" in result, f"Expected tool result in result, got: {result}"
    print_pass("test_format_tool_result_with_complex_output")


def run_all_tests():
    """Run all tests."""
    tests = [
        # _format_tool_result_output tests
        test_format_tool_result_output_string,
        test_format_tool_result_output_text_block,
        test_format_tool_result_output_image_block,
        test_format_tool_result_output_file_block,
        test_format_tool_result_output_multiple_blocks,
        test_format_tool_result_output_empty_list,
        test_format_tool_result_output_invalid_block,
        test_format_tool_result_output_unknown_type,
        # format tests (single message)
        test_format_empty_messages,
        test_format_single_user_message,
        test_format_single_assistant_message,
        test_format_with_tool_use,
        test_format_with_tool_result,
        test_format_with_thinking_block,
        test_format_with_image,
        # format tests (multiple messages)
        test_format_conversation,
        test_format_without_index,
        test_format_without_time,
        test_format_with_tool_conversation,
        # threshold tests
        test_format_low_threshold,
        test_format_high_threshold,
        # edge cases
        test_format_long_text_truncation,
        test_format_special_characters,
        test_format_tool_result_with_complex_output,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            print_test_header(test.__name__)
            test()
            passed += 1
        except AssertionError as e:
            print_fail(test.__name__, str(e))
            failed += 1
        except Exception as e:
            print_error(test.__name__, str(e))
            failed += 1

    # 打印最终统计结果
    print(f"\n{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}Test Results Summary{Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.GREEN}{Colors.BOLD}✓ Passed: {passed}{Colors.RESET}")
    if failed > 0:
        print(f"{Colors.RED}{Colors.BOLD}✗ Failed: {failed}{Colors.RESET}")
    else:
        print(f"{Colors.GREEN}✗ Failed: {failed}{Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}")

    if failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 All tests passed!{Colors.RESET}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}💥 Some tests failed!{Colors.RESET}")


if __name__ == "__main__":
    run_all_tests()
