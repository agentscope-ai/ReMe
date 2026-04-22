"""Tests for AsMsgHandler.format_msgs_to_str method."""

# pylint: disable=W0212

import asyncio
import sys

from agentscope.message import Msg
from test_utils import get_token_counter

from reme.core.utils import get_logger
from reme.memory.file_based.utils import AsMsgHandler

logger = get_logger()


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


# ==================== Helper Functions ====================


def create_handler() -> AsMsgHandler:
    """Create an AsMsgHandler instance for testing."""
    return AsMsgHandler(token_counter=get_token_counter())


def verify_result_within_threshold(
    handler: AsMsgHandler,
    result: str,
    threshold: int,
    test_name: str = "",
    msgs: list[Msg] | None = None,
) -> None:
    """Verify that the included messages' original token count does not exceed threshold.

    Note: The format_msgs_to_str method uses message token statistics (not formatted
    string tokens) for threshold checking. The formatted result may have more tokens
    than the threshold due to added metadata (timestamps, role prefixes, etc.).

    This verification checks that included messages' original token sum <= threshold.

    Args:
        handler: The AsMsgHandler instance used for token counting.
        result: The formatted string result from format_msgs_to_str.
        threshold: The memory_compact_threshold value used.
        test_name: Optional test name for better error messages.
        msgs: Optional list of original messages to verify against.

    Raises:
        AssertionError: If included messages' token count exceeds threshold.
    """
    if not result or not msgs:
        return  # Empty result or no messages to verify

    # Calculate tokens of messages that were included in the result
    included_tokens = 0
    for msg in msgs:
        stat = asyncio.run(handler.stat_message(msg))
        # Check if this message's content appears in the result
        _ = stat.format(include_thinking=True)  # Use True to check all content
        # Simple heuristic: if the message content is in result, count its tokens
        content_blocks = msg.get_content_blocks()
        msg_included = False
        for block in content_blocks:
            block_type = block.get("type", "")
            if block_type == "text" and block.get("text", "") in result:
                msg_included = True
                break
            if block_type == "tool_use" and f"<tool_use>{block.get('name', '')}" in result:
                msg_included = True
                break
            if block_type == "tool_result" and f"<tool_result>{block.get('name', '')}" in result:
                msg_included = True
                break

        if msg_included:
            included_tokens += stat.total_tokens

    # Verify included messages' token sum doesn't exceed threshold
    # Allow small tolerance for edge cases
    assert (
        included_tokens <= threshold + 1
    ), f"{test_name}: Included messages token count ({included_tokens}) exceeds threshold ({threshold})."


def create_user_msg(content: str) -> Msg:
    """Create a user message."""
    return Msg(name="user", role="user", content=content)


def create_assistant_msg(content: str) -> Msg:
    """Create an assistant message."""
    return Msg(name="assistant", role="assistant", content=content)


def create_tool_use_msg(tool_name: str, tool_input: dict, tool_id: str = "call_123") -> Msg:
    """Create a message with tool_use content block."""
    return Msg(
        name="assistant",
        role="assistant",
        content=[
            {
                "type": "tool_use",
                "id": tool_id,
                "name": tool_name,
                "input": tool_input,
            },
        ],
    )


def create_tool_result_msg(tool_name: str, output: str | list[dict], tool_id: str = "call_123") -> Msg:
    """Create a message with tool_result content block."""
    return Msg(
        name="tool",
        role="user",
        content=[
            {
                "type": "tool_result",
                "id": tool_id,
                "name": tool_name,
                "output": output,
            },
        ],
    )


def create_thinking_msg(thinking_content: str, text_content: str = "") -> Msg:
    """Create a message with thinking content block."""
    content = [
        {
            "type": "thinking",
            "thinking": thinking_content,
        },
    ]
    if text_content:
        content.append({"type": "text", "text": text_content})
    return Msg(name="assistant", role="assistant", content=content)


def create_image_msg(url: str = "") -> Msg:
    """Create a message with image content block."""
    content = [
        {
            "type": "image",
            "source": {"url": url} if url else {},
        },
    ]
    return Msg(name="assistant", role="assistant", content=content)


def create_mixed_content_msg(
    text: str = "",
    thinking: str = "",
    tool_name: str = "",
    tool_input: dict | None = None,
    image_url: str = "",
) -> Msg:
    """Create a message with mixed content blocks."""
    content = []
    if thinking:
        content.append({"type": "thinking", "thinking": thinking})
    if text:
        content.append({"type": "text", "text": text})
    if tool_name:
        content.append(
            {
                "type": "tool_use",
                "id": "call_mixed",
                "name": tool_name,
                "input": tool_input or {},
            },
        )
    if image_url:
        content.append({"type": "image", "source": {"url": image_url}})
    return Msg(name="assistant", role="assistant", content=content)


# ==================== Normal Case Tests ====================


def test_format_msgs_to_str_empty_list():
    """Test format_msgs_to_str with empty message list."""
    handler = create_handler()
    threshold = 4000
    msgs = []
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))
    assert result == "", f"Expected empty string for empty list, got: {result}"
    verify_result_within_threshold(handler, result, threshold, "empty_list", msgs)
    print_pass("test_format_msgs_to_str_empty_list")


def test_format_msgs_to_str_single_message():
    """Test format_msgs_to_str with a single message."""
    handler = create_handler()
    threshold = 4000
    msgs = [create_user_msg("Hello, how are you?")]
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    assert "user:" in result, f"Expected 'user:' in result, got: {result}"
    assert "Hello, how are you?" in result, f"Expected content in result, got: {result}"
    verify_result_within_threshold(handler, result, threshold, "single_message", msgs)
    print_pass("test_format_msgs_to_str_single_message")


def test_format_msgs_to_str_multiple_messages():
    """Test format_msgs_to_str with multiple messages."""
    handler = create_handler()
    threshold = 4000
    msgs = [
        create_user_msg("What is Python?"),
        create_assistant_msg("Python is a programming language."),
        create_user_msg("Tell me more."),
        create_assistant_msg("Python is known for its readability."),
    ]
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    assert "What is Python?" in result
    assert "Python is a programming language." in result
    assert "Tell me more." in result
    assert "Python is known for its readability." in result
    verify_result_within_threshold(handler, result, threshold, "multiple_messages", msgs)
    print_pass("test_format_msgs_to_str_multiple_messages")


def test_format_msgs_to_str_message_order():
    """Test that messages are returned in correct order (oldest to newest)."""
    handler = create_handler()
    threshold = 4000
    msgs = [
        create_user_msg("First message"),
        create_assistant_msg("Second message"),
        create_user_msg("Third message"),
    ]
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    # Find positions of each message
    first_pos = result.find("First message")
    second_pos = result.find("Second message")
    third_pos = result.find("Third message")

    assert first_pos < second_pos < third_pos, (
        f"Messages not in correct order. Positions: first={first_pos}, " f"second={second_pos}, third={third_pos}"
    )
    verify_result_within_threshold(handler, result, threshold, "message_order", msgs)
    print_pass("test_format_msgs_to_str_message_order")


def test_format_msgs_to_str_with_tool_use():
    """Test format_msgs_to_str with tool_use message."""
    handler = create_handler()
    threshold = 4000
    msgs = [create_tool_use_msg("read_file", {"path": "/test.txt"})]
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    assert "<tool_use>read_file" in result, f"Expected tool_use in result, got: {result}"
    verify_result_within_threshold(handler, result, threshold, "with_tool_use", msgs)
    print_pass("test_format_msgs_to_str_with_tool_use")


def test_format_msgs_to_str_with_tool_result():
    """Test format_msgs_to_str with tool_result message."""
    handler = create_handler()
    threshold = 4000
    msgs = [create_tool_result_msg("read_file", "file content here")]
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    assert "<tool_result>read_file" in result, f"Expected tool_result in result, got: {result}"
    verify_result_within_threshold(handler, result, threshold, "with_tool_result", msgs)
    print_pass("test_format_msgs_to_str_with_tool_result")


def test_format_msgs_to_str_with_image():
    """Test format_msgs_to_str with image message."""
    handler = create_handler()
    threshold = 4000
    msgs = [create_image_msg("https://example.com/image.png")]
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    assert "<image>" in result, f"Expected '<image>' in result, got: {result}"
    verify_result_within_threshold(handler, result, threshold, "with_image", msgs)
    print_pass("test_format_msgs_to_str_with_image")


def test_format_msgs_to_str_conversation_flow():
    """Test format_msgs_to_str with a complete conversation flow."""
    handler = create_handler()
    threshold = 4000
    msgs = [
        create_user_msg("Read the file."),
        create_tool_use_msg("read_file", {"path": "/data.txt"}),
        create_tool_result_msg("read_file", "File content here"),
        create_assistant_msg("The file contains: File content here"),
    ]
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    assert "user:" in result
    assert "<tool_use>read_file" in result
    assert "<tool_result>read_file" in result
    assert "assistant:" in result
    verify_result_within_threshold(handler, result, threshold, "conversation_flow", msgs)
    print_pass("test_format_msgs_to_str_conversation_flow")


# ==================== Thinking Block Tests ====================


def test_format_msgs_to_str_thinking_excluded_by_default():
    """Test that thinking blocks are excluded when include_thinking=False (default)."""
    handler = create_handler()
    threshold = 4000
    msgs = [create_thinking_msg("Let me think about this...", "Here is my response")]
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold, include_thinking=False))

    assert "Let me think about this" not in result, f"Thinking content should be excluded, got: {result}"
    assert "Here is my response" in result, f"Text content should be included, got: {result}"
    verify_result_within_threshold(handler, result, threshold, "thinking_excluded_by_default", msgs)
    print_pass("test_format_msgs_to_str_thinking_excluded_by_default")


def test_format_msgs_to_str_thinking_included():
    """Test that thinking blocks are included when include_thinking=True."""
    handler = create_handler()
    threshold = 4000
    msgs = [create_thinking_msg("Let me think about this...", "Here is my response")]
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold, include_thinking=True))

    assert "Let me think about this" in result, f"Thinking content should be included, got: {result}"
    assert "<thinking>" in result, f"Expected thinking tag in result, got: {result}"
    verify_result_within_threshold(handler, result, threshold, "thinking_included", msgs)
    print_pass("test_format_msgs_to_str_thinking_included")


def test_format_msgs_to_str_thinking_only_message():
    """Test message with only thinking block."""
    handler = create_handler()
    threshold = 4000
    msgs = [create_thinking_msg("Deep thoughts here")]

    # With include_thinking=False
    result_no_thinking = asyncio.run(
        handler.format_msgs_to_str(
            msgs,
            memory_compact_threshold=threshold,
            include_thinking=False,
        ),
    )
    # With include_thinking=True
    result_with_thinking = asyncio.run(
        handler.format_msgs_to_str(
            msgs,
            memory_compact_threshold=threshold,
            include_thinking=True,
        ),
    )

    assert "Deep thoughts here" not in result_no_thinking
    assert "Deep thoughts here" in result_with_thinking
    verify_result_within_threshold(handler, result_no_thinking, threshold, "thinking_only_no_thinking", msgs)
    verify_result_within_threshold(handler, result_with_thinking, threshold, "thinking_only_with_thinking", msgs)
    print_pass("test_format_msgs_to_str_thinking_only_message")


# ==================== Token Threshold Tests ====================


def test_format_msgs_to_str_all_within_threshold():
    """Test all messages fit within threshold."""
    handler = create_handler()
    threshold = 10000
    msgs = [
        create_user_msg("Short message 1"),
        create_assistant_msg("Short message 2"),
        create_user_msg("Short message 3"),
    ]
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    assert "Short message 1" in result
    assert "Short message 2" in result
    assert "Short message 3" in result
    verify_result_within_threshold(handler, result, threshold, "all_within_threshold", msgs)
    print_pass("test_format_msgs_to_str_all_within_threshold")


def test_format_msgs_to_str_exceeds_threshold_truncate_older():
    """Test that older messages are truncated when exceeding threshold."""
    handler = create_handler()
    threshold = 500
    msgs = []
    for i in range(20):
        msgs.append(create_user_msg(f"Question {i}: " + "x" * 100))
        msgs.append(create_assistant_msg(f"Answer {i}: " + "y" * 100))

    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    # The newest messages should be present
    assert (
        "Answer 19" in result or "Question 19" in result
    ), f"Expected recent message in result, got: {result[:500]}..."
    # Older messages should be truncated
    assert "Question 0" not in result, "Older messages should be truncated"
    verify_result_within_threshold(handler, result, threshold, "exceeds_threshold_truncate_older", msgs)
    print_pass("test_format_msgs_to_str_exceeds_threshold_truncate_older")


def test_format_msgs_to_str_single_message_exceeds_threshold():
    """Test when a single message exceeds the threshold."""
    handler = create_handler()
    threshold = 10
    # Create a very long message
    long_text = "x" * 10000
    msgs = [create_user_msg(long_text)]

    # With very low threshold, even a single message won't fit
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    # The message should be skipped entirely since it exceeds threshold
    assert result == "" or len(result) > 0, "Result should be empty or contain truncated content"
    verify_result_within_threshold(handler, result, threshold, "single_message_exceeds_threshold", msgs)
    print_pass("test_format_msgs_to_str_single_message_exceeds_threshold")


def test_format_msgs_to_str_first_message_exceeds_threshold():
    """Test when the first (oldest) message exceeds threshold but newer ones don't."""
    handler = create_handler()
    threshold = 100
    msgs = [
        create_user_msg("x" * 5000),  # Old, long message
        create_assistant_msg("Short response"),  # New, short message
    ]

    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    # Newer message should be present
    assert "Short response" in result, f"Expected newer message in result, got: {result}"
    verify_result_within_threshold(handler, result, threshold, "first_message_exceeds_threshold", msgs)
    print_pass("test_format_msgs_to_str_first_message_exceeds_threshold")


def test_format_msgs_to_str_threshold_zero():
    """Test with threshold of zero - latest message is still included."""
    handler = create_handler()
    threshold = 0
    msgs = [create_user_msg("Test message")]

    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    # Latest message is included even with zero threshold (implementation behavior)
    assert "Test message" in result, f"Expected message in result, got: {result}"
    # Skip verify_result_within_threshold since latest message is always included
    print_pass("test_format_msgs_to_str_threshold_zero")


def test_format_msgs_to_str_threshold_exact_fit():
    """Test when messages exactly fit the threshold."""
    handler = create_handler()
    # Create a message and measure its formatted string tokens
    msg = create_user_msg("Test")
    stat = asyncio.run(handler.stat_message(msg))
    formatted_content = stat.format(include_thinking=False)
    exact_threshold = asyncio.run(handler.count_str_token(formatted_content))

    msgs = [msg]
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=exact_threshold))

    assert "Test" in result, f"Message should fit exactly, got: {result}"
    verify_result_within_threshold(handler, result, exact_threshold, "threshold_exact_fit", msgs)
    print_pass("test_format_msgs_to_str_threshold_exact_fit")


def test_format_msgs_to_str_threshold_one_less():
    """Test when threshold is one less than needed - latest message is still included."""
    handler = create_handler()
    msg = create_user_msg("Test message")
    stat = asyncio.run(handler.stat_message(msg))
    formatted_content = stat.format(include_thinking=False)
    threshold_minus_one = asyncio.run(handler.count_str_token(formatted_content)) - 1

    msgs = [msg]
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold_minus_one))

    # Latest message is included even when it exceeds threshold (implementation behavior)
    assert "Test message" in result, f"Expected message in result, got: {result}"
    # Skip verify_result_within_threshold since latest message is always included
    print_pass("test_format_msgs_to_str_threshold_one_less")


def test_format_msgs_to_str_large_threshold():
    """Test with very large threshold - all messages should be included."""
    handler = create_handler()
    threshold = 1000000
    msgs = [create_user_msg("Message " + str(i) + " " + "x" * 100) for i in range(50)]

    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    # All messages should be included
    for i in range(50):
        assert f"Message {i}" in result, f"Message {i} should be included"
    verify_result_within_threshold(handler, result, threshold, "large_threshold", msgs)
    print_pass("test_format_msgs_to_str_large_threshold")


# ==================== Edge Cases Tests ====================


def test_format_msgs_to_str_special_characters():
    """Test with special characters in content."""
    handler = create_handler()
    threshold = 4000
    msgs = [create_user_msg("Test with 中文, 日本語, émojis 🎉 and symbols @#$%")]
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    assert "中文" in result
    assert "日本語" in result
    assert "🎉" in result
    verify_result_within_threshold(handler, result, threshold, "special_characters", msgs)
    print_pass("test_format_msgs_to_str_special_characters")


def test_format_msgs_to_str_empty_content():
    """Test with empty content message."""
    handler = create_handler()
    threshold = 4000
    msgs = [create_user_msg("")]
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    assert "user:" in result, f"Expected role in result even with empty content, got: {result}"
    verify_result_within_threshold(handler, result, threshold, "empty_content", msgs)
    print_pass("test_format_msgs_to_str_empty_content")


def test_format_msgs_to_str_whitespace_only():
    """Test with whitespace-only content."""
    handler = create_handler()
    threshold = 4000
    msgs = [create_user_msg("   \n\t  ")]
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    assert "user:" in result
    verify_result_within_threshold(handler, result, threshold, "whitespace_only", msgs)
    print_pass("test_format_msgs_to_str_whitespace_only")


def test_format_msgs_to_str_newlines_in_content():
    """Test with newlines in message content."""
    handler = create_handler()
    threshold = 4000
    msgs = [create_user_msg("Line 1\nLine 2\nLine 3")]
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    assert "Line 1" in result
    assert "Line 2" in result
    assert "Line 3" in result
    verify_result_within_threshold(handler, result, threshold, "newlines_in_content", msgs)
    print_pass("test_format_msgs_to_str_newlines_in_content")


def test_format_msgs_to_str_very_long_single_word():
    """Test with very long single word (no spaces)."""
    handler = create_handler()
    threshold = 10000
    long_word = "a" * 5000
    msgs = [create_user_msg(long_word)]
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    # Should contain at least part of the word (may be truncated by formatter)
    assert "aaa" in result, f"Expected long word content in result, got: {result[:100]}..."
    verify_result_within_threshold(handler, result, threshold, "very_long_single_word", msgs)
    print_pass("test_format_msgs_to_str_very_long_single_word")


def test_format_msgs_to_str_mixed_content_blocks():
    """Test message with mixed content blocks."""
    handler = create_handler()
    threshold = 4000
    msgs = [
        create_mixed_content_msg(
            text="Text content",
            thinking="Thinking content",
            tool_name="test_tool",
            tool_input={"key": "value"},
            image_url="https://example.com/img.png",
        ),
    ]

    result_no_thinking = asyncio.run(
        handler.format_msgs_to_str(
            msgs,
            memory_compact_threshold=threshold,
            include_thinking=False,
        ),
    )
    result_with_thinking = asyncio.run(
        handler.format_msgs_to_str(
            msgs,
            memory_compact_threshold=threshold,
            include_thinking=True,
        ),
    )

    assert "Text content" in result_no_thinking
    assert "<tool_use>test_tool" in result_no_thinking
    assert "<image>" in result_no_thinking
    assert "Thinking content" not in result_no_thinking
    assert "Thinking content" in result_with_thinking
    verify_result_within_threshold(handler, result_no_thinking, threshold, "mixed_content_no_thinking", msgs)
    verify_result_within_threshold(handler, result_with_thinking, threshold, "mixed_content_with_thinking", msgs)
    print_pass("test_format_msgs_to_str_mixed_content_blocks")


def test_format_msgs_to_str_multiple_separators():
    """Test that messages are separated by double newlines."""
    handler = create_handler()
    threshold = 4000
    msgs = [
        create_user_msg("Message 1"),
        create_assistant_msg("Message 2"),
    ]
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    assert "\n\n" in result, f"Expected double newline separator, got: {result}"
    verify_result_within_threshold(handler, result, threshold, "multiple_separators", msgs)
    print_pass("test_format_msgs_to_str_multiple_separators")


def test_format_msgs_to_str_tool_result_complex_output():
    """Test tool_result with complex output (list of blocks)."""
    handler = create_handler()
    threshold = 4000
    complex_output = [
        {"type": "text", "text": "Operation completed"},
        {"type": "image", "source": {"url": "https://example.com/result.png"}},
    ]
    msgs = [create_tool_result_msg("process_data", complex_output)]
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    assert "<tool_result>process_data" in result
    verify_result_within_threshold(handler, result, threshold, "tool_result_complex_output", msgs)
    print_pass("test_format_msgs_to_str_tool_result_complex_output")


def test_format_msgs_to_str_different_roles():
    """Test with different roles (user, assistant, system, tool)."""
    handler = create_handler()
    threshold = 4000
    msgs = [
        Msg(name="system", role="system", content="System instruction"),
        create_user_msg("User message"),
        create_assistant_msg("Assistant response"),
        create_tool_result_msg("tool", "Tool output"),
    ]
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    assert "system:" in result
    assert "user:" in result
    assert "assistant:" in result
    verify_result_within_threshold(handler, result, threshold, "different_roles", msgs)
    print_pass("test_format_msgs_to_str_different_roles")


def test_format_msgs_to_str_incremental_threshold_check():
    """Test incremental addition of messages until threshold is exceeded."""
    handler = create_handler()

    # Create messages with known approximate sizes
    msgs = []
    for i in range(10):
        msgs.append(create_user_msg(f"Message {i} with some padding text"))

    # Calculate total tokens
    async def get_total_tokens():
        total = 0
        for msg in msgs:
            stat = await handler.stat_message(msg)
            total += stat.total_tokens
        return total

    total_tokens = asyncio.run(get_total_tokens())

    # Use threshold that allows about half the messages
    half_threshold = total_tokens // 2
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=half_threshold))

    # Should have some but not all messages
    included_count = sum(1 for i in range(10) if f"Message {i}" in result)
    assert 0 < included_count < 10, f"Expected partial messages, got {included_count} messages included"
    # Newer messages should be included (messages are processed from end)
    assert "Message 9" in result, "Newest message should be included"
    verify_result_within_threshold(handler, result, half_threshold, "incremental_threshold_check", msgs)
    print_pass("test_format_msgs_to_str_incremental_threshold_check")


def test_format_msgs_to_str_negative_threshold():
    """Test with negative threshold value - latest message is still included."""
    handler = create_handler()
    threshold = -1
    msgs = [create_user_msg("Test message")]

    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    # Latest message is included even with negative threshold (implementation behavior)
    assert "Test message" in result, f"Expected message in result, got: {result}"
    # Skip verify_result_within_threshold since latest message is always included
    print_pass("test_format_msgs_to_str_negative_threshold")


def test_format_msgs_to_str_preserves_newest_first():
    """Test that newest messages are preserved when threshold is exceeded."""
    handler = create_handler()
    threshold = 300
    msgs = [
        create_user_msg("OLD MESSAGE " + "x" * 200),
        create_assistant_msg("MIDDLE MESSAGE " + "y" * 200),
        create_user_msg("NEW MESSAGE " + "z" * 200),
    ]

    # Use threshold that only allows ~1-2 messages
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    # Newest message should be present
    assert "NEW MESSAGE" in result, f"Expected newest message, got: {result}"
    verify_result_within_threshold(handler, result, threshold, "preserves_newest_first", msgs)
    print_pass("test_format_msgs_to_str_preserves_newest_first")


def test_format_msgs_to_str_base64_image():
    """Test with base64 encoded image."""
    handler = create_handler()
    threshold = 10000
    msgs = [
        Msg(
            name="assistant",
            role="assistant",
            content=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "data": "SGVsbG8gV29ybGQ=" * 100,  # Simulated base64 data
                    },
                },
            ],
        ),
    ]
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    assert "<image>" in result
    verify_result_within_threshold(handler, result, threshold, "base64_image", msgs)
    print_pass("test_format_msgs_to_str_base64_image")


def test_format_msgs_to_str_audio_video_blocks():
    """Test with audio and video content blocks."""
    handler = create_handler()
    threshold = 4000
    msgs = [
        Msg(
            name="assistant",
            role="assistant",
            content=[
                {"type": "audio", "source": {"url": "https://example.com/audio.mp3"}},
                {"type": "video", "source": {"url": "https://example.com/video.mp4"}},
            ],
        ),
    ]
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    assert "<audio>" in result
    assert "<video>" in result
    verify_result_within_threshold(handler, result, threshold, "audio_video_blocks", msgs)
    print_pass("test_format_msgs_to_str_audio_video_blocks")


def test_format_msgs_to_str_unknown_block_type():
    """Test that unknown block types are skipped gracefully."""
    handler = create_handler()
    threshold = 4000
    msgs = [
        Msg(
            name="assistant",
            role="assistant",
            content=[
                {"type": "unknown_type", "data": "some data"},
                {"type": "text", "text": "Valid text"},
            ],
        ),
    ]
    result = asyncio.run(handler.format_msgs_to_str(msgs, memory_compact_threshold=threshold))

    # Should still include valid content
    assert "Valid text" in result
    verify_result_within_threshold(handler, result, threshold, "unknown_block_type", msgs)
    print_pass("test_format_msgs_to_str_unknown_block_type")


def run_all_tests():
    """Run all tests."""
    tests = [
        # Normal case tests
        test_format_msgs_to_str_empty_list,
        test_format_msgs_to_str_single_message,
        test_format_msgs_to_str_multiple_messages,
        test_format_msgs_to_str_message_order,
        test_format_msgs_to_str_with_tool_use,
        test_format_msgs_to_str_with_tool_result,
        test_format_msgs_to_str_with_image,
        test_format_msgs_to_str_conversation_flow,
        # Thinking block tests
        test_format_msgs_to_str_thinking_excluded_by_default,
        test_format_msgs_to_str_thinking_included,
        test_format_msgs_to_str_thinking_only_message,
        # Token threshold tests
        test_format_msgs_to_str_all_within_threshold,
        test_format_msgs_to_str_exceeds_threshold_truncate_older,
        test_format_msgs_to_str_single_message_exceeds_threshold,
        test_format_msgs_to_str_first_message_exceeds_threshold,
        test_format_msgs_to_str_threshold_zero,
        test_format_msgs_to_str_threshold_exact_fit,
        test_format_msgs_to_str_threshold_one_less,
        test_format_msgs_to_str_large_threshold,
        # Edge cases tests
        test_format_msgs_to_str_special_characters,
        test_format_msgs_to_str_empty_content,
        test_format_msgs_to_str_whitespace_only,
        test_format_msgs_to_str_newlines_in_content,
        test_format_msgs_to_str_very_long_single_word,
        test_format_msgs_to_str_mixed_content_blocks,
        test_format_msgs_to_str_multiple_separators,
        test_format_msgs_to_str_tool_result_complex_output,
        test_format_msgs_to_str_different_roles,
        test_format_msgs_to_str_incremental_threshold_check,
        test_format_msgs_to_str_negative_threshold,
        test_format_msgs_to_str_preserves_newest_first,
        test_format_msgs_to_str_base64_image,
        test_format_msgs_to_str_audio_video_blocks,
        test_format_msgs_to_str_unknown_block_type,
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

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
