"""Tests for AsMsgHandler.context_check method."""

import asyncio

from agentscope.message import Msg
from test_utils import get_token_counter

from reme.core.utils import get_logger
from reme.memory.file_based.utils import AsMsgHandler

logger = get_logger()


# ANSI color codes
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
    """Print test passed message."""
    print(f"{Colors.GREEN}{Colors.BOLD}✓ {test_name} PASSED{Colors.RESET}")


def print_fail(test_name: str, error: str):
    """Print test failed message."""
    print(f"{Colors.RED}{Colors.BOLD}✗ {test_name} FAILED: {error}{Colors.RESET}")


def print_error(test_name: str, error: str):
    """Print test error message."""
    print(f"{Colors.YELLOW}{Colors.BOLD}⚠ {test_name} ERROR: {error}{Colors.RESET}")


def print_test_header(test_name: str):
    """Print test header."""
    print(f"\n{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}Running: {test_name}{Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}")


def create_handler() -> AsMsgHandler:
    """Create an AsMsgHandler instance for testing."""
    return AsMsgHandler(token_counter=get_token_counter())


def verify_context_check_invariants(
    handler: AsMsgHandler,
    messages: list[Msg],
    to_compact: list[Msg],
    to_keep: list[Msg],
    memory_compact_threshold: int,
    memory_compact_reserve: int,
    test_name: str,
):
    """Verify that context_check results satisfy all invariants.

    This function checks:
    1. Threshold requirement: If total tokens <= threshold, no compaction should occur
    2. Reserve requirement: Kept messages' total tokens should not exceed reserve
    3. Order requirement: Both to_compact and to_keep should preserve original order

    Args:
        handler: The AsMsgHandler instance
        messages: Original messages list
        to_compact: Messages to compact returned by context_check
        to_keep: Messages to keep returned by context_check
        memory_compact_threshold: The threshold parameter used
        memory_compact_reserve: The reserve parameter used
        test_name: Name of the test for error reporting

    Raises:
        AssertionError: If any invariant is violated
    """
    # Calculate total tokens of original messages
    total_tokens = sum(asyncio.run(handler.stat_message(m)).total_tokens for m in messages)

    # 1. Threshold requirement check
    if total_tokens <= memory_compact_threshold:
        assert len(to_compact) == 0, (
            f"[{test_name}] Threshold violation: total_tokens ({total_tokens}) <= "
            f"threshold ({memory_compact_threshold}), but to_compact is not empty "
            f"(has {len(to_compact)} messages)"
        )
        assert to_keep == messages, (
            f"[{test_name}] Threshold violation: total_tokens ({total_tokens}) <= "
            f"threshold ({memory_compact_threshold}), but to_keep differs from original messages"
        )

    # 2. Reserve requirement check
    kept_tokens = sum(asyncio.run(handler.stat_message(m)).total_tokens for m in to_keep)
    assert kept_tokens <= memory_compact_reserve or len(to_keep) == 0, (
        f"[{test_name}] Reserve violation: kept_tokens ({kept_tokens}) > " f"reserve ({memory_compact_reserve})"
    )

    # 3. Order requirement check - both lists should preserve original order
    # Create a mapping of message id to original index
    msg_to_idx = {id(m): i for i, m in enumerate(messages)}

    # Check to_compact order
    compact_indices = [msg_to_idx.get(id(m), -1) for m in to_compact]
    for i in range(len(compact_indices) - 1):
        assert compact_indices[i] < compact_indices[i + 1], (
            f"[{test_name}] Order violation in to_compact: message at original index "
            f"{compact_indices[i]} appears before message at index {compact_indices[i + 1]}"
        )

    # Check to_keep order
    keep_indices = [msg_to_idx.get(id(m), -1) for m in to_keep]
    for i in range(len(keep_indices) - 1):
        assert keep_indices[i] < keep_indices[i + 1], (
            f"[{test_name}] Order violation in to_keep: message at original index "
            f"{keep_indices[i]} appears before message at index {keep_indices[i + 1]}"
        )

    # 4. Additional check: to_compact indices should all be less than to_keep indices
    # (compact messages come from the beginning, keep messages come from the end)
    if to_compact and to_keep:
        max_compact_idx = max(compact_indices) if compact_indices else -1
        min_keep_idx = min(keep_indices) if keep_indices else len(messages)
        assert max_compact_idx < min_keep_idx, (
            f"[{test_name}] Partition violation: max compact index ({max_compact_idx}) >= "
            f"min keep index ({min_keep_idx}). Compact and keep should be a clean partition."
        )

    # 5. Check that all messages are accounted for (no duplicates, no missing)
    assert len(to_compact) + len(to_keep) == len(messages), (
        f"[{test_name}] Count mismatch: to_compact ({len(to_compact)}) + "
        f"to_keep ({len(to_keep)}) != original ({len(messages)})"
    )

    all_returned = set(id(m) for m in to_compact) | set(id(m) for m in to_keep)
    all_original = set(id(m) for m in messages)
    assert all_returned == all_original, f"[{test_name}] Message set mismatch: returned messages differ from original"


def create_user_msg(content: str) -> Msg:
    """Create a user message."""
    return Msg(name="user", role="user", content=content)


def create_assistant_msg(content: str) -> Msg:
    """Create an assistant message."""
    return Msg(name="assistant", role="assistant", content=content)


def create_tool_use_msg(tool_id: str, tool_name: str, tool_input: dict) -> Msg:
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


def create_tool_result_msg(tool_id: str, tool_name: str, output: str) -> Msg:
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


def create_mixed_tool_msg(
    tool_use_id: str,
    tool_use_name: str,
    tool_use_input: dict,
    tool_result_id: str,
    tool_result_name: str,
    tool_result_output: str,
) -> Msg:
    """Create a message with both tool_use and tool_result blocks."""
    return Msg(
        name="assistant",
        role="assistant",
        content=[
            {
                "type": "tool_use",
                "id": tool_use_id,
                "name": tool_use_name,
                "input": tool_use_input,
            },
            {
                "type": "tool_result",
                "id": tool_result_id,
                "name": tool_result_name,
                "output": tool_result_output,
            },
        ],
    )


# =============================================================================
# Normal Cases
# =============================================================================


def test_empty_messages():
    """Test context_check with empty messages list."""
    handler = create_handler()
    messages = []
    threshold, reserve = 1000, 500
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,
            memory_compact_reserve=reserve,
        ),
    )
    assert not to_compact, f"Expected empty compact list, got: {to_compact}"
    assert to_keep == [], f"Expected empty keep list, got: {to_keep}"
    verify_context_check_invariants(handler, messages, to_compact, to_keep, threshold, reserve, "test_empty_messages")
    print_pass("test_empty_messages")


def test_below_threshold_returns_all():
    """Test that messages below threshold are all kept."""
    handler = create_handler()
    messages = [
        create_user_msg("Hello"),
        create_assistant_msg("Hi there!"),
        create_user_msg("How are you?"),
    ]
    threshold, reserve = 10000, 5000
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,  # Very high threshold
            memory_compact_reserve=reserve,
        ),
    )
    assert not to_compact, f"Expected empty compact list, got: {len(to_compact)}"
    assert len(to_keep) == 3, f"Expected 3 messages to keep, got: {len(to_keep)}"
    assert to_keep == messages, "Messages to keep should be the original messages"
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_below_threshold_returns_all",
    )
    print_pass("test_below_threshold_returns_all")


def test_above_threshold_triggers_compaction():
    """Test that messages above threshold are split correctly."""
    handler = create_handler()
    # Create messages that will exceed threshold
    messages = [
        create_user_msg("First message " * 100),
        create_assistant_msg("Second message " * 100),
        create_user_msg("Third message " * 100),
        create_assistant_msg("Fourth message " * 100),
    ]
    threshold, reserve = 100, 200
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,  # Low threshold to trigger compaction
            memory_compact_reserve=reserve,
        ),
    )
    # Should have some messages compacted and some kept
    assert len(to_compact) + len(to_keep) == len(messages), "Total messages should match"
    assert len(to_compact) > 0, "Expected some messages to be compacted"
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_above_threshold_triggers_compaction",
    )
    print_pass("test_above_threshold_triggers_compaction")


def test_message_order_preserved():
    """Test that message order is preserved in both lists."""
    handler = create_handler()
    messages = [
        create_user_msg("First " * 50),
        create_assistant_msg("Second " * 50),
        create_user_msg("Third " * 50),
        create_assistant_msg("Fourth " * 50),
        create_user_msg("Fifth " * 10),
    ]
    threshold, reserve = 100, 150
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,  # Low threshold
            memory_compact_reserve=reserve,
        ),
    )
    # Check order preservation - compact messages should appear first in original
    all_messages = to_compact + to_keep
    for i, msg in enumerate(all_messages):
        assert msg in messages, f"Message {i} not found in original messages"
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_message_order_preserved",
    )
    print_pass("test_message_order_preserved")


# =============================================================================
# Edge Cases - Threshold and Reserve Boundaries
# =============================================================================


def test_single_message_below_threshold():
    """Test single message below threshold."""
    handler = create_handler()
    messages = [create_user_msg("Short message")]
    threshold, reserve = 1000, 500
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,
            memory_compact_reserve=reserve,
        ),
    )
    assert not to_compact, "Should not compact single message below threshold"
    assert len(to_keep) == 1, "Should keep the single message"
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_single_message_below_threshold",
    )
    print_pass("test_single_message_below_threshold")


def test_single_message_above_threshold():
    """Test single message that exceeds threshold - nothing can be kept in reserve."""
    handler = create_handler()
    long_content = "Very long message " * 1000
    messages = [create_user_msg(long_content)]
    threshold, reserve = 10, 5
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,  # Very low threshold
            memory_compact_reserve=reserve,  # Even lower reserve
        ),
    )
    # Message exceeds both threshold and reserve, so it's compacted
    assert len(to_compact) == 1, "Single large message should be compacted"
    assert len(to_keep) == 0, "Nothing can fit in reserve"
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_single_message_above_threshold",
    )
    print_pass("test_single_message_above_threshold")


def test_reserve_zero():
    """Test with reserve=0, no messages can be kept."""
    handler = create_handler()
    messages = [
        create_user_msg("Hello"),
        create_assistant_msg("Hi there!"),
    ]
    threshold, reserve = 1, 0
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,  # Trigger compaction
            memory_compact_reserve=reserve,  # Zero reserve
        ),
    )
    # All messages should be compacted since reserve is 0
    assert len(to_compact) == 2, f"All messages should be compacted, got {len(to_compact)}"
    assert len(to_keep) == 0, f"No messages should be kept, got {len(to_keep)}"
    verify_context_check_invariants(handler, messages, to_compact, to_keep, threshold, reserve, "test_reserve_zero")
    print_pass("test_reserve_zero")


def test_threshold_zero():
    """Test with threshold=0, always triggers compaction."""
    handler = create_handler()
    messages = [create_user_msg("A")]  # Minimal message
    threshold, reserve = 0, 1000
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,  # Zero threshold - always triggers
            memory_compact_reserve=reserve,
        ),
    )
    # Even minimal message triggers compaction with threshold=0
    # But reserve is high so it should be kept
    assert len(to_compact) == 0 or len(to_keep) == 1, "Message should fit in reserve"
    verify_context_check_invariants(handler, messages, to_compact, to_keep, threshold, reserve, "test_threshold_zero")
    print_pass("test_threshold_zero")


def test_exact_threshold_boundary():
    """Test messages exactly at threshold boundary."""
    handler = create_handler()
    messages = [create_user_msg("Test message")]

    # Get exact token count
    stat = asyncio.run(handler.stat_message(messages[0]))
    exact_tokens = stat.total_tokens
    threshold, reserve = exact_tokens, exact_tokens

    # Test at exact boundary
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,  # Exactly at boundary
            memory_compact_reserve=reserve,
        ),
    )
    # At exact boundary (<=), should not trigger compaction
    assert not to_compact, "Should not compact at exact boundary"
    assert len(to_keep) == 1, "Should keep message at exact boundary"
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_exact_threshold_boundary",
    )
    print_pass("test_exact_threshold_boundary")


def test_reserve_larger_than_threshold():
    """Test when reserve is larger than threshold (unusual but valid config)."""
    handler = create_handler()
    messages = [
        create_user_msg("Message one " * 20),
        create_assistant_msg("Message two " * 20),
    ]
    threshold, reserve = 50, 10000
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,  # Low threshold
            memory_compact_reserve=reserve,  # High reserve
        ),
    )
    # Compaction triggered but reserve can hold everything
    # Total messages should be preserved
    assert len(to_compact) + len(to_keep) == 2
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_reserve_larger_than_threshold",
    )
    print_pass("test_reserve_larger_than_threshold")


# =============================================================================
# Edge Cases - Tool Use/Result Pairing
# =============================================================================


def test_tool_use_result_paired():
    """Test that tool_use and tool_result pairs are kept together."""
    handler = create_handler()
    messages = [
        create_user_msg("Please run the tool " * 50),
        create_tool_use_msg("call_001", "test_tool", {"arg": "value"}),
        create_tool_result_msg("call_001", "test_tool", "Tool output"),
        create_assistant_msg("The tool returned results"),
    ]
    threshold, reserve = 50, 1000
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,  # Trigger compaction
            memory_compact_reserve=reserve,  # Enough for tool pair
        ),
    )

    # If tool_result is kept, tool_use should also be kept
    tool_result_in_keep = any(any(b.get("type") == "tool_result" for b in m.get_content_blocks()) for m in to_keep)
    tool_use_in_keep = any(any(b.get("type") == "tool_use" for b in m.get_content_blocks()) for m in to_keep)

    if tool_result_in_keep:
        assert tool_use_in_keep, "tool_use should be kept when tool_result is kept"
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_tool_use_result_paired",
    )
    print_pass("test_tool_use_result_paired")


def test_tool_use_without_result():
    """Test tool_use message without corresponding tool_result."""
    handler = create_handler()
    messages = [
        create_user_msg("Run the tool"),
        create_tool_use_msg("call_orphan", "orphan_tool", {"arg": "value"}),
        create_assistant_msg("Something happened"),
    ]
    threshold, reserve = 10, 1000
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,
            memory_compact_reserve=reserve,
        ),
    )
    # Should not crash, just process normally
    assert len(to_compact) + len(to_keep) == 3
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_tool_use_without_result",
    )
    print_pass("test_tool_use_without_result")


def test_tool_result_without_use():
    """Test tool_result message without corresponding tool_use."""
    handler = create_handler()
    messages = [
        create_user_msg("Here's a result"),
        create_tool_result_msg("call_orphan", "orphan_tool", "Some output"),
        create_assistant_msg("Got it"),
    ]
    threshold, reserve = 10, 1000
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,
            memory_compact_reserve=reserve,
        ),
    )
    # Should not crash even with orphan tool_result
    assert len(to_compact) + len(to_keep) == 3
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_tool_result_without_use",
    )
    print_pass("test_tool_result_without_use")


def test_multiple_tool_pairs():
    """Test multiple tool_use/tool_result pairs."""
    handler = create_handler()
    messages = [
        create_user_msg("Start task " * 50),
        create_tool_use_msg("call_001", "tool_a", {"a": 1}),
        create_tool_result_msg("call_001", "tool_a", "Result A"),
        create_tool_use_msg("call_002", "tool_b", {"b": 2}),
        create_tool_result_msg("call_002", "tool_b", "Result B"),
        create_tool_use_msg("call_003", "tool_c", {"c": 3}),
        create_tool_result_msg("call_003", "tool_c", "Result C"),
        create_assistant_msg("All done"),
    ]
    threshold, reserve = 50, 500
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,
            memory_compact_reserve=reserve,
        ),
    )

    # Verify tool pairs integrity - for each kept tool_result, its tool_use should be kept
    for msg in to_keep:
        for block in msg.get_content_blocks("tool_result"):
            tool_id = block.get("id", "")
            if tool_id:
                # Find corresponding tool_use
                tool_use_found = False
                for keep_msg in to_keep:
                    for use_block in keep_msg.get_content_blocks("tool_use"):
                        if use_block.get("id") == tool_id:
                            tool_use_found = True
                            break
                assert tool_use_found, f"tool_use for {tool_id} should be kept with tool_result"
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_multiple_tool_pairs",
    )
    print_pass("test_multiple_tool_pairs")


def test_tool_dependency_causes_extra_inclusion():
    """Test that tool_use is included even if it exceeds simple reserve calculation."""
    handler = create_handler()
    # Create a scenario where:
    # - First message (tool_use) is large
    # - Later message (tool_result) references it
    # - Reserve alone wouldn't fit tool_use, but dependency requires it
    large_tool_input = {"data": "x" * 200}
    messages = [
        create_user_msg("Start " * 100),  # Large message
        create_tool_use_msg("call_dep", "dep_tool", large_tool_input),  # Medium
        create_user_msg("Middle " * 100),  # Large message
        create_tool_result_msg("call_dep", "dep_tool", "Result"),  # Small
        create_assistant_msg("End"),  # Small
    ]
    threshold, reserve = 100, 500
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,  # Trigger compaction
            memory_compact_reserve=reserve,  # Medium reserve
        ),
    )

    # Check pair integrity
    result_kept = any(
        any(b.get("id") == "call_dep" and b.get("type") == "tool_result" for b in m.get_content_blocks())
        for m in to_keep
    )
    use_kept = any(
        any(b.get("id") == "call_dep" and b.get("type") == "tool_use" for b in m.get_content_blocks()) for m in to_keep
    )

    if result_kept:
        assert use_kept, "Dependent tool_use should be included with tool_result"
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_tool_dependency_causes_extra_inclusion",
    )
    print_pass("test_tool_dependency_causes_extra_inclusion")


def test_tool_dependency_exceeds_reserve():
    """Test when tool_result + its tool_use dependency would exceed reserve."""
    handler = create_handler()
    # tool_use is very large, making the pair not fit in reserve
    very_large_input = {"data": "x" * 2000}
    messages = [
        create_user_msg("First"),
        create_tool_use_msg("call_big", "big_tool", very_large_input),  # Very large
        create_tool_result_msg("call_big", "big_tool", "Small result"),
        create_assistant_msg("Last message"),
    ]
    threshold, reserve = 10, 100
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,  # Trigger compaction
            memory_compact_reserve=reserve,  # Small reserve - can't fit the pair
        ),
    )

    # The tool pair is too large, so it should be excluded or partially handled
    # Either both are compacted (pair excluded) or neither is kept
    result_kept = any(
        any(b.get("id") == "call_big" and b.get("type") == "tool_result" for b in m.get_content_blocks())
        for m in to_keep
    )

    if result_kept:
        # If result is kept, use must also be kept (pair integrity)
        use_kept = any(
            any(b.get("id") == "call_big" and b.get("type") == "tool_use" for b in m.get_content_blocks())
            for m in to_keep
        )
        assert use_kept, "Pair integrity violated"
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_tool_dependency_exceeds_reserve",
    )
    print_pass("test_tool_dependency_exceeds_reserve")


def test_interleaved_tool_pairs():
    """Test interleaved tool_use/tool_result (not strictly sequential)."""
    handler = create_handler()
    messages = [
        create_user_msg("Multi-tool task " * 30),
        create_tool_use_msg("call_a", "tool_a", {"a": 1}),
        create_tool_use_msg("call_b", "tool_b", {"b": 2}),  # Two uses before results
        create_tool_result_msg("call_a", "tool_a", "Result A"),
        create_tool_result_msg("call_b", "tool_b", "Result B"),
        create_assistant_msg("Both done"),
    ]
    threshold, reserve = 50, 1000
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,
            memory_compact_reserve=reserve,
        ),
    )

    # Verify pair integrity for interleaved pairs
    for msg in to_keep:
        for block in msg.get_content_blocks("tool_result"):
            tool_id = block.get("id", "")
            if tool_id:
                use_found = any(
                    any(ub.get("id") == tool_id and ub.get("type") == "tool_use" for ub in km.get_content_blocks())
                    for km in to_keep
                )
                assert use_found, f"Interleaved tool_use {tool_id} should be kept"
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_interleaved_tool_pairs",
    )
    print_pass("test_interleaved_tool_pairs")


# =============================================================================
# Edge Cases - Message Content Variations
# =============================================================================


def test_message_with_empty_content():
    """Test message with empty string content."""
    handler = create_handler()
    messages = [
        create_user_msg(""),  # Empty content
        create_assistant_msg("Response"),
    ]
    threshold, reserve = 1000, 500
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,
            memory_compact_reserve=reserve,
        ),
    )
    assert len(to_compact) + len(to_keep) == 2
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_message_with_empty_content",
    )
    print_pass("test_message_with_empty_content")


def test_message_with_whitespace_only():
    """Test message with whitespace-only content."""
    handler = create_handler()
    messages = [
        create_user_msg("   \n\t  "),  # Whitespace only
        create_assistant_msg("Response"),
    ]
    threshold, reserve = 1000, 500
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,
            memory_compact_reserve=reserve,
        ),
    )
    assert len(to_compact) + len(to_keep) == 2
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_message_with_whitespace_only",
    )
    print_pass("test_message_with_whitespace_only")


def test_very_long_single_message():
    """Test very long single message that exceeds any reasonable reserve."""
    handler = create_handler()
    huge_content = "x" * 100000  # Very long
    messages = [create_user_msg(huge_content)]
    threshold, reserve = 100, 1000
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,
            memory_compact_reserve=reserve,
        ),
    )
    # Single huge message - either kept alone or compacted
    assert len(to_compact) + len(to_keep) == 1
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_very_long_single_message",
    )
    print_pass("test_very_long_single_message")


def test_many_small_messages():
    """Test many small messages."""
    handler = create_handler()
    messages = [create_user_msg(f"Msg {i}") for i in range(100)]
    threshold, reserve = 100, 200
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,  # Low threshold
            memory_compact_reserve=reserve,
        ),
    )
    # Should compact older messages and keep recent ones
    assert len(to_compact) + len(to_keep) == 100
    assert len(to_keep) > 0, "Should keep some messages"
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_many_small_messages",
    )
    print_pass("test_many_small_messages")


def test_unicode_content():
    """Test messages with unicode characters."""
    handler = create_handler()
    messages = [
        create_user_msg("你好世界！🎉 Emoji and 中文"),
        create_assistant_msg("مرحبا العالم 🌍 Arabic and more"),
        create_user_msg("日本語テスト 🇯🇵"),
    ]
    threshold, reserve = 1000, 500
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,
            memory_compact_reserve=reserve,
        ),
    )
    assert len(to_compact) + len(to_keep) == 3
    verify_context_check_invariants(handler, messages, to_compact, to_keep, threshold, reserve, "test_unicode_content")
    print_pass("test_unicode_content")


def test_special_characters_content():
    """Test messages with special characters."""
    handler = create_handler()
    messages = [
        create_user_msg("Special chars: <>&\"'`~!@#$%^&*()[]{}|\\"),
        create_assistant_msg("More: \n\r\t\0 nulls and newlines"),
    ]
    threshold, reserve = 1000, 500
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,
            memory_compact_reserve=reserve,
        ),
    )
    assert len(to_compact) + len(to_keep) == 2
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_special_characters_content",
    )
    print_pass("test_special_characters_content")


# =============================================================================
# Edge Cases - Boundary Conditions
# =============================================================================


def test_all_messages_fit_exactly_in_reserve():
    """Test when all messages fit exactly in reserve after threshold exceeded."""
    handler = create_handler()
    messages = [
        create_user_msg("Message 1"),
        create_assistant_msg("Message 2"),
    ]

    # Calculate total tokens
    total = sum(asyncio.run(handler.stat_message(m)).total_tokens for m in messages)
    threshold, reserve = total - 1, total

    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,  # Just below total to trigger
            memory_compact_reserve=reserve,  # Exactly fits all
        ),
    )
    # All should be kept since reserve can hold everything
    assert len(to_keep) == 2, f"All messages should fit in reserve, got {len(to_keep)}"
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_all_messages_fit_exactly_in_reserve",
    )
    print_pass("test_all_messages_fit_exactly_in_reserve")


def test_first_message_only_compacted():
    """Test when only the first message is compacted."""
    handler = create_handler()
    messages = [
        create_user_msg("Large first message " * 100),  # Large
        create_assistant_msg("Small"),  # Small
        create_user_msg("Tiny"),  # Tiny
    ]

    # Calculate tokens to set appropriate reserve
    small_msg_tokens = asyncio.run(handler.stat_message(messages[1])).total_tokens
    tiny_msg_tokens = asyncio.run(handler.stat_message(messages[2])).total_tokens
    threshold, reserve = 50, small_msg_tokens + tiny_msg_tokens + 10

    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,  # Low to trigger
            memory_compact_reserve=reserve,  # Fits last 2
        ),
    )

    assert len(to_compact) >= 1, "At least first message should be compacted"
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_first_message_only_compacted",
    )
    print_pass("test_first_message_only_compacted")


def test_last_message_only_kept():
    """Test when only the last message can be kept."""
    handler = create_handler()
    messages = [
        create_user_msg("Large " * 200),
        create_assistant_msg("Large " * 200),
        create_user_msg("Tiny"),  # Only this fits
    ]

    tiny_tokens = asyncio.run(handler.stat_message(messages[2])).total_tokens
    threshold, reserve = 10, tiny_tokens + 5

    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,
            memory_compact_reserve=reserve,  # Only fits last message
        ),
    )

    if len(to_keep) == 1:
        # Last message should be the one kept
        assert to_keep[0] == messages[2], "Only last message should be kept"
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_last_message_only_kept",
    )
    print_pass("test_last_message_only_kept")


def test_all_messages_compacted():
    """Test when all messages need to be compacted (nothing fits in reserve)."""
    handler = create_handler()
    messages = [
        create_user_msg("Large message " * 100),
        create_assistant_msg("Large message " * 100),
    ]
    threshold, reserve = 10, 1
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,  # Trigger compaction
            memory_compact_reserve=reserve,  # Too small for anything
        ),
    )
    assert len(to_compact) == 2, "All messages should be compacted"
    assert len(to_keep) == 0, "No messages should be kept"
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_all_messages_compacted",
    )
    print_pass("test_all_messages_compacted")


# =============================================================================
# Edge Cases - Message Roles
# =============================================================================


def test_system_message():
    """Test handling of system role messages."""
    handler = create_handler()
    system_msg = Msg(name="system", role="system", content="You are a helpful assistant.")
    messages = [
        system_msg,
        create_user_msg("Hello"),
        create_assistant_msg("Hi there!"),
    ]
    threshold, reserve = 1000, 500
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,
            memory_compact_reserve=reserve,
        ),
    )
    assert len(to_compact) + len(to_keep) == 3
    verify_context_check_invariants(handler, messages, to_compact, to_keep, threshold, reserve, "test_system_message")
    print_pass("test_system_message")


def test_mixed_roles():
    """Test messages with various roles (user, assistant, system)."""
    handler = create_handler()
    # agentscope.message.Msg only supports: user, assistant, system
    messages = [
        Msg(name="system", role="system", content="System prompt"),
        Msg(name="user", role="user", content="User message"),
        Msg(name="assistant", role="assistant", content="Assistant response"),
        Msg(name="tool", role="user", content="Tool output as user role"),
        Msg(name="helper", role="assistant", content="Another assistant message"),
    ]
    threshold, reserve = 1000, 500
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,
            memory_compact_reserve=reserve,
        ),
    )
    assert len(to_compact) + len(to_keep) == 5
    verify_context_check_invariants(handler, messages, to_compact, to_keep, threshold, reserve, "test_mixed_roles")
    print_pass("test_mixed_roles")


# =============================================================================
# Edge Cases - Tool Block Variations
# =============================================================================


def test_tool_use_with_empty_id():
    """Test tool_use block with empty id."""
    handler = create_handler()
    messages = [
        create_user_msg("Run tool"),
        create_tool_use_msg("", "test_tool", {"arg": "value"}),  # Empty ID
        create_assistant_msg("Done"),
    ]
    threshold, reserve = 10, 1000
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,
            memory_compact_reserve=reserve,
        ),
    )
    # Should handle gracefully
    assert len(to_compact) + len(to_keep) == 3
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_tool_use_with_empty_id",
    )
    print_pass("test_tool_use_with_empty_id")


def test_tool_result_with_empty_id():
    """Test tool_result block with empty id."""
    handler = create_handler()
    messages = [
        create_user_msg("Got result"),
        create_tool_result_msg("", "test_tool", "Output"),  # Empty ID
        create_assistant_msg("Noted"),
    ]
    threshold, reserve = 10, 1000
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,
            memory_compact_reserve=reserve,
        ),
    )
    # Should handle gracefully
    assert len(to_compact) + len(to_keep) == 3
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_tool_result_with_empty_id",
    )
    print_pass("test_tool_result_with_empty_id")


def test_duplicate_tool_ids():
    """Test messages with duplicate tool IDs (unusual but possible)."""
    handler = create_handler()
    messages = [
        create_tool_use_msg("call_dup", "tool_a", {"a": 1}),
        create_tool_result_msg("call_dup", "tool_a", "Result A"),
        create_tool_use_msg("call_dup", "tool_b", {"b": 2}),  # Same ID, different tool
        create_tool_result_msg("call_dup", "tool_b", "Result B"),
    ]
    threshold, reserve = 10, 1000
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,
            memory_compact_reserve=reserve,
        ),
    )
    # Should not crash with duplicate IDs
    assert len(to_compact) + len(to_keep) == 4
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_duplicate_tool_ids",
    )
    print_pass("test_duplicate_tool_ids")


def test_message_with_multiple_tool_blocks():
    """Test single message containing multiple tool blocks."""
    handler = create_handler()
    msg_with_multiple_tools = Msg(
        name="assistant",
        role="assistant",
        content=[
            {"type": "tool_use", "id": "call_1", "name": "tool1", "input": {}},
            {"type": "tool_use", "id": "call_2", "name": "tool2", "input": {}},
            {"type": "tool_use", "id": "call_3", "name": "tool3", "input": {}},
        ],
    )
    messages = [
        create_user_msg("Do multiple things"),
        msg_with_multiple_tools,
        create_tool_result_msg("call_1", "tool1", "Result 1"),
        create_tool_result_msg("call_2", "tool2", "Result 2"),
        create_tool_result_msg("call_3", "tool3", "Result 3"),
    ]
    threshold, reserve = 10, 2000
    to_compact, to_keep, _ = asyncio.run(
        handler.context_check(
            messages=messages,
            memory_compact_threshold=threshold,
            memory_compact_reserve=reserve,
        ),
    )
    assert len(to_compact) + len(to_keep) == 5
    verify_context_check_invariants(
        handler,
        messages,
        to_compact,
        to_keep,
        threshold,
        reserve,
        "test_message_with_multiple_tool_blocks",
    )
    print_pass("test_message_with_multiple_tool_blocks")


# =============================================================================
# Run All Tests
# =============================================================================


def run_all_tests():
    """Run all tests."""
    tests = [
        # Normal cases
        test_empty_messages,
        test_below_threshold_returns_all,
        test_above_threshold_triggers_compaction,
        test_message_order_preserved,
        # Edge cases - boundaries
        test_single_message_below_threshold,
        test_single_message_above_threshold,
        test_reserve_zero,
        test_threshold_zero,
        test_exact_threshold_boundary,
        test_reserve_larger_than_threshold,
        # Edge cases - tool pairing
        test_tool_use_result_paired,
        test_tool_use_without_result,
        test_tool_result_without_use,
        test_multiple_tool_pairs,
        test_tool_dependency_causes_extra_inclusion,
        test_tool_dependency_exceeds_reserve,
        test_interleaved_tool_pairs,
        # Edge cases - content variations
        test_message_with_empty_content,
        test_message_with_whitespace_only,
        test_very_long_single_message,
        test_many_small_messages,
        test_unicode_content,
        test_special_characters_content,
        # Edge cases - boundaries
        test_all_messages_fit_exactly_in_reserve,
        test_first_message_only_compacted,
        test_last_message_only_kept,
        test_all_messages_compacted,
        # Edge cases - roles
        test_system_message,
        test_mixed_roles,
        # Edge cases - tool blocks
        test_tool_use_with_empty_id,
        test_tool_result_with_empty_id,
        test_duplicate_tool_ids,
        test_message_with_multiple_tool_blocks,
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

    # Print summary
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
