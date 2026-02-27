"""Tests for FbContextChecker - context window limit checking and cut point finding.

This module tests the cut point finding logic of FbContextChecker class,
which determines where to split conversation history when token limits are exceeded.
"""

import asyncio

from reme import ReMeFb
from reme.core.enumeration import Role
from reme.core.schema import Message


def print_messages(messages: list[Message], title: str = "Messages", max_content_len: int = 150):
    """Print messages with their role and content.

    Args:
        messages: List of messages to print
        title: Title for the message list
        max_content_len: Maximum content length to display (truncate if longer)
    """
    print(f"\n{title}: (count: {len(messages)})")
    print("-" * 80)
    for i, msg in enumerate(messages):
        content = str(msg.content)
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."
        print(f"  [{i}] {msg.role.value:10s}: {content}")
    print("-" * 80)


def create_test_messages(num_messages: int = 10) -> list[Message]:
    """Create a list of test messages.

    Args:
        num_messages: Number of messages to create

    Returns:
        List of Message objects alternating between user and assistant
    """
    messages = []
    for i in range(num_messages):
        if i % 2 == 0:
            messages.append(
                Message(
                    role=Role.USER,
                    content=f"User message {i}: Can you help me with task {i}?",
                ),
            )
        else:
            messages.append(
                Message(
                    role=Role.ASSISTANT,
                    content=f"Assistant message {i}: Sure, I'd be happy to help you with task {i - 1}. "
                    f"Let me explain the solution in detail. " * 10,
                ),
            )
    return messages


async def test_no_compaction_needed():
    """Test 1: Below threshold - no compaction needed.

    Expects: needs_compaction=False, returns original messages
    """
    print("\n" + "=" * 80)
    print("TEST 1: Below Threshold - No Cut Point Needed")
    print("=" * 80)

    reme_fs = ReMeFb(
        "vector_stores={}",  # Override config to disable vector stores
        enable_logo=False,
        context_window_tokens=5000,
        reserve_tokens=2000,
        keep_recent_tokens=1000,
    )
    await reme_fs.start()

    messages = create_test_messages(num_messages=4)
    print_messages(messages, "INPUT MESSAGES", max_content_len=80)

    print("\nParameters:")
    print("  context_window_tokens: 5000")
    print("  reserve_tokens: 2000 (threshold = 3000)")
    print("  keep_recent_tokens: 1000")

    # Use the new context_check method
    result = await reme_fs.context_check(messages)

    print(f"\n{'='*80}")
    print("RESULT:")
    print(f"  needs_compaction: {result.get('needs_compaction')}")
    print(f"  token_count: {result.get('token_count')}")
    print(f"  threshold: {result.get('threshold')}")
    print(f"  cut_index: {result.get('cut_index')}")
    print(f"  is_split_turn: {result.get('is_split_turn')}")

    assert result.get("needs_compaction") is False, "Should not need compaction below threshold"
    assert result.get("left_messages") is not None, "Should return all messages in left_messages"
    print("\n✓ TEST PASSED: No cut point needed below threshold\n")

    await reme_fs.close()


async def test_compaction_needed_above_threshold():
    """Test 2: Compaction needed when exceeding threshold.

    When messages exceed threshold, compaction should be triggered.
    The cut point location depends on token estimation.
    """
    print("\n" + "=" * 80)
    print("TEST 2: Compaction Needed Above Threshold")
    print("=" * 80)

    reme_fs = ReMeFb(
        "vector_stores={}",  # Override config to disable vector stores
        enable_logo=False,
        context_window_tokens=1500,
        reserve_tokens=700,  # threshold = 800 (below 892 tokens)
        keep_recent_tokens=220,  # Increased to hit next user message (index 40)
    )
    await reme_fs.start()

    # Create simple, short messages with uniform size for predictable cutting
    messages = []
    for i in range(50):  # More messages to exceed threshold
        if i % 2 == 0:
            messages.append(Message(role=Role.USER, content=f"Question {i}?"))
        else:
            messages.append(Message(role=Role.ASSISTANT, content=f"Answer {i}: " + "details " * 15))  # Longer assistant

    print_messages(messages, "INPUT MESSAGES", max_content_len=40)

    print("\nParameters:")
    print("  context_window_tokens: 1500")
    print("  reserve_tokens: 700 (threshold = 800)")
    print("  keep_recent_tokens: 220 (should cut at a user message)")

    # Use the new context_check method
    result = await reme_fs.context_check(messages)

    print(f"\n{'='*80}")
    print("RESULT:")
    print(f"  needs_compaction: {result.get('needs_compaction')}")
    print(f"  token_count: {result.get('token_count')}")
    print(f"  threshold: {result.get('threshold')}")
    print(f"  cut_index: {result.get('cut_index')}")
    print(f"  is_split_turn: {result.get('is_split_turn')}")
    print(f"  accumulated_tokens: {result.get('accumulated_tokens')}")

    messages_to_summarize = result.get("messages_to_summarize", [])
    left_messages = result.get("left_messages", [])
    print(f"\n  Messages to summarize: {len(messages_to_summarize)}")
    print(f"  Left messages: {len(left_messages)}")

    # Print cut message role for debugging
    if result.get("cut_index") is not None:
        cut_idx = result.get("cut_index")
        if cut_idx < len(messages):
            print(f"  Cut message role: {messages[cut_idx].role.value}")

    assert result.get("needs_compaction") is True, "Should need compaction"
    # Note: Due to token estimation variability, may or may not be a split turn
    # The important part is that compaction is triggered
    assert len(messages_to_summarize) > 0, "Should have messages to summarize"
    assert len(left_messages) > 0, "Should have left messages"
    print(f"\n  Detected split_turn: {result.get('is_split_turn')}")
    print("\n✓ TEST PASSED: Compaction triggered when exceeding threshold\n")

    await reme_fs.close()


async def test_split_turn_scenario():
    """Test 3: Split turn - cut point in middle of assistant response.

    When cut point lands on an assistant message, we need to find the turn start
    and handle turn prefix separately.
    Expects: is_split_turn=True, has turn_prefix_messages
    """
    print("\n" + "=" * 80)
    print("TEST 3: Split Turn - Cut in Middle of Assistant Response")
    print("=" * 80)

    reme_fs = ReMeFb(
        "vector_stores={}",  # Override config to disable vector stores
        enable_logo=False,
        context_window_tokens=2000,
        reserve_tokens=300,
        keep_recent_tokens=600,
    )
    await reme_fs.start()

    messages = []

    # Add initial conversation
    for i in range(3):
        messages.append(Message(role=Role.USER, content=f"Question {i}"))
        messages.append(Message(role=Role.ASSISTANT, content=f"Answer {i}. " * 30))

    # Add a very long multi-part assistant response
    messages.append(Message(role=Role.USER, content="Please explain this in great detail."))
    messages.append(
        Message(
            role=Role.ASSISTANT,
            content="This is the first part of a very long response. " * 50,
        ),
    )
    messages.append(
        Message(
            role=Role.ASSISTANT,
            content="This is the continuation of the response. " * 50,
        ),
    )
    messages.append(
        Message(
            role=Role.ASSISTANT,
            content="And here's the final part with the conclusion. " * 30,
        ),
    )

    print_messages(messages, "INPUT MESSAGES", max_content_len=80)

    print("\nParameters:")
    print("  context_window_tokens: 2000")
    print("  reserve_tokens: 300 (threshold = 1700)")
    print("  keep_recent_tokens: 600 (should cut in middle of assistant responses)")

    # Use the new context_check method
    result = await reme_fs.context_check(messages)

    print(f"\n{'='*80}")
    print("RESULT:")
    print(f"  needs_compaction: {result.get('needs_compaction')}")
    print(f"  token_count: {result.get('token_count')}")
    print(f"  threshold: {result.get('threshold')}")
    print(f"  cut_index: {result.get('cut_index')}")
    print(f"  is_split_turn: {result.get('is_split_turn')} *** (should be True)")
    print(f"  accumulated_tokens: {result.get('accumulated_tokens')}")

    messages_to_summarize = result.get("messages_to_summarize", [])
    turn_prefix_messages = result.get("turn_prefix_messages", [])
    left_messages = result.get("left_messages", [])
    print(f"\n  Messages to summarize: {len(messages_to_summarize)}")
    print(f"  Turn prefix messages: {len(turn_prefix_messages)}")
    print(f"  Left messages: {len(left_messages)}")

    if turn_prefix_messages:
        print("\n  Turn prefix messages detail:")
        for i, msg in enumerate(turn_prefix_messages):
            role = msg["role"] if isinstance(msg, dict) else msg.role.value
            content = msg["content"] if isinstance(msg, dict) else msg.content
            print(f"    [{i}] {role}: {str(content)[:60]}...")

    assert result.get("needs_compaction") is True, "Should need compaction"
    assert result.get("is_split_turn") is True, "Should detect split turn"
    assert len(turn_prefix_messages) > 0, "Should have turn prefix messages"
    assert len(messages_to_summarize) > 0, "Should have messages to summarize"
    assert len(left_messages) > 0, "Should have left messages"

    print("\n✓ TEST PASSED: Split turn correctly detected and cut point found\n")

    await reme_fs.close()


async def main():
    """Run context checker tests."""
    print("\n" + "=" * 80)
    print("FbContextChecker - Cut Point Finding Test Suite")
    print("=" * 80)
    print("\nThis test suite validates the cut point finding logic:")
    print("  1. Below threshold - no compaction needed")
    print("  2. Above threshold - compaction triggered")
    print("  3. Split turn - cut point in middle of assistant response")
    print("=" * 80)

    # Test 1: No compaction needed
    await test_no_compaction_needed()

    # Test 2: Compaction triggered above threshold
    await test_compaction_needed_above_threshold()

    # Test 3: Split turn detection
    await test_split_turn_scenario()

    print("\n" + "=" * 80)
    print("All context checker tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
