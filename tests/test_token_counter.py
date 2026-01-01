"""
Unit tests for TokenCounter classes covering:
- BaseTokenCounter (rule-based estimation)
- OpenAITokenCounter (tiktoken-based)
- HFTokenCounter (HuggingFace tokenizer-based)

Usage:
    python test_token_counter.py --base        # Test BaseTokenCounter only
    python test_token_counter.py --openai      # Test OpenAITokenCounter only
    python test_token_counter.py --hf          # Test HFTokenCounter only
    python test_token_counter.py --all         # Test all token counters
"""

import argparse
from typing import Type, List

from reme_ai.core.enumeration import Role
from reme_ai.core.schema import Message, ToolCall
from reme_ai.core.token_counter import BaseTokenCounter, OpenAITokenCounter, HFTokenCounter


def get_token_counter(counter_class: Type[BaseTokenCounter], **kwargs) -> BaseTokenCounter:
    """Create and return a token counter instance."""
    default_kwargs = {
        "model_name": "gpt-4o",
    }
    default_kwargs.update(kwargs)
    return counter_class(**default_kwargs)


def get_test_messages() -> List[Message]:
    """Create test messages for token counting."""
    return [
        Message(role=Role.SYSTEM, content="You are a helpful assistant."),
        Message(role=Role.USER, content="Hello, how are you today?"),
        Message(role=Role.ASSISTANT, content="I'm doing well, thank you for asking! How can I help you?"),
        Message(role=Role.USER, content="Can you explain what machine learning is?"),
        Message(
            role=Role.ASSISTANT,
            content="Machine learning is a subset of artificial intelligence that enables computers to "
            "learn from data without being explicitly programmed.",
        ),
    ]


def get_chinese_messages() -> List[Message]:
    """Create test messages with Chinese content."""
    return [
        Message(role=Role.SYSTEM, content="你是一个有帮助的助手。"),
        Message(role=Role.USER, content="你好，今天天气怎么样？"),
        Message(role=Role.ASSISTANT, content="今天天气很好，阳光明媚，适合外出活动。"),
        Message(role=Role.USER, content="能给我推荐一些好看的电影吗？"),
        Message(role=Role.ASSISTANT, content="当然可以！我推荐《肖申克的救赎》、《阿甘正传》和《泰坦尼克号》。"),
    ]


def get_mixed_messages() -> List[Message]:
    """Create test messages with mixed English and Chinese content."""
    return [
        Message(role=Role.SYSTEM, content="You are a bilingual assistant. 你是一个双语助手。"),
        Message(role=Role.USER, content="What is AI? 什么是人工智能？"),
        Message(
            role=Role.ASSISTANT,
            content="AI (Artificial Intelligence) 是人工智能的英文缩写，它是计算机科学的一个分支。",
        ),
    ]


def get_messages_with_reasoning() -> List[Message]:
    """Create test messages with reasoning content."""
    return [
        Message(role=Role.USER, content="What is 2 + 2?"),
        Message(
            role=Role.ASSISTANT,
            content="The answer is 4.",
            reasoning_content="Let me think about this step by step. 2 + 2 "
            "equals 4 because addition combines two quantities.",
        ),
    ]


def get_test_tools() -> List[ToolCall]:
    """Create test tool calls for token counting."""
    return [
        ToolCall(
            **{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a specified location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and country, e.g., 'Beijing, China'",
                            },
                            "unit": {
                                "type": "string",
                                "description": "Temperature unit: 'celsius' or 'fahrenheit'",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            },
        ),
        ToolCall(
            **{
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web for information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query",
                            },
                            "num_results": {
                                "type": "integer",
                                "description": "Number of results to return",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ),
    ]


def get_tool_call_messages() -> List[Message]:
    """Create messages with tool call responses."""
    return [
        Message(role=Role.USER, content="What's the weather in Beijing?"),
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[
                ToolCall(
                    id="call_123",
                    name="get_weather",
                    arguments='{"location": "Beijing, China", "unit": "celsius"}',
                ),
            ],
        ),
        Message(
            role=Role.TOOL,
            content='{"temperature": 25, "condition": "sunny", "humidity": 60}',
            tool_call_id="call_123",
        ),
        Message(
            role=Role.ASSISTANT,
            content="The weather in Beijing is sunny with a temperature of 25°C and 60% humidity.",
        ),
    ]


def test_basic_token_count(counter_class: Type[BaseTokenCounter], counter_name: str, **kwargs):
    """Test basic token counting with simple messages."""
    print(f"\n{'=' * 60}")
    print(f"Testing {counter_name}: Basic Token Count")
    print(f"{'=' * 60}")

    counter = get_token_counter(counter_class, **kwargs)
    messages = get_test_messages()

    print(f"Input: {len(messages)} messages")
    for i, msg in enumerate(messages, 1):
        content_preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
        print(f"  {i}. [{msg.role.value}] {content_preview}")

    token_count = counter.count_token(messages)

    assert token_count is not None, f"{counter_name}: Token count is None"
    assert isinstance(token_count, int), f"{counter_name}: Token count is not an integer"
    assert token_count > 0, f"{counter_name}: Token count should be positive"

    print(f"\n✓ Token count: {token_count}")
    print(f"✓ PASSED: {counter_name} basic token count")


def test_chinese_token_count(counter_class: Type[BaseTokenCounter], counter_name: str, **kwargs):
    """Test token counting with Chinese content."""
    print(f"\n{'=' * 60}")
    print(f"Testing {counter_name}: Chinese Token Count")
    print(f"{'=' * 60}")

    counter = get_token_counter(counter_class, **kwargs)
    messages = get_chinese_messages()

    print(f"Input: {len(messages)} Chinese messages")
    for i, msg in enumerate(messages, 1):
        content_preview = msg.content[:30] + "..." if len(msg.content) > 30 else msg.content
        print(f"  {i}. [{msg.role.value}] {content_preview}")

    token_count = counter.count_token(messages)

    assert token_count is not None, f"{counter_name}: Token count is None"
    assert isinstance(token_count, int), f"{counter_name}: Token count is not an integer"
    assert token_count > 0, f"{counter_name}: Token count should be positive"

    print(f"\n✓ Token count: {token_count}")
    print(f"✓ PASSED: {counter_name} Chinese token count")


def test_mixed_language_token_count(counter_class: Type[BaseTokenCounter], counter_name: str, **kwargs):
    """Test token counting with mixed English and Chinese content."""
    print(f"\n{'=' * 60}")
    print(f"Testing {counter_name}: Mixed Language Token Count")
    print(f"{'=' * 60}")

    counter = get_token_counter(counter_class, **kwargs)
    messages = get_mixed_messages()

    print(f"Input: {len(messages)} mixed language messages")
    for i, msg in enumerate(messages, 1):
        content_preview = msg.content[:40] + "..." if len(msg.content) > 40 else msg.content
        print(f"  {i}. [{msg.role.value}] {content_preview}")

    token_count = counter.count_token(messages)

    assert token_count is not None, f"{counter_name}: Token count is None"
    assert isinstance(token_count, int), f"{counter_name}: Token count is not an integer"
    assert token_count > 0, f"{counter_name}: Token count should be positive"

    print(f"\n✓ Token count: {token_count}")
    print(f"✓ PASSED: {counter_name} mixed language token count")


def test_reasoning_content_token_count(counter_class: Type[BaseTokenCounter], counter_name: str, **kwargs):
    """Test token counting with reasoning content."""
    print(f"\n{'=' * 60}")
    print(f"Testing {counter_name}: Reasoning Content Token Count")
    print(f"{'=' * 60}")

    counter = get_token_counter(counter_class, **kwargs)
    messages = get_messages_with_reasoning()

    print(f"Input: {len(messages)} messages with reasoning content")
    for i, msg in enumerate(messages, 1):
        print(f"  {i}. [{msg.role.value}] content: {msg.content[:30]}...")
        if msg.reasoning_content:
            print(f"     reasoning: {msg.reasoning_content[:30]}...")

    token_count = counter.count_token(messages)

    assert token_count is not None, f"{counter_name}: Token count is None"
    assert isinstance(token_count, int), f"{counter_name}: Token count is not an integer"
    assert token_count > 0, f"{counter_name}: Token count should be positive"

    print(f"\n✓ Token count: {token_count}")
    print(f"✓ PASSED: {counter_name} reasoning content token count")


def test_token_count_with_tools(counter_class: Type[BaseTokenCounter], counter_name: str, **kwargs):
    """Test token counting with tool definitions."""
    print(f"\n{'=' * 60}")
    print(f"Testing {counter_name}: Token Count with Tools")
    print(f"{'=' * 60}")

    counter = get_token_counter(counter_class, **kwargs)
    messages = get_test_messages()[:2]
    tools = get_test_tools()

    print(f"Input: {len(messages)} messages, {len(tools)} tools")
    for tool in tools:
        print(f"  Tool: {tool.name} - {tool.description[:40]}...")

    token_count = counter.count_token(messages, tools=tools)

    assert token_count is not None, f"{counter_name}: Token count is None"
    assert isinstance(token_count, int), f"{counter_name}: Token count is not an integer"
    assert token_count > 0, f"{counter_name}: Token count should be positive"

    # Token count with tools should be higher than without
    token_count_no_tools = counter.count_token(messages)
    assert token_count > token_count_no_tools, f"{counter_name}: Token count with tools should be higher"

    print(f"\n✓ Token count without tools: {token_count_no_tools}")
    print(f"✓ Token count with tools: {token_count}")
    print(f"✓ Tools added {token_count - token_count_no_tools} tokens")
    print(f"✓ PASSED: {counter_name} token count with tools")


def test_tool_call_messages_token_count(counter_class: Type[BaseTokenCounter], counter_name: str, **kwargs):
    """Test token counting with messages containing tool calls."""
    print(f"\n{'=' * 60}")
    print(f"Testing {counter_name}: Tool Call Messages Token Count")
    print(f"{'=' * 60}")

    counter = get_token_counter(counter_class, **kwargs)
    messages = get_tool_call_messages()

    print(f"Input: {len(messages)} messages with tool calls")
    for i, msg in enumerate(messages, 1):
        if msg.tool_calls:
            print(f"  {i}. [{msg.role.value}] tool_calls: {[tc.name for tc in msg.tool_calls]}")
        else:
            content_preview = msg.content[:40] + "..." if len(msg.content) > 40 else msg.content
            print(f"  {i}. [{msg.role.value}] {content_preview}")

    token_count = counter.count_token(messages)

    assert token_count is not None, f"{counter_name}: Token count is None"
    assert isinstance(token_count, int), f"{counter_name}: Token count is not an integer"
    assert token_count > 0, f"{counter_name}: Token count should be positive"

    print(f"\n✓ Token count: {token_count}")
    print(f"✓ PASSED: {counter_name} tool call messages token count")


def test_empty_messages(counter_class: Type[BaseTokenCounter], counter_name: str, **kwargs):
    """Test token counting with empty message list."""
    print(f"\n{'=' * 60}")
    print(f"Testing {counter_name}: Empty Messages")
    print(f"{'=' * 60}")

    # HFTokenCounter does not support empty message list (apply_chat_template requires at least one message)
    if counter_class == HFTokenCounter:
        print("⊘ SKIPPED: HFTokenCounter does not support empty message list")
        return

    counter = get_token_counter(counter_class, **kwargs)
    messages: List[Message] = []

    print("Input: 0 messages")

    token_count = counter.count_token(messages)

    assert token_count is not None, f"{counter_name}: Token count is None"
    assert isinstance(token_count, int), f"{counter_name}: Token count is not an integer"

    # OpenAITokenCounter adds 3 tokens for reply priming even with empty messages
    if counter_class == OpenAITokenCounter:
        assert token_count == 3, f"{counter_name}: Empty messages should have 3 tokens (reply priming)"
        print(f"\n✓ Token count: {token_count} (includes 3 tokens for reply priming)")
    else:
        assert token_count == 0, f"{counter_name}: Empty messages should have 0 tokens"
        print(f"\n✓ Token count: {token_count}")

    print(f"✓ PASSED: {counter_name} empty messages")


def test_single_message(counter_class: Type[BaseTokenCounter], counter_name: str, **kwargs):
    """Test token counting with a single message."""
    print(f"\n{'=' * 60}")
    print(f"Testing {counter_name}: Single Message")
    print(f"{'=' * 60}")

    counter = get_token_counter(counter_class, **kwargs)
    messages = [Message(role=Role.USER, content="Hello!")]

    print(f"Input: 1 message - '{messages[0].content}'")

    token_count = counter.count_token(messages)

    assert token_count is not None, f"{counter_name}: Token count is None"
    assert isinstance(token_count, int), f"{counter_name}: Token count is not an integer"
    assert token_count > 0, f"{counter_name}: Token count should be positive"

    print(f"\n✓ Token count: {token_count}")
    print(f"✓ PASSED: {counter_name} single message")


def test_long_content(counter_class: Type[BaseTokenCounter], counter_name: str, **kwargs):
    """Test token counting with long content."""
    print(f"\n{'=' * 60}")
    print(f"Testing {counter_name}: Long Content")
    print(f"{'=' * 60}")

    counter = get_token_counter(counter_class, **kwargs)

    # Create a long message
    long_text = "This is a test sentence. " * 100
    messages = [Message(role=Role.USER, content=long_text)]

    print(f"Input: 1 message with {len(long_text)} characters")

    token_count = counter.count_token(messages)

    assert token_count is not None, f"{counter_name}: Token count is None"
    assert isinstance(token_count, int), f"{counter_name}: Token count is not an integer"
    assert token_count > 0, f"{counter_name}: Token count should be positive"

    # Long content should have more tokens
    short_messages = [Message(role=Role.USER, content="This is a test sentence.")]
    short_token_count = counter.count_token(short_messages)
    assert token_count > short_token_count, f"{counter_name}: Long content should have more tokens"

    print(f"\n✓ Short content token count: {short_token_count}")
    print(f"✓ Long content token count: {token_count}")
    print(f"✓ PASSED: {counter_name} long content")


def run_all_tests_for_counter(counter_class: Type[BaseTokenCounter], counter_name: str, **kwargs):
    """Run all tests for a specific token counter class."""
    print(f"\n\n{'#' * 60}")
    print(f"# Running all tests for: {counter_name}")
    print(f"{'#' * 60}")

    test_basic_token_count(counter_class, counter_name, **kwargs)
    test_chinese_token_count(counter_class, counter_name, **kwargs)
    test_mixed_language_token_count(counter_class, counter_name, **kwargs)
    test_reasoning_content_token_count(counter_class, counter_name, **kwargs)
    test_token_count_with_tools(counter_class, counter_name, **kwargs)
    test_tool_call_messages_token_count(counter_class, counter_name, **kwargs)
    test_empty_messages(counter_class, counter_name, **kwargs)
    test_single_message(counter_class, counter_name, **kwargs)
    test_long_content(counter_class, counter_name, **kwargs)

    print(f"\n{'=' * 60}")
    print(f"✓ All tests passed for {counter_name}!")
    print(f"{'=' * 60}")


def main():
    """Main entry point for running tests."""
    parser = argparse.ArgumentParser(
        description="Run token counter tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_token_counter.py --base        # Test BaseTokenCounter only
  python test_token_counter.py --openai      # Test OpenAITokenCounter only
  python test_token_counter.py --hf          # Test HFTokenCounter only
  python test_token_counter.py --all         # Test all token counters
        """,
    )
    parser.add_argument(
        "--base",
        action="store_true",
        help="Test BaseTokenCounter (rule-based)",
    )
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Test OpenAITokenCounter (tiktoken-based)",
    )
    parser.add_argument(
        "--hf",
        action="store_true",
        help="Test HFTokenCounter (HuggingFace tokenizer-based)",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model name for HFTokenCounter (default: Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run tests for all available token counters",
    )

    args = parser.parse_args()

    # Determine which counters to test
    counters_to_test = []

    if args.all:
        counters_to_test.append((BaseTokenCounter, "BaseTokenCounter", {}))
        counters_to_test.append((OpenAITokenCounter, "OpenAITokenCounter", {}))
        counters_to_test.append(
            (HFTokenCounter, "HFTokenCounter", {"model_name": args.hf_model, "trust_remote_code": True}),
        )
    else:
        if args.base:
            counters_to_test.append((BaseTokenCounter, "BaseTokenCounter", {}))
        if args.openai:
            counters_to_test.append((OpenAITokenCounter, "OpenAITokenCounter", {}))
        if args.hf:
            counters_to_test.append(
                (HFTokenCounter, "HFTokenCounter", {"model_name": args.hf_model, "trust_remote_code": True}),
            )

    if not counters_to_test:
        # Default to all counters if no argument provided
        counters_to_test = [
            (BaseTokenCounter, "BaseTokenCounter", {}),
            (OpenAITokenCounter, "OpenAITokenCounter", {}),
            (HFTokenCounter, "HFTokenCounter", {"model_name": args.hf_model, "trust_remote_code": True}),
        ]
        print("No counter specified, defaulting to test all counters")
        print("Use --base/--openai/--hf to test specific ones\n")

    # Run tests for each counter
    for counter_class, counter_name, kwargs in counters_to_test:
        try:
            run_all_tests_for_counter(counter_class, counter_name, **kwargs)
        except Exception as e:
            print(f"\n✗ FAILED: {counter_name} tests failed with error:")
            print(f"  {type(e).__name__}: {e}")
            raise

    # Final summary
    print(f"\n\n{'#' * 60}")
    print("# TEST SUMMARY")
    print(f"{'#' * 60}")
    print(f"✓ All tests passed for {len(counters_to_test)} token counter(s):")
    for _, counter_name, _ in counters_to_test:
        print(f"  - {counter_name}")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    main()
