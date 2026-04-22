"""Tests for Compactor."""

import asyncio

from agentscope.message import Msg
from test_utils import (
    get_dash_chat_model,
    get_formatter,
    get_token_counter,
)

from reme.core.utils import get_logger
from reme.memory.file_based.components import Compactor

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


def create_tool_result_msg(tool_name: str, output: str) -> Msg:
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


def create_compactor():
    """Create a Compactor instance for testing."""
    return Compactor(
        memory_compact_threshold=4000,
        as_token_counter=get_token_counter(),
        as_llm=get_dash_chat_model(),
        as_llm_formatter=get_formatter(),
        language="zh",
    )


def test_empty_messages():
    """Test that empty messages return empty string."""
    compactor = create_compactor()
    result = asyncio.run(compactor.call(messages=[]))
    assert result == "", f"Expected empty string, got: {result}"
    print("test_empty_messages PASSED")


def test_short_conversation():
    """Test compaction of a short conversation."""
    compactor = create_compactor()
    messages = [
        create_user_msg("Hello, I need help with Python."),
        create_assistant_msg("Sure, I'd be happy to help. What do you need?"),
        create_user_msg("How do I read a file?"),
        create_assistant_msg("You can use open() function: with open('file.txt', 'r') as f: content = f.read()"),
    ]

    logger.info(f"Input messages count: {len(messages)}")
    for i, msg in enumerate(messages):
        logger.debug(
            f"Message {i}: role={msg.role}, content="
            f"{msg.content[:50] if isinstance(msg.content, str) else msg.content}...",
        )

    result = asyncio.run(compactor.call(messages=messages))

    logger.info(f"Result type: {type(result)}")
    logger.info(f"Result: {result}")

    assert result, "Result should not be empty"
    assert isinstance(result, str), f"Result should be string, got: {type(result)}"
    assert "##" in result, "Result should have markdown headers"
    print_pass("test_short_conversation")


def test_medium_conversation():
    """Test compaction of a medium-length conversation with tool calls."""
    compactor = create_compactor()
    messages = [
        create_user_msg("Help me create a Python script to process data."),
        create_assistant_msg("I'll help you create a data processing script. Let me first check the data format."),
        create_tool_use_msg("read_file", {"path": "/data/input.csv"}),
        create_tool_result_msg("read_file", "id,name,value\n1,Alice,100\n2,Bob,200\n3,Charlie,300"),
        create_assistant_msg(
            "I see the data is in CSV format. Here's a script to process it:\n"
            "```python\nimport csv\n\ndef process_data(filepath):\n"
            "    with open(filepath, 'r') as f:\n        reader = csv.DictReader(f)\n"
            "        return [row for row in reader]\n```",
        ),
        create_user_msg("Can you add a filter function?"),
        create_assistant_msg(
            "Sure, here's the updated script with filtering:\n"
            "```python\ndef filter_by_value(data, min_value):\n"
            "    return [row for row in data if int(row['value']) >= min_value]\n```",
        ),
    ]

    result = asyncio.run(compactor.call(messages=messages))

    assert result, "Result should not be empty"
    assert isinstance(result, str), f"Result should be string, got: {type(result)}"
    assert "##" in result, "Result should have markdown headers"
    print_pass("test_medium_conversation")


def test_long_conversation():
    """Test compaction of a long conversation that exceeds threshold."""
    compactor = create_compactor()
    messages = []
    for i in range(100000):
        messages.append(create_user_msg(f"Question {i}: How do I implement feature {i}?"))
        messages.append(
            create_assistant_msg(
                f"Answer {i}: Here's how to implement feature {i}. " * 20,
            ),
        )

    result = asyncio.run(compactor.call(messages=messages))

    assert result, "Result should not be empty"
    assert isinstance(result, str), f"Result should be string, got: {type(result)}"
    assert "##" in result, "Result should have markdown headers"
    print_pass("test_long_conversation")


def test_with_previous_summary():
    """Test compaction with an existing previous summary."""
    compactor = create_compactor()
    previous_summary = """## Goal
User wants to build a REST API with FastAPI.

## Constraints & Preferences
- Use Python 3.10+
- Follow RESTful best practices

## Progress
### Done
- [x] Set up project structure
- [x] Created main.py with basic FastAPI app

### In Progress
- [ ] Add user authentication

### Blocked
- (none)

## Key Decisions
- **Framework**: FastAPI for performance and type hints

## Next Steps
1. Implement JWT authentication
2. Add user endpoints

## Critical Context
- Using SQLAlchemy for database
- PostgreSQL as database backend
"""

    messages = [
        create_user_msg("Let's implement the JWT authentication now."),
        create_assistant_msg("I'll implement JWT authentication. First, let me install the required packages."),
        create_tool_use_msg("run_command", {"command": "pip install python-jose[cryptography] passlib[bcrypt]"}),
        create_tool_result_msg("run_command", "Successfully installed python-jose-3.3.0 passlib-1.7.4"),
        create_assistant_msg("Dependencies installed. Now let's create the auth module with JWT token generation."),
    ]

    result = asyncio.run(
        compactor.call(
            messages=messages,
            previous_summary=previous_summary,
        ),
    )

    assert result, "Result should not be empty"
    assert isinstance(result, str), f"Result should be string, got: {type(result)}"
    assert "##" in result, "Result should have markdown headers"
    print_pass("test_with_previous_summary")


def test_conversation_with_multiple_tool_calls():
    """Test compaction of conversation with multiple sequential tool calls."""
    compactor = create_compactor()
    messages = [
        create_user_msg("Help me debug this Python script that's failing."),
        create_assistant_msg("Let me check the script first."),
        create_tool_use_msg("read_file", {"path": "/app/main.py"}),
        create_tool_result_msg(
            "read_file",
            "def process():\n    data = load_data()\n    result = analyze(data)\n    return result",
        ),
        create_tool_use_msg("read_file", {"path": "/app/utils.py"}),
        create_tool_result_msg("read_file", "def load_data():\n    return open('data.json').read()"),
        create_tool_use_msg("run_command", {"command": "python /app/main.py"}),
        create_tool_result_msg("run_command", "FileNotFoundError: [Errno 2] No such file or directory: 'data.json'"),
        create_assistant_msg(
            "I found the issue! The script is looking for 'data.json' "
            "in the current directory instead of an absolute path.",
        ),
        create_user_msg("How should I fix it?"),
        create_assistant_msg(
            "Update load_data() to use an absolute path:\n"
            "```python\nimport os\ndef load_data():\n"
            "    script_dir = os.path.dirname(__file__)\n"
            "    return open(os.path.join(script_dir, 'data.json')).read()\n```",
        ),
    ]

    result = asyncio.run(compactor.call(messages=messages))

    assert result, "Result should not be empty"
    assert isinstance(result, str), f"Result should be string, got: {type(result)}"
    print_pass("test_conversation_with_multiple_tool_calls")


def test_low_threshold():
    """Test compaction with low memory threshold."""
    compactor = Compactor(
        memory_compact_threshold=500,
        as_token_counter=get_token_counter(),
        as_llm=get_dash_chat_model(),
        as_llm_formatter=get_formatter(),
    )

    messages = [
        create_user_msg("Build a web scraper."),
        create_assistant_msg("I'll create a web scraper using BeautifulSoup and requests."),
        create_user_msg("Make it handle pagination."),
        create_assistant_msg("Here's the paginated scraper implementation with error handling."),
    ]

    result = asyncio.run(compactor.call(messages=messages))

    assert result, "Result should not be empty"
    assert isinstance(result, str), f"Result should be string, got: {type(result)}"
    print("test_low_threshold PASSED")


def test_high_threshold():
    """Test compaction with high memory threshold."""
    compactor = Compactor(
        memory_compact_threshold=10000,
        as_token_counter=get_token_counter(),
        as_llm=get_dash_chat_model(),
        as_llm_formatter=get_formatter(),
    )

    messages = [
        create_user_msg("Create a calculator class."),
        create_assistant_msg("Here's a Calculator class with basic operations: add, subtract, multiply, divide."),
    ]

    result = asyncio.run(compactor.call(messages=messages))

    assert result, "Result should not be empty"
    assert isinstance(result, str), f"Result should be string, got: {type(result)}"
    print("test_high_threshold PASSED")


def run_all_tests():
    """Run all tests."""
    tests = [
        test_empty_messages,
        test_short_conversation,
        test_medium_conversation,
        test_long_conversation,
        test_with_previous_summary,
        test_conversation_with_multiple_tool_calls,
        test_low_threshold,
        test_high_threshold,
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
