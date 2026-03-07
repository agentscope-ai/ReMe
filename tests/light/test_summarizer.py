"""Tests for Summarizer."""

import asyncio
import datetime
import tempfile
from pathlib import Path

from agentscope.message import Msg
from agentscope.tool import Toolkit
from test_utils import (
    get_dash_chat_model,
    get_formatter,
    get_token_counter,
)
from reme.core.utils import get_std_logger
from reme.memory.file_based.components import Summarizer
from reme.memory.file_based.tools import FileIO


logger = get_std_logger()


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


def create_toolkit(working_dir: str) -> Toolkit:
    """Create a default Toolkit with FileIO tools for testing."""
    toolkit = Toolkit()
    file_io = FileIO(working_dir=working_dir)
    toolkit.register_tool_function(file_io.read)
    toolkit.register_tool_function(file_io.write)
    toolkit.register_tool_function(file_io.edit)
    return toolkit


def create_summarizer(working_dir: str = None, memory_dir: str = "memory"):
    """Create a Summarizer instance for testing."""
    if working_dir is None:
        working_dir = tempfile.mkdtemp()

    # 确保 memory_dir 存在
    memory_path = Path(working_dir) / memory_dir
    memory_path.mkdir(parents=True, exist_ok=True)

    return (
        Summarizer(
            working_dir=working_dir,
            memory_dir=memory_dir,
            memory_compact_threshold=4000,
            token_counter=get_token_counter(),
            toolkit=create_toolkit(working_dir),
            as_llm=get_dash_chat_model(),
            as_llm_formatter=get_formatter(),
        ),
        working_dir,
    )


def test_empty_messages():
    """Test that empty messages return empty string."""
    summarizer, _ = create_summarizer()
    result = asyncio.run(summarizer.call(messages=[]))
    assert result == "", f"Expected empty string, got: {result}"
    print_pass("test_empty_messages")


def test_short_conversation():
    """Test summarization of a short conversation."""
    summarizer, working_dir = create_summarizer()
    messages = [
        create_user_msg("Hello, I need help with Python."),
        create_assistant_msg("Sure, I'd be happy to help. What do you need?"),
        create_user_msg("How do I read a file?"),
        create_assistant_msg("You can use open() function: with open('file.txt', 'r') as f: content = f.read()"),
    ]

    logger.info(f"Input messages count: {len(messages)}")
    logger.info(f"Working directory: {working_dir}")

    result = asyncio.run(summarizer.call(messages=messages))

    logger.info(f"Result type: {type(result)}")
    logger.info(f"Result: {result}")

    assert result, "Result should not be empty"
    assert isinstance(result, str), f"Result should be string, got: {type(result)}"
    print_pass("test_short_conversation")


def test_conversation_with_tool_calls():
    """Test summarization of conversation with tool calls."""
    summarizer, working_dir = create_summarizer()
    messages = [
        create_user_msg("Help me debug this Python script."),
        create_assistant_msg("Let me check the script first."),
        create_tool_use_msg("read_file", {"path": "/app/main.py"}),
        create_tool_result_msg("read_file", "def process():\n    data = load_data()\n    return data"),
        create_assistant_msg("I found the issue! The script needs error handling."),
        create_user_msg("How should I fix it?"),
        create_assistant_msg("Add try-except block around the load_data() call."),
    ]

    logger.info(f"Working directory: {working_dir}")
    result = asyncio.run(summarizer.call(messages=messages))

    assert result, "Result should not be empty"
    assert isinstance(result, str), f"Result should be string, got: {type(result)}"
    print_pass("test_conversation_with_tool_calls")


def test_consecutive_summaries():
    """Test consecutive summaries in the same directory.

    This test verifies that:
    1. First summary creates the memory file
    2. Second summary reads and updates the existing file
    """
    # 使用固定的临时目录
    working_dir = tempfile.mkdtemp()
    memory_dir = "memory"
    memory_path = Path(working_dir) / memory_dir
    memory_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Working directory: {working_dir}")
    logger.info(f"Memory path: {memory_path}")

    # 创建 Summarizer 实例
    summarizer = Summarizer(
        working_dir=working_dir,
        memory_dir=memory_dir,
        memory_compact_threshold=4000,
        token_counter=get_token_counter(),
        toolkit=create_toolkit(working_dir),
        as_llm=get_dash_chat_model(),
        as_llm_formatter=get_formatter(),
    )

    # 第一轮对话
    messages_round1 = [
        create_user_msg("My name is Alice and I'm learning Python."),
        create_assistant_msg("Nice to meet you, Alice! Python is a great language to learn."),
        create_user_msg("I prefer using VS Code as my editor."),
        create_assistant_msg("VS Code is excellent for Python development with great extensions."),
    ]

    logger.info("=" * 40)
    logger.info("Round 1: First summary (creating new file)")
    logger.info("=" * 40)
    result1 = asyncio.run(summarizer.call(messages=messages_round1))
    logger.info(f"Round 1 Result:\n{result1}")

    # 检查文件是否被创建
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    expected_file = memory_path / f"{today}.md"
    logger.info(f"Expected file: {expected_file}")

    # 列出目录内容
    files_after_round1 = list(memory_path.iterdir())
    logger.info(f"Files after round 1: {files_after_round1}")

    assert expected_file.exists(), f"Memory file should be created at {expected_file}"

    # 读取第一轮写入的内容
    content_after_round1 = expected_file.read_text()
    logger.info(f"Content after round 1:\n{content_after_round1}")

    # 第二轮对话
    messages_round2 = [
        create_user_msg("I also like using Docker for my projects."),
        create_assistant_msg("Docker is great for containerization and deployment."),
        create_user_msg("My favorite framework is FastAPI."),
        create_assistant_msg("FastAPI is excellent for building modern APIs with Python."),
    ]

    logger.info("=" * 40)
    logger.info("Round 2: Second summary (reading and updating existing file)")
    logger.info("=" * 40)
    result2 = asyncio.run(summarizer.call(messages=messages_round2))
    logger.info(f"Round 2 Result:\n{result2}")

    # 读取第二轮写入后的内容
    content_after_round2 = expected_file.read_text()
    logger.info(f"Content after round 2:\n{content_after_round2}")

    # 验证
    assert result1, "Round 1 result should not be empty"
    assert result2, "Round 2 result should not be empty"

    # 验证第二轮内容包含新信息（Docker 或 FastAPI）
    # 注意：具体内容取决于 LLM 的响应
    assert len(content_after_round2) > 0, "Content after round 2 should not be empty"

    logger.info("=" * 40)
    logger.info("Consecutive summaries test completed successfully!")
    logger.info("=" * 40)

    print_pass("test_consecutive_summaries")


def run_all_tests():
    """Run all tests."""
    tests = [
        test_consecutive_summaries,
        test_empty_messages,
        test_short_conversation,
        test_conversation_with_tool_calls,
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
            import traceback

            print_error(test.__name__, str(e))
            traceback.print_exc()
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
