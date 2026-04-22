"""Tests for tool operations including search and execution tools.

This module contains test functions for various tool operations such as
search tools (Dashscope, Mock, Tavily) and execution tools (Code, Shell).
"""

# pylint: disable=too-many-statements

import asyncio

from reme import ReMe


async def test_search(_app):
    """Test search tool operations.

    Tests DashscopeSearch, MockSearch, and TavilySearch operations
    with a sample query to verify they work correctly.
    """
    from reme.core.tools import DashscopeSearch, MockSearch, TavilySearch

    query = "美股DFDV是做什么的？"

    for op in [
        DashscopeSearch(),
        MockSearch(),
        TavilySearch(),
    ]:
        print("\n" + "=" * 60)
        print(f"Testing {op.__class__.__name__}")
        print("=" * 60)
        print(f"Query: {query}")
        output = await op.call(query=query, service_context=_app.service_context)
        print(f"Output:\n{output}")


async def test_execute(_app):
    """Test code and shell execution tool operations.

    Tests ExecuteCode and ExecuteShell operations with various scenarios
    including successful execution, syntax errors, runtime errors, and
    invalid commands to verify error handling.
    """
    from reme.core.tools import ExecuteCode, ExecuteShell

    # Test ExecuteCode
    print("\n" + "=" * 60)
    print("Testing ExecuteCode")
    print("=" * 60)

    op = ExecuteCode()
    code_to_execute = "print('hello world')"
    print(f"Executing Python code: {code_to_execute}")
    output = await op.call(code=code_to_execute)
    print(f"Output:\n{output}")

    # Test ExecuteCode with more complex code
    print("\n" + "=" * 60)
    print("Testing ExecuteCode with calculation")
    print("=" * 60)

    op = ExecuteCode()
    code_to_execute = "result = sum(range(1, 11))\nprint(f'Sum of 1-10: {result}')"
    print(f"Executing Python code:\n{code_to_execute}")
    output = await op.call(code=code_to_execute)
    print(f"Output:\n{output}")

    # Test ExecuteShell
    print("\n" + "=" * 60)
    print("Testing ExecuteShell")
    print("=" * 60)

    op = ExecuteShell()
    command = "ls"
    print(f"Executing shell command: {command}")
    output = await op.call(command=command)
    print(f"Output:\n{output}")

    # Test ExecuteShell with echo
    print("\n" + "=" * 60)
    print("Testing ExecuteShell with echo")
    print("=" * 60)

    op = ExecuteShell()
    command = "echo 'Hello from shell!'"
    print(f"Executing shell command: {command}")
    output = await op.call(command=command)
    print(f"Output:\n{output}")

    # Test ExecuteCode with error (syntax error)
    print("\n" + "=" * 60)
    print("Testing ExecuteCode with syntax error (expected to fail)")
    print("=" * 60)

    op = ExecuteCode()
    code_to_execute = "print('missing closing quote)"
    print(f"Executing Python code with syntax error:\n{code_to_execute}")
    output = await op.call(code=code_to_execute)
    print(f"Output:\n{output}")

    # Test ExecuteCode with runtime error
    print("\n" + "=" * 60)
    print("Testing ExecuteCode with runtime error (expected to fail)")
    print("=" * 60)

    op = ExecuteCode()
    code_to_execute = "x = 1 / 0"
    print(f"Executing Python code with runtime error:\n{code_to_execute}")
    output = await op.call(code=code_to_execute)
    print(f"Output:\n{output}")

    # Test ExecuteCode with undefined variable
    print("\n" + "=" * 60)
    print("Testing ExecuteCode with undefined variable (expected to fail)")
    print("=" * 60)

    op = ExecuteCode()
    code_to_execute = "print(undefined_variable)"
    print(f"Executing Python code with undefined variable:\n{code_to_execute}")
    output = await op.call(code=code_to_execute)
    print(f"Output:\n{output}")

    # Test ExecuteShell with invalid command
    print("\n" + "=" * 60)
    print("Testing ExecuteShell with invalid command (expected to fail)")
    print("=" * 60)

    op = ExecuteShell()
    command = "this_command_does_not_exist"
    print(f"Executing invalid shell command: {command}")
    output = await op.call(command=command)
    print(f"Output:\n{output}")

    # Test ExecuteShell with command that returns non-zero exit code
    print("\n" + "=" * 60)
    print("Testing ExecuteShell with failing command (expected to fail)")
    print("=" * 60)

    op = ExecuteShell()
    command = "ls /nonexistent_directory_12345"
    print(f"Executing shell command that should fail: {command}")
    output = await op.call(command=command)
    print(f"Output:\n{output}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


async def test_simple_chat(app):
    """Test simple chat operation.

    Tests the SimpleChat agent with a basic query to verify
    it can process and respond to user input.
    """
    from reme.extension import SimpleChat

    op = SimpleChat()
    output = await op.call(query="你好", service_context=app.service_context)
    print(output)


async def test_stream_chat(app):
    """Test streaming chat operation.

    Tests the StreamChat agent with a query to verify it can
    process and stream responses in real-time using async operations.
    """
    from reme.extension import StreamChat
    from reme.core.utils import execute_stream_task
    from reme.core import RuntimeContext
    from asyncio import Queue

    op = StreamChat()
    context = RuntimeContext(query="你好，详细介绍一下自己", stream_queue=Queue(), service_context=app.service_context)

    async def task():
        await op.call(context)
        await op.context.add_stream_done()

    async for chunk in execute_stream_task(
        stream_queue=context.stream_queue,
        task=asyncio.create_task(task()),
        task_name="test_stream_chat",
        output_format="str",
    ):
        print(chunk, end="")


async def main():
    """Main entry point for running tool tests."""
    app = ReMe()
    await app.start()
    await test_search(app)
    await test_execute(app)
    await test_simple_chat(app)
    await test_stream_chat(app)
    await app.close()


if __name__ == "__main__":
    asyncio.run(main())
