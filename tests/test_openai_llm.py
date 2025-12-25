"""
Unit tests for OpenAILLM covering:
- Sync non-streaming chat
- Sync streaming chat
- Async non-streaming chat
- Async streaming chat
"""
import asyncio
import sys
from pathlib import Path


from reme_ai.core.utils import load_env

load_env()

from reme_ai.core.llm import OpenAILLM
from reme_ai.core.schema import Message, ToolCall
from reme_ai.core.enumeration import Role, ChunkEnum


def get_llm() -> OpenAILLM:
    """Create and return an OpenAILLM instance."""
    return OpenAILLM(
        model_name="qwen3-30b-a3b-instruct-2507",
        max_retries=2,
        raise_exception=True,
    )


def get_test_messages() -> list[Message]:
    """Create test messages for chat."""
    return [
        Message(role=Role.USER, content="Say 'Hello' in one word only.")
    ]


def get_test_tools() -> list[ToolCall]:
    """Create test tools for tool calling."""
    return [
        ToolCall(
            **{
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "input_schema": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. San Francisco",
                        "required": True,
                    },
                    "unit": {
                        "type": "string",
                        "description": "Temperature unit",
                        "required": False,
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
            }
        )
    ]


def get_tool_test_messages() -> list[Message]:
    """Create test messages that should trigger tool calling."""
    return [
        Message(
            role=Role.USER,
            content="What's the weather like in San Francisco?",
        )
    ]


def test_sync_chat():
    """Test synchronous non-streaming chat."""
    print("\n=== Test: Sync Non-Streaming Chat ===")
    llm = get_llm()
    messages = get_test_messages()

    response = llm.chat(messages=messages)

    assert response is not None
    assert response.role == Role.ASSISTANT
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    print(f"Response: {response.content}")
    print(f"Full message:\n{response.simple_dump()}")

    llm.close()
    print("PASSED: Sync non-streaming chat")


def test_sync_stream_chat():
    """Test synchronous streaming chat."""
    print("\n=== Test: Sync Streaming Chat ===")
    llm = get_llm()
    messages = get_test_messages()

    chunks = []
    answer_content = ""

    for chunk in llm.stream_chat(messages=messages):
        chunks.append(chunk)
        if chunk.chunk_type == ChunkEnum.ANSWER:
            answer_content += chunk.chunk
            print(chunk.chunk, end="", flush=True)

    print()

    assert len(chunks) > 0
    assert len(answer_content) > 0

    # Check that we received at least one ANSWER or USAGE chunk
    chunk_types = [c.chunk_type for c in chunks]
    assert ChunkEnum.ANSWER in chunk_types or ChunkEnum.USAGE in chunk_types
    
    # Print the final assembled message
    if chunks and hasattr(chunks[-1], 'message') and chunks[-1].message:
        print(f"Full message:\n{chunks[-1].message.simple_dump()}")

    llm.close()
    print("PASSED: Sync streaming chat")


def test_async_chat():
    """Test asynchronous non-streaming chat."""
    print("\n=== Test: Async Non-Streaming Chat ===")

    async def _test():
        llm = get_llm()
        messages = get_test_messages()

        response = await llm.achat(messages=messages)

        assert response is not None
        assert response.role == Role.ASSISTANT
        assert isinstance(response.content, str)
        assert len(response.content) > 0
        print(f"Response: {response.content}")
        print(f"Full message:\n{response.simple_dump()}")

        await llm.async_close()
        return response

    response = asyncio.run(_test())
    print("PASSED: Async non-streaming chat")


def test_async_stream_chat():
    """Test asynchronous streaming chat."""
    print("\n=== Test: Async Streaming Chat ===")

    async def _test():
        llm = get_llm()
        messages = get_test_messages()

        chunks = []
        answer_content = ""

        async for chunk in llm.astream_chat(messages=messages):
            chunks.append(chunk)
            if chunk.chunk_type == ChunkEnum.ANSWER:
                answer_content += chunk.chunk
                print(chunk.chunk, end="", flush=True)

        print()

        assert len(chunks) > 0
        assert len(answer_content) > 0

        # Check that we received at least one ANSWER or USAGE chunk
        chunk_types = [c.chunk_type for c in chunks]
        assert ChunkEnum.ANSWER in chunk_types or ChunkEnum.USAGE in chunk_types
        
        # Print the final assembled message
        if chunks and hasattr(chunks[-1], 'message') and chunks[-1].message:
            print(f"Full message:\n{chunks[-1].message.simple_dump()}")

        await llm.async_close()
        return chunks

    asyncio.run(_test())
    print("PASSED: Async streaming chat")


def test_sync_chat_with_stream_print():
    """Test synchronous chat with stream print enabled."""
    print("\n=== Test: Sync Chat with Stream Print ===")
    llm = get_llm()
    messages = get_test_messages()

    response = llm.chat(messages=messages, enable_stream_print=True)

    assert response is not None
    assert response.role == Role.ASSISTANT
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    print(f"\nFull message:\n{response.simple_dump()}")

    llm.close()
    print("PASSED: Sync chat with stream print")


def test_async_chat_with_stream_print():
    """Test asynchronous chat with stream print enabled."""
    print("\n=== Test: Async Chat with Stream Print ===")

    async def _test():
        llm = get_llm()
        messages = get_test_messages()

        response = await llm.achat(messages=messages, enable_stream_print=True)

        assert response is not None
        assert response.role == Role.ASSISTANT
        assert isinstance(response.content, str)
        assert len(response.content) > 0
        print(f"\nFull message:\n{response.simple_dump()}")

        await llm.async_close()
        return response

    asyncio.run(_test())
    print("PASSED: Async chat with stream print")


def test_sync_chat_with_tools():
    """Test synchronous non-streaming chat with tool calling."""
    print("\n=== Test: Sync Non-Streaming Chat with Tools ===")
    llm = get_llm()
    messages = get_tool_test_messages()
    tools = get_test_tools()

    response = llm.chat(messages=messages, tools=tools)

    assert response is not None
    assert response.role == Role.ASSISTANT
    # Response should contain either content or tool_calls
    assert response.content or response.tool_calls
    
    if response.tool_calls:
        print(f"Tool calls: {len(response.tool_calls)}")
        for tool_call in response.tool_calls:
            print(f"  - Tool: {tool_call.name}")
            print(f"    Arguments: {tool_call.arguments}")
            # Validate that arguments are valid JSON
            assert tool_call.check_argument()
    else:
        print(f"Response (no tool call): {response.content}")
    
    print(f"Full message:\n{response.simple_dump()}")

    llm.close()
    print("PASSED: Sync chat with tools")


def test_async_chat_with_tools():
    """Test asynchronous non-streaming chat with tool calling."""
    print("\n=== Test: Async Non-Streaming Chat with Tools ===")

    async def _test():
        llm = get_llm()
        messages = get_tool_test_messages()
        tools = get_test_tools()

        response = await llm.achat(messages=messages, tools=tools)

        assert response is not None
        assert response.role == Role.ASSISTANT
        # Response should contain either content or tool_calls
        assert response.content or response.tool_calls
        
        if response.tool_calls:
            print(f"Tool calls: {len(response.tool_calls)}")
            for tool_call in response.tool_calls:
                print(f"  - Tool: {tool_call.name}")
                print(f"    Arguments: {tool_call.arguments}")
                # Validate that arguments are valid JSON
                assert tool_call.check_argument()
        else:
            print(f"Response (no tool call): {response.content}")
        
        print(f"Full message:\n{response.simple_dump()}")

        await llm.async_close()
        return response

    asyncio.run(_test())
    print("PASSED: Async chat with tools")


if __name__ == "__main__":
    test_sync_chat()
    test_sync_stream_chat()
    test_async_chat()
    test_async_stream_chat()
    test_sync_chat_with_stream_print()
    test_async_chat_with_stream_print()
    test_sync_chat_with_tools()
    test_async_chat_with_tools()
    print("\n" + "=" * 50)
    print("All OpenAILLM tests passed!")

