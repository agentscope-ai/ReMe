"""
Async unit tests for LLM classes (OpenAILLM and LiteLLM) covering:
- Async non-streaming chat
- Async chat with stream print
- Async streaming chat
- Async chat with tools

Usage:
    python test_llm.py --openai      # Test OpenAILLM only
    python test_llm.py --litellm     # Test LiteLLM only
    python test_llm.py --all         # Test both LLMs
"""

# flake8: noqa: E402
# pylint: disable=C0413

import argparse
import asyncio
from typing import Type

from reme.core.utils import load_env

load_env()

from reme.core.llm import OpenAILLM, LiteLLM, BaseLLM
from reme.core.schema import Message, ToolCall
from reme.core.enumeration import Role, ChunkEnum


def get_llm(llm_class: Type[BaseLLM]) -> BaseLLM:
    """Create and return an LLM instance."""
    return llm_class(
        model_name="qwen3-30b-a3b-instruct-2507",
        max_retries=2,
        raise_exception=True,
    )


def get_multi_turn_messages() -> list[Message]:
    """Create multi-turn conversation messages for testing."""
    return [
        Message(
            role=Role.SYSTEM,
            content="You are a helpful AI assistant with expertise in mathematics, science, and general knowledge.",
        ),
        Message(
            role=Role.USER,
            content="Hello! I'm working on a science project about renewable energy. "
            "Can you help me understand the basics?",
        ),
        Message(
            role=Role.ASSISTANT,
            content="Of course! I'd be happy to help. Renewable energy comes from sources that naturally replenish, "
            "like solar, wind, hydro, geothermal, and biomass. What specific aspect would you like to explore?",
        ),
        Message(
            role=Role.USER,
            content="I'm particularly interested in solar energy. Can you explain how solar panels work and calculate "
            "how much energy a typical home solar system might produce?",
        ),
        Message(
            role=Role.ASSISTANT,
            content="Solar panels work through photovoltaic cells that convert sunlight into electricity. "
            "When photons hit the silicon cells, they knock electrons loose, creating an electric current."
            "\n\nFor energy calculation: A typical home solar system is 5-10kW. With average 4-5 peak sun "
            "hours per day, a 6kW system would produce approximately 24-30 kWh daily, or 720-900 kWh monthly.",
        ),
        Message(
            role=Role.USER,
            content="That's helpful! Now, given that calculation, if electricity costs $0.12 per kWh, "
            "estimate the annual savings. Also, briefly mention what factors might affect this.",
        ),
    ]


def get_test_tools() -> list[ToolCall]:
    """Create comprehensive test tools for tool calling."""
    return [
        ToolCall(
            **{
                "type": "function",
                "function": {
                    "name": "calculate_energy_savings",
                    "description": "Calculate annual energy savings based on solar production and electricity rates",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "monthly_kwh": {
                                "type": "number",
                                "description": "Monthly energy production in kWh",
                            },
                            "electricity_rate": {
                                "type": "number",
                                "description": "Electricity cost per kWh in dollars",
                            },
                            "system_efficiency": {
                                "type": "number",
                                "description": "System efficiency factor (0-1), defaults to 0.85",
                            },
                        },
                        "required": ["monthly_kwh", "electricity_rate"],
                    },
                },
            },
        ),
        ToolCall(
            **{
                "type": "function",
                "function": {
                    "name": "get_weather_data",
                    "description": "Get current weather and solar irradiance data for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name or coordinates, e.g., 'San Francisco' or '37.7749,-122.4194'",
                            },
                            "include_forecast": {
                                "type": "boolean",
                                "description": "Whether to include 7-day forecast",
                            },
                            "unit": {
                                "type": "string",
                                "description": "Temperature unit",
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
                    "name": "analyze_panel_efficiency",
                    "description": "Analyze solar panel efficiency based on various environmental factors",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "panel_type": {
                                "type": "string",
                                "description": "Type of solar panel",
                                "enum": ["monocrystalline", "polycrystalline", "thin-film"],
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Ambient temperature in Celsius",
                            },
                            "age_years": {
                                "type": "number",
                                "description": "Age of the panel in years",
                            },
                        },
                        "required": ["panel_type", "temperature"],
                    },
                },
            },
        ),
    ]


def get_tool_test_messages() -> list[Message]:
    """Create multi-turn messages that should trigger tool calling."""
    return [
        Message(
            role=Role.SYSTEM,
            content="You are a helpful assistant with access to weather and energy calculation tools. Use them when "
            "appropriate.",
        ),
        Message(
            role=Role.USER,
            content="I'm planning to install solar panels in San Francisco. Can you help me understand the weather "
            "patterns there?",
        ),
        Message(
            role=Role.ASSISTANT,
            content="I'd be happy to help! San Francisco has a Mediterranean climate with "
            "mild temperatures year-round. Let me get the current weather data for you.",
        ),
        Message(
            role=Role.USER,
            content="Great! Also, I'm considering monocrystalline panels. If my system produces 800 kWh monthly "
            "and electricity costs $0.15 per kWh, what would be my annual savings?",
        ),
    ]


async def test_async_chat(llm_class: Type[BaseLLM], llm_name: str):
    """Test asynchronous non-streaming chat with multi-turn conversation."""
    print(f"\n{'='*60}")
    print(f"Testing {llm_name}: Async Non-Streaming Chat")
    print(f"{'='*60}")

    llm = get_llm(llm_class)
    messages = get_multi_turn_messages()

    print(f"Input: {len(messages)} messages in conversation")
    print(f"Last user message: {messages[-1].content[:100]}...")

    response = await llm.chat(messages=messages)

    assert response is not None, f"{llm_name}: Response is None"
    assert response.role == Role.ASSISTANT, f"{llm_name}: Wrong role"
    assert isinstance(response.content, str), f"{llm_name}: Content is not string"
    assert len(response.content) > 0, f"{llm_name}: Empty response"

    print(f"\nResponse preview: {response.content[:200]}...")
    print(f"Full response length: {len(response.content)} characters")
    print(f"\nFull message:\n{response.simple_dump()}")

    await llm.close()
    print(f"✓ PASSED: {llm_name} async chat")


async def test_async_chat_with_stream_print(llm_class: Type[BaseLLM], llm_name: str):
    """Test asynchronous chat with stream print enabled."""
    print(f"\n{'='*60}")
    print(f"Testing {llm_name}: Async Chat with Stream Print")
    print(f"{'='*60}")

    llm = get_llm(llm_class)
    messages = get_multi_turn_messages()

    print(f"Input: {len(messages)} messages in conversation")
    print(f"Last user message: {messages[-1].content[:100]}...")
    print("\nStreaming output:")
    print("-" * 60)

    response = await llm.chat(messages=messages, enable_stream_print=True)

    print("\n" + "-" * 60)

    assert response is not None, f"{llm_name}: Response is None"
    assert response.role == Role.ASSISTANT, f"{llm_name}: Wrong role"
    assert isinstance(response.content, str), f"{llm_name}: Content is not string"
    assert len(response.content) > 0, f"{llm_name}: Empty response"

    print(f"\nFull message:\n{response.simple_dump()}")

    await llm.close()
    print(f"✓ PASSED: {llm_name} async chat with stream print")


async def test_async_stream_chat(llm_class: Type[BaseLLM], llm_name: str):
    """Test asynchronous streaming chat."""
    print(f"\n{'='*60}")
    print(f"Testing {llm_name}: Async Streaming Chat")
    print(f"{'='*60}")

    llm = get_llm(llm_class)
    messages = get_multi_turn_messages()

    print(f"Input: {len(messages)} messages in conversation")
    print(f"Last user message: {messages[-1].content[:100]}...")
    print("\nStreaming chunks:")
    print("-" * 60)

    chunks = []
    answer_content = ""

    async for chunk in llm.stream_chat(messages=messages):
        chunks.append(chunk)
        if chunk.chunk_type == ChunkEnum.ANSWER:
            answer_content += chunk.chunk
            print(chunk.chunk, end="", flush=True)

    print("\n" + "-" * 60)

    assert len(chunks) > 0, f"{llm_name}: No chunks received"
    assert len(answer_content) > 0, f"{llm_name}: Empty answer content"

    # Check that we received at least one ANSWER or USAGE chunk
    chunk_types = [c.chunk_type for c in chunks]
    assert ChunkEnum.ANSWER in chunk_types or ChunkEnum.USAGE in chunk_types, f"{llm_name}: No ANSWER or USAGE chunks"

    # Print the final assembled message
    if chunks and hasattr(chunks[-1], "message") and chunks[-1].message:
        print(f"\nFull message:\n{chunks[-1].message.simple_dump()}")

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Answer length: {len(answer_content)} characters")

    await llm.close()
    print(f"✓ PASSED: {llm_name} async streaming chat")


async def test_async_chat_with_tools(llm_class: Type[BaseLLM], llm_name: str):
    """Test asynchronous chat with tool calling."""
    print(f"\n{'='*60}")
    print(f"Testing {llm_name}: Async Chat with Tools")
    print(f"{'='*60}")

    llm = get_llm(llm_class)
    messages = get_tool_test_messages()
    tools = get_test_tools()

    print(f"Input: {len(messages)} messages, {len(tools)} tools available")
    print(f"Tools: {[tool.name for tool in tools]}")
    print(f"Last user message: {messages[-1].content[:100]}...")

    response = await llm.chat(messages=messages, tools=tools)

    assert response is not None, f"{llm_name}: Response is None"
    assert response.role == Role.ASSISTANT, f"{llm_name}: Wrong role"
    # Response should contain either content or tool_calls
    assert response.content or response.tool_calls, f"{llm_name}: No content or tool_calls"

    if response.tool_calls:
        print(f"\n✓ Tool calls detected: {len(response.tool_calls)}")
        for i, tool_call in enumerate(response.tool_calls, 1):
            print(f"\n  Tool call #{i}:")
            print(f"    - Name: {tool_call.name}")
            print(f"    - Arguments: {tool_call.arguments}")
            # Validate that arguments are valid JSON
            assert tool_call.check_argument(), f"{llm_name}: Invalid tool arguments"
            print("    - ✓ Arguments validated")
    else:
        print("\n⚠ No tool calls (response with text instead)")
        print(f"Response preview: {response.content[:200]}...")

    print(f"\nFull message:\n{response.simple_dump()}")

    await llm.close()
    print(f"✓ PASSED: {llm_name} async chat with tools")


async def run_all_tests_for_llm(llm_class: Type[BaseLLM], llm_name: str):
    """Run all tests for a specific LLM class."""
    print(f"\n\n{'#'*60}")
    print(f"# Running all tests for: {llm_name}")
    print(f"{'#'*60}")

    await test_async_chat(llm_class, llm_name)
    await test_async_chat_with_stream_print(llm_class, llm_name)
    await test_async_stream_chat(llm_class, llm_name)
    await test_async_chat_with_tools(llm_class, llm_name)

    print(f"\n{'='*60}")
    print(f"✓ All tests passed for {llm_name}!")
    print(f"{'='*60}")


async def main():
    """Main entry point for running tests."""
    parser = argparse.ArgumentParser(
        description="Run async LLM tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_llm.py --openai      # Test OpenAILLM only
  python test_llm.py --litellm     # Test LiteLLM only
  python test_llm.py --all         # Test both LLMs
        """,
    )
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Test OpenAILLM",
    )
    parser.add_argument(
        "--litellm",
        action="store_true",
        help="Test LiteLLM",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run tests for all available LLMs",
    )

    args = parser.parse_args()

    # Determine which LLMs to test
    if args.all:
        llms_to_test = [
            (OpenAILLM, "OpenAILLM"),
            (LiteLLM, "LiteLLM"),
        ]
    elif args.openai and args.litellm:
        llms_to_test = [
            (OpenAILLM, "OpenAILLM"),
            (LiteLLM, "LiteLLM"),
        ]
    elif args.openai:
        llms_to_test = [(OpenAILLM, "OpenAILLM")]
    elif args.litellm:
        llms_to_test = [(LiteLLM, "LiteLLM")]
    else:
        # Default to all LLMs if no argument provided
        llms_to_test = [
            (OpenAILLM, "OpenAILLM"),
            (LiteLLM, "LiteLLM"),
        ]
        print("No LLM specified, defaulting to --all (testing all LLMs)")
        print("Use --openai or --litellm to test a specific one\n")

    # Run tests for each LLM
    for llm_class, llm_name in llms_to_test:
        try:
            await run_all_tests_for_llm(llm_class, llm_name)
        except Exception as e:
            print(f"\n✗ FAILED: {llm_name} tests failed with error:")
            print(f"  {type(e).__name__}: {e}")
            raise

    # Final summary
    print(f"\n\n{'#'*60}")
    print("# TEST SUMMARY")
    print(f"{'#'*60}")
    print(f"✓ All tests passed for {len(llms_to_test)} LLM(s):")
    for _, llm_name in llms_to_test:
        print(f"  - {llm_name}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
