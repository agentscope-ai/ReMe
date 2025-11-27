"""Test script for MessageOffloadOp.

This script provides test cases for MessageOffloadOp class.
It can be run directly with: python test_context_offload_op.py
"""

import asyncio

from flowllm.core.enumeration import Role
from flowllm.core.schema import Message
from loguru import logger

from reme_ai.enumeration import WorkingSummaryMode
from reme_ai.main import ReMeApp
from reme_ai.retrieve.working import BatchWriteFileOp
from reme_ai.summary.working import MessageOffloadOp


async def test_compact_mode():
    """Test COMPACT mode - Only apply compaction with MessageOffloadOp."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: COMPACT mode - Only apply compaction")
    logger.info("=" * 60)

    # Create test messages with system, user, assistant, tool sequence
    messages = [
        Message(role=Role.SYSTEM, content="You are a helpful assistant."),
        Message(role=Role.USER, content="What is the weather today?"),
        Message(
            role=Role.ASSISTANT,
            content="I'll check the weather for you.",
        ),
        Message(
            role=Role.TOOL,
            content="A" * 5000,  # Large tool message that should be compacted
            tool_call_id="call_001",
        ),
        Message(
            role=Role.ASSISTANT,
            content="Let me also check the forecast.",
        ),
        Message(
            role=Role.TOOL,
            content="B" * 5000,  # Another large tool message
            tool_call_id="call_002",
        ),
        Message(
            role=Role.USER,
            content="What about tomorrow?",
        ),
        Message(
            role=Role.ASSISTANT,
            content="I'll check tomorrow's weather.",
        ),
        Message(
            role=Role.TOOL,
            content="C" * 5000,  # Third large tool message
            tool_call_id="call_003",
        ),
        Message(
            role=Role.TOOL,
            content="Recent result",  # Recent tool message (should be kept)
            tool_call_id="call_004",
        ),
    ]

    op = MessageOffloadOp() >> BatchWriteFileOp()

    await op.async_call(
        messages=[m.model_dump() for m in messages],
        context_manage_mode=WorkingSummaryMode.COMPACT,
        max_total_tokens=1000,  # Low threshold to trigger compaction
        max_tool_message_tokens=100,  # Low threshold to compact tool messages
        preview_char_length=50,  # Keep 50 chars in preview
        keep_recent_count=1,  # Keep 1 recent tool message
        store_dir="./test_compact_storage",
    )

    result = op.context.response.answer
    logger.info(f"✓ COMPACT mode result: {len(result)} messages")
    logger.info(f"  Success: {op.context.response.success}")


async def test_compress_mode():
    """Test COMPRESS mode - Only apply compression with MessageOffloadOp."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: COMPRESS mode - Only apply compression")
    logger.info("=" * 60)

    # Create test messages with system, user, assistant, tool sequence
    messages = [
        Message(role=Role.SYSTEM, content="You are a helpful assistant."),
        Message(role=Role.USER, content="What is the weather today?"),
        Message(
            role=Role.ASSISTANT,
            content="I'll check the weather for you.",
        ),
        Message(
            role=Role.TOOL,
            content="A" * 5000,  # Large tool message that should be compacted
            tool_call_id="call_001",
        ),
        Message(
            role=Role.ASSISTANT,
            content="Let me also check the forecast.",
        ),
        Message(
            role=Role.TOOL,
            content="B" * 5000,  # Another large tool message
            tool_call_id="call_002",
        ),
        Message(
            role=Role.USER,
            content="What about tomorrow?",
        ),
        Message(
            role=Role.ASSISTANT,
            content="I'll check tomorrow's weather.",
        ),
        Message(
            role=Role.TOOL,
            content="C" * 5000,  # Third large tool message
            tool_call_id="call_003",
        ),
        Message(
            role=Role.TOOL,
            content="Recent result",  # Recent tool message (should be kept)
            tool_call_id="call_004",
        ),
    ]

    op = MessageOffloadOp() >> BatchWriteFileOp()

    await op.async_call(
        messages=[m.model_dump() for m in messages],
        context_manage_mode=WorkingSummaryMode.COMPRESS,
        max_total_tokens=2000,  # Low threshold to trigger compression
        keep_recent_count=2,
        store_dir="./test_compact_storage",
    )

    result = op.context.response.answer
    logger.info(f"✓ COMPRESS mode result: {len(result)} messages")
    logger.info(f"  Success: {op.context.response.success}")


async def test_auto_mode():
    """Test AUTO mode - Apply compaction first, then compression if needed using MessageOffloadOp."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: AUTO mode - Apply compaction first, then compression if needed")
    logger.info("=" * 60)

    # Create messages with extensive user content to ensure compact ratio exceeds threshold
    auto_messages = [
        Message(role=Role.SYSTEM, content="You are a helpful assistant."),
        Message(role=Role.USER, content="What is the weather today?"),
        Message(
            role=Role.ASSISTANT,
            content="I'll check the weather for you.",
        ),
        Message(
            role=Role.TOOL,
            content="A" * 5000,  # Large tool message that should be compacted
            tool_call_id="call_001",
        ),
        Message(
            role=Role.USER,
            content="I need detailed information about the weather forecast for the next week. "
            "Please provide temperature, humidity, wind speed, and precipitation chances for each day. "
            "Also, I want to know about any weather warnings or advisories. "
            "This is very important for my travel planning." * 50,  # Long user message
        ),
        Message(
            role=Role.ASSISTANT,
            content="I'll gather comprehensive weather information for you. Let me check multiple sources." * 3,
        ),
        Message(
            role=Role.TOOL,
            content="B" * 5000,  # Another large tool message
            tool_call_id="call_002",
        ),
        Message(
            role=Role.USER,
            content="Can you also provide information about air quality, UV index, and sunrise/sunset times? "
            "I'm planning outdoor activities and need to know the best times to be outside. "
            "Also, please include historical weather data for comparison." * 4,  # More long user content
        ),
        Message(
            role=Role.ASSISTANT,
            content="Absolutely! I'll get all that information for you including air quality metrics and UV data." * 2,
        ),
        Message(
            role=Role.TOOL,
            content="C" * 5000,  # Third large tool message
            tool_call_id="call_003",
        ),
        Message(
            role=Role.USER,
            content="What about tomorrow?",
        ),
        Message(
            role=Role.ASSISTANT,
            content="I'll check tomorrow's weather.",
        ),
        Message(
            role=Role.TOOL,
            content="Recent result",  # Recent tool message (should be kept)
            tool_call_id="call_004",
        ),
    ]

    op = MessageOffloadOp() >> BatchWriteFileOp()

    await op.async_call(
        messages=[m.model_dump() for m in auto_messages],
        context_manage_mode=WorkingSummaryMode.AUTO,
        compact_ratio_threshold=0.2,  # Low threshold, should trigger compression after compact
        max_total_tokens=1000,
        max_tool_message_tokens=100,
        preview_char_length=50,
        keep_recent_count=1,
        store_dir="./test_compact_storage",
    )

    result = op.context.response.answer
    logger.info(f"✓ AUTO mode result: {len(result)} messages")
    logger.info(f"  Success: {op.context.response.success}")


async def async_main():
    """Test function for MessageOffloadOp."""
    async with ReMeApp():
        logger.info("=" * 80)
        logger.info("Testing MessageOffloadOp - Context Management Orchestration")
        logger.info("=" * 80)

        await test_compact_mode()
        await test_compress_mode()
        await test_auto_mode()

        logger.info("\n" + "=" * 80)
        logger.info("All tests completed!")
        logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(async_main())
