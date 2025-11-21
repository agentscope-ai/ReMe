"""Test script for ContextCompactOp.

This script provides test cases for ContextCompactOp class.
It can be run directly with: python test_context_compact_op.py
"""

import asyncio

from flowllm.core.enumeration import Role
from flowllm.core.schema import Message

from reme_ai.context.file_tool import BatchWriteFileOp
from reme_ai.context.offload import ContextCompactOp
from reme_ai.main import ReMeApp


async def async_main():
    """Test function for ContextCompactOp."""
    async with ReMeApp():
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

        # Create op with lower thresholds for testing
        op = ContextCompactOp() >> BatchWriteFileOp()

        # Execute the compaction
        await op.async_call(
            messages=[m.model_dump() for m in messages],
            max_total_tokens=1000,  # Low threshold to trigger compaction
            max_tool_message_tokens=100,  # Low threshold to compact tool messages
            preview_char_length=50,  # Keep 50 chars in preview
            keep_recent_count=1,  # Keep 1 recent tool message
            store_dir="./test_compact_storage",
        )

        # Print results
        result = op.context.response.answer
        print(f"Context compaction result: {result}")


if __name__ == "__main__":
    asyncio.run(async_main())
