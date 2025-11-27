"""Test script for AgenticRetrieveOp.

This script provides a simple end-to-end test case for AgenticRetrieveOp.
It can be run directly with: python test_agentic_retrieve_op.py
"""

import asyncio
import json

from flowllm.core.enumeration import Role
from flowllm.core.schema import Message, ToolCall
from loguru import logger

from reme_ai.agent.react.agentic_retrieve_op import AgenticRetrieveOp
from reme_ai.main import ReMeApp


async def test_agentic_retrieve_basic():
    """Basic test for AgenticRetrieveOp with a short conversation history."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: AgenticRetrieveOp basic behavior")
    logger.info("=" * 60)

    tool_call_id = "call_6596dafa2a6a46f7a217da"
    f = open("README.md", encoding="utf-8")
    readme_content = f.read()
    f.close()

    messages = [
        Message(
            role=Role.SYSTEM,
            content=(
                "You are a helpful assistant. "
                "请先使用`Grep`匹配关键词或者正则表达式所在行数，然后通过`ReadFile`读取位置附近的代码。"
                "如果没有找到匹配项，永远不要放弃尝试，尝试其他的参数，比如只搜索部分关键词。"
                "`Grep`之后通过 `ReadFile` 命令，你可以从指定偏移位置`offset`+长度`limit`开始查看内容，不要超过100行。"
                "如果当前内容不足，`ReadFile` 命令也可以不断尝试不同的`offset`和`limit`参数"
            ),
        ),
        Message(
            role=Role.USER,
            content="搜索下reme项目的的README内容",
        ),
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[
                ToolCall(
                    **{
                        "index": 0,
                        "id": tool_call_id,
                        "function": {
                            "arguments": '{"query": "readme"}',
                            "name": "web_search",
                        },
                        "type": "function",
                    },
                ),
            ],
        ),
        Message(
            role=Role.TOOL,
            content=readme_content * 4,
            tool_call_id=tool_call_id,
        ),
        Message(
            role=Role.USER,
            content="根据readme回答task memory在appworld的效果是多少，需要具体的数值",
        ),
    ]

    # llm = "qwen3_coder_plus"
    llm = "qwen3_30b_instruct"
    # llm = "qwen3_30b_thinking"
    # llm = "qwen3_coder_30b_instruct"
    # llm = "qwen3_max_instruct"
    op = AgenticRetrieveOp(llm=llm)

    await op.async_call(
        messages=[m.model_dump() for m in messages],
        working_summary_mode="auto",
        compact_ratio_threshold=0.75,
        max_total_tokens=20000,
        max_tool_message_tokens=2000,
        group_token_threshold=None,
        keep_recent_count=1,
        store_dir="./test_working_memory",
        chat_id="c123",
    )

    answer = op.context.response.answer
    messages = op.context.response.metadata["messages"]
    logger.info(f"✓ AgenticRetrieveOp result answer: {answer}")
    logger.info(f"✓ AgenticRetrieveOp result messages: {json.dumps(messages, ensure_ascii=False, indent=2)}")
    logger.info(f"  Success: {op.context.response.success}")


async def async_main():
    """Entry point for running AgenticRetrieveOp test."""
    async with ReMeApp():
        logger.info("=" * 80)
        logger.info("Testing AgenticRetrieveOp - ReAct Retrieval Workflow")
        logger.info("=" * 80)

        await test_agentic_retrieve_basic()

        logger.info("\n" + "=" * 80)
        logger.info("All AgenticRetrieveOp tests completed!")
        logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(async_main())
