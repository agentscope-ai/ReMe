import asyncio

from flowllm.core.enumeration import Role
from flowllm.core.schema import ToolCall, Message
from loguru import logger

from react_agent_with_working_memory import ReactAgent


async def main():
    """
    End-to-end demo for using `ReactAgent` with ReMe working memory.

    The scenario is:
    - The user asks about the ReMe project's README.
    - The assistant (ReAct agent) should first use a `web_search` tool call
      (simulated here) to obtain README content.
    - The full README (multiplied by 4 to simulate a long context) is injected
      as a TOOL message.
    - Then the user asks a question that requires reading that long content:
        "According to the README, what is the quantitative effect of task memory
         in appworld?"
    - `ReactAgent` plus ReMe working memory compresses this long context and
      tries to answer while keeping token usage low.
    """

    # A fake tool_call_id to tie TOOL messages back to the ASSISTANT's tool call.
    tool_call_id = "call_6596dafa2a6a46f7a217da"

    # Load the ReMe README as the "raw data" we want the agent to read.
    # In a real scenario, this could be code, docs, logs, etc.
    with open("../../README.md", encoding="utf-8") as f:
        readme_content = f.read()

    # Build a conversation that simulates:
    # 1. System message with instructions (how to use Grep/ReadFile tools).
    # 2. User requesting to search README.
    # 3. Assistant triggering a `web_search` tool call (simulated).
    # 4. TOOL message that returns the README content (repeated 4 times).
    # 5. A follow-up user question that requires understanding that README.
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
        # Simulate the tool result: the README content is returned as if from `web_search`.
        # We repeat it 4 times to create a large context and better showcase the
        # compression ability of ReMe working memory.
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

    # Measure token count before running the agent, so we can see compression ratio.
    origin_token_count = ReactAgent.token_count(messages)

    # You can change this to any OpenAI-compatible model name that your backend exposes.
    # For example:
    #   model_name = "qwen3-30b-a3b-instruct-2507"
    model_name = "qwen3-coder-30b-a3b-instruct"

    # Initialize the agent with a maximum number of reasoning/tool steps.
    agent = ReactAgent(model_name=model_name, max_steps=50)

    # Run the ReAct loop with ReMe working memory summarization.
    messages = await agent.run(messages)

    # Count tokens again to see how much the context has been compressed.
    after_token_count = ReactAgent.token_count(messages)

    logger.info(f"result: {messages}")

    # Example numbers (from a previous run):
    #   origin_token_count: 24586 after_token_count: 1565 compress_ratio=0.06
    logger.info(
        f"origin_token_count: {origin_token_count} "
        f"after_token_count: {after_token_count} "
        f"compress_ratio={after_token_count / origin_token_count:.2f}"
    )


if __name__ == "__main__":
    # Run the async demo entrypoint.
    asyncio.run(main())
