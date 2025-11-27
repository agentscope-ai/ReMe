import json
from typing import List, Dict

from flowllm.core.enumeration import Role
from flowllm.core.schema import ToolCall, Message
from flowllm.core.token import BaseToken
from flowllm.core.utils import load_env
from loguru import logger

load_env()

from flowllm.core.llm import OpenAICompatibleLLM
from flowllm.core.utils import FastMcpClient, HttpClient


class ReactAgent:
    """
    A simple ReAct-style agent that:

    - Talks to an OpenAI-compatible LLM backend.
    - Uses ReMe's working memory summary flow (`summary_working_memory`) to
      automatically compress context before each reasoning step.
    - Calls two MCP tools exposed by the ReMe backend:
        - `grep_working_memory`: search working memory by keyword / regex.
        - `read_working_memory`: read a specific segment of working memory.

    This demo focuses on how to integrate ReMe working memory into a ReAct loop,
    rather than on complex agent logic.
    """

    def __init__(self,
                 model_name="",
                 max_steps: int = 50):

        # You can replace this with your own LLM wrapper if needed.
        self.llm = OpenAICompatibleLLM(model_name=model_name)
        self.max_steps = max_steps

    @staticmethod
    def token_count(messages: List[Message]):
        """
        Count tokens for a list of `Message` objects using FlowLLM's token counter.

        This is useful for measuring compression effects of working memory summary.
        """
        return BaseToken().token_count(messages)

    async def run(self, messages: List[Message]):
        """
        Run the ReAct loop with ReMe working memory.

        Requirements (services to start before running this demo):

        - ReMe MCP server, exposing `grep_working_memory` and `read_working_memory` tools, e.g.:
            `reme backend=mcp mcp.port=8002 ...`
        - ReMe HTTP server, exposing the `summary_working_memory` flow, e.g.:
            `reme backend=http http.port=8003 ...`

        The loop roughly does:
        1. Summarize / compress the conversation with `summary_working_memory`.
        2. Ask the LLM what to do next (possibly call tools).
        3. Execute tool calls via MCP if requested, append tool results.
        4. Repeat until the model no longer requests tools or `max_steps` is reached.
        """

        # Prepare all available tools from the MCP server.
        tool_dict: Dict[str, ToolCall] = {}
        async with FastMcpClient("reme_mcp_server", {
            "type": "sse",
            "url": "http://0.0.0.0:8002/sse",
        }) as mcp_client, HttpClient(base_url="http://localhost:8003") as http_client:
            tool_calls = await mcp_client.list_tool_calls()

            for tool_call in tool_calls:
                if tool_call.name in ["grep_working_memory", "read_working_memory"]:
                    tool_dict[tool_call.name] = tool_call

                    # Log the tool call schema in Qwen3-compatible format for debugging.
                    # (This is the standard "tool" format for Qwen3 / BaiLian.)
                    tool_call_str = json.dumps(tool_call.simple_input_dump(), ensure_ascii=False, indent=2)
                    logger.info(f"tool_call {tool_call.name} {tool_call_str}")

            # Main ReAct loop.
            for i in range(self.max_steps):

                # Before every LLM call, run `summary_working_memory` to:
                # - compress long histories,
                # - offload detailed context into working memory storage,
                # - keep the recent message(s) for short-term reasoning.
                result = await http_client.execute_flow("summary_working_memory",
                                                        messages=[x.simple_dump() for x in messages],
                                                        working_summary_mode="auto",
                                                        compact_ratio_threshold=0.75,
                                                        max_total_tokens=20000,
                                                        max_tool_message_tokens=2000,
                                                        group_token_threshold=None,
                                                        keep_recent_count=1,
                                                        store_dir="./test_working_memory")

                # Convert the API result back into `Message` objects for the LLM.
                messages = [Message(**x) for x in result.answer]

                # Ask the LLM what to do next.
                # You can plug in your own tool-calling strategy here.
                assistant_message: Message = await self.llm.achat(messages=messages, tools=[
                    tool_dict["grep_working_memory"],
                    tool_dict["read_working_memory"],
                ])

                messages.append(assistant_message)

                if not assistant_message.tool_calls:
                    # If the LLM does not request any tools, we assume it has finished.
                    break

                for j, tool_call in enumerate(assistant_message.tool_calls):
                    if tool_call.name not in tool_dict:
                        logger.exception(f"unknown tool_call.name={tool_call.name}")
                        continue

                    logger.info(f"round{i + 1}.{j} submit tool_calls={tool_call.name} "
                                f"argument={tool_call.argument_dict}")

                    # Execute the tool via MCP and parse the result.
                    result = await mcp_client.call_tool(tool_call.name,
                                                        arguments=tool_call.argument_dict,
                                                        parse_result=True)

                    # Attach the tool result as a TOOL-role message so the LLM
                    # can see and reason about it in the next step.
                    messages.append(Message(
                        role=Role.TOOL,
                        tool_call_id=tool_call.id,
                        content=result,
                    ))

            return messages
