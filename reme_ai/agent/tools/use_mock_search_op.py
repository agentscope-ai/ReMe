"""Tool selection and execution operation for mock search tools.

This module provides an operation that intelligently selects and executes
the most appropriate mock search tool based on query complexity.
"""

import asyncio
import datetime
import json

from flowllm.core.context import C
from flowllm.core.enumeration import Role
from flowllm.core.op import BaseAsyncToolOp
from flowllm.core.schema import Message
from flowllm.core.schema import ToolCall
from flowllm.core.utils import Timer
from loguru import logger

from reme_ai.agent.tools.mock_search_tools import SearchToolA, SearchToolB, SearchToolC
from reme_ai.schema.memory import ToolCallResult


@C.register_op()
class UseMockSearchOp(BaseAsyncToolOp):
    """Operation that selects and executes the most appropriate mock search tool.

    This operation uses LLM to intelligently select from available search tools
    (SearchToolA, SearchToolB, SearchToolC) based on query characteristics,
    then executes the selected tool and records performance metrics.
    """

    file_path: str = __file__

    def __init__(self, llm: str = "qwen3_30b_instruct", **kwargs):
        """Initialize the UseMockSearchOp.

        Args:
            llm: LLM model name to use for tool selection
            **kwargs: Additional arguments passed to BaseAsyncToolOp
        """
        super().__init__(llm=llm, save_answer=True, **kwargs)

    def build_tool_call(self) -> ToolCall:
        """Build the tool call schema for the search tool selector.

        Returns:
            ToolCall object defining the search tool selector interface
        """
        return ToolCall(
            **{
                "description": (
                    "Intelligently selects and executes the most appropriate search tool "
                    "based on query complexity. Automatically tracks performance metrics "
                    "and records tool usage for optimization."
                ),
                "input_schema": {
                    "query": {
                        "type": "string",
                        "description": "query",
                        "required": True,
                    },
                },
            },
        )

    async def select_tool(self, query: str, tool_ops: list[BaseAsyncToolOp]) -> ToolCall | None:
        """Select the most appropriate tool for the given query using LLM.

        Args:
            query: The search query to process
            tool_ops: List of available tool operations to choose from

        Returns:
            Selected ToolCall if a tool was chosen, None otherwise
        """
        assistant_message = await self.llm.achat(
            messages=[Message(role=Role.USER, content=query)],
            tools=[x.tool_call for x in tool_ops],
        )
        logger.info(f"assistant_message={assistant_message.model_dump_json()}")
        if assistant_message.tool_calls:
            return assistant_message.tool_calls[0]

        return None

    async def async_execute(self):
        """Execute the tool selection and execution workflow.

        This method selects an appropriate tool, executes it, measures performance,
        and creates a ToolCallResult with metrics.
        """
        query: str = self.context["query"]
        logger.info(f"query={query}")

        tool_ops = [
            SearchToolA(),
            SearchToolB(),
            SearchToolC(),
        ]

        # Step 1: Select the appropriate tool using LLM
        tool_call = await self.select_tool(query, tool_ops)

        if tool_call is None:
            # No tool selected
            error_result = ToolCallResult(
                create_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                tool_name="None",
                input={"query": query},
                output="No appropriate tool was selected for the query",
                token_cost=0,
                success=False,
                time_cost=0.0,
            )
            self.set_output(error_result.model_dump_json())
            return

            # Step 2: Execute the selected tool
        selected_op = None
        for op in tool_ops:
            if op.tool_call.name == tool_call.name:
                selected_op = op
                break

        if selected_op is None:
            # Tool not found (should not happen)
            error_result = ToolCallResult(
                create_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                tool_name=tool_call.name,
                input=tool_call.arguments,
                output=f"Tool {tool_call.name} not found in available tools",
                token_cost=0,
                success=False,
                time_cost=0.0,
            )
            self.set_output(error_result.model_dump_json())
            return

        # Step 3: Execute the tool with timer
        timer = Timer("tool execute")
        with timer:
            await selected_op.async_call(query=query)
            selected_op_output = json.loads(selected_op.output)
            content = selected_op_output["content"]
            success = selected_op_output["success"]
            token_cost = len(content) // 4  # Estimate using a method where every 4 characters constitute one token.

        time_cost = timer.time_cost

        # Create ToolCallResult
        tool_call_result = ToolCallResult(
            create_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            tool_name=tool_call.name,
            input={"query": query},
            output=content,
            token_cost=token_cost,
            success=success,
            time_cost=round(time_cost, 3),
        )

        self.set_output(tool_call_result.model_dump_json())


async def async_main():
    """Main function for testing the UseMockSearchOp with various queries."""
    from reme_ai.main import ReMeApp

    async with ReMeApp():
        test_queries = [
            "What is the capital of France?",
            "How does quantum computing work?",
            "Analyze the impact of artificial intelligence on global economy, employment, and society",
            "When was Python programming language created?",
            "Compare different types of renewable energy sources",
        ]

        for query in test_queries:
            op = UseMockSearchOp()
            await op.async_call(query=query)
            print(op.output)


if __name__ == "__main__":
    asyncio.run(async_main())
