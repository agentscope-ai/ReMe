"""Simple ReAct operation module.

This module provides a simple ReAct (Reasoning and Acting) agent implementation
that extends the base ReactSearchOp for answering user queries through iterative
reasoning and search actions.
"""

import asyncio

from flowllm.core.context import C, FlowContext
from flowllm.gallery.agent import ReactSearchOp


@C.register_op()
class SimpleReactOp(ReactSearchOp):
    """A simple ReAct (Reasoning and Acting) agent operation.

    This operation extends ReactSearchOp to provide a straightforward implementation
    of a ReAct agent that answers user queries by reasoning about the problem and
    taking search actions iteratively until a final answer is reached.

    The agent inherits all functionality from ReactSearchOp, including:
    - Iterative reasoning and action cycles
    - Search tool integration
    - Maximum step limits for preventing infinite loops
    """


async def main():
    """Main function to demonstrate SimpleReactOp usage.

    This function initializes the FlowLLM context with ReMe configuration,
    creates a SimpleReactOp instance, and processes a sample query about
    stock prices for Maotai and Wuliangye.

    Example:
        Run this module directly to test the SimpleReactOp:
        ```bash
        python -m reme_ai.agent.react.simple_react_op
        ```
    """
    from reme_ai.config.config_parser import ConfigParser

    C.set_service_config(parser=ConfigParser, config_name="config=default").init_by_service_config()
    context = FlowContext(query="茅台和五粮现在股价多少？")

    op = SimpleReactOp()
    await op.async_call(context=context)
    print(context.response.answer)


if __name__ == "__main__":
    asyncio.run(main())
