"""Demo step that drives an Agent via BaseStep.llm."""

from agentscope.agent import Agent
from agentscope.message import Msg, TextBlock
from agentscope.tool import FunctionTool, Toolkit

from ..base_step import BaseStep
from ...components import R


def _add(a: float, b: float) -> str:
    """Add two numbers and return the sum.

    Args:
        a: first addend
        b: second addend
    """
    return str(a + b)


@R.register("llm_demo_step")
class LLMDemoStep(BaseStep):
    """Drive an Agent powered by ``self.llm``.

    Inputs (from RuntimeContext):
        query     (str, required): user message content.
        sys_prompt (str, optional): system prompt for the agent.
        use_add_tool (bool, optional): register the ``add`` tool when True.

    Output (written to context.response.answer):
        The agent's final reply text.
    """

    DEFAULT_SYS_PROMPT = "You are a concise assistant. Reply in one short sentence."

    async def execute(self):
        assert self.context is not None
        query: str = self.context.get("query", "")
        sys_prompt: str = self.context.get("sys_prompt") or self.DEFAULT_SYS_PROMPT
        use_add_tool: bool = bool(self.context.get("use_add_tool", False))

        if not query:
            self.context.response.success = False
            self.context.response.answer = "Skipped: empty query"
            return self.context.response

        toolkit = Toolkit(tools=[FunctionTool(_add)]) if use_add_tool else Toolkit()

        agent = Agent(
            name=self.name,
            system_prompt=sys_prompt,
            model=self.llm,
            toolkit=toolkit,
        )

        response: Msg = await agent.reply(
            Msg(name="user", role="user", content=[TextBlock(text=query)]),
        )
        text = (response.get_text_content() or "").strip()
        self.logger.info(f"[{self.name}] response: {text!r}")

        self.context.response.success = True
        self.context.response.answer = text
        self.context.response.metadata.update(
            {
                "query": query,
                "sys_prompt": sys_prompt,
                "use_add_tool": use_add_tool,
                "response": text,
            },
        )
        return self.context.response
