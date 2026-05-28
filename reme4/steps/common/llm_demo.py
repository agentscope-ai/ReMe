"""Demo step that drives a ReActAgent via BaseStep.as_llm/as_llm_formatter."""

from agentscope.agent import ReActAgent
from agentscope.message import Msg, TextBlock
from agentscope.tool import Toolkit, ToolResponse

from ..base_step import BaseStep
from ...components import R


def _add(a: float, b: float) -> ToolResponse:
    """Add two numbers and return the sum.

    Args:
        a: first addend
        b: second addend
    """
    return ToolResponse(content=[TextBlock(type="text", text=str(a + b))])


@R.register("llm_demo_step")
class LLMDemoStep(BaseStep):
    """Drive a ReActAgent powered by ``self.as_llm`` / ``self.as_llm_formatter``.

    Inputs (from RuntimeContext):
        query     (str, required): user message content.
        sys_prompt (str, optional): system prompt for the agent.
        use_add_tool (bool, optional): register the ``add`` tool when True.
        console_enabled (bool, optional): mirror agent output to stdout.

    Output (written to context.response.answer):fa
        The agent's final reply text.
    """

    DEFAULT_SYS_PROMPT = "You are a concise assistant. Reply in one short sentence."

    async def execute(self):
        assert self.context is not None
        query: str = self.context.get("query", "")
        sys_prompt: str = self.context.get("sys_prompt") or self.DEFAULT_SYS_PROMPT
        use_add_tool: bool = bool(self.context.get("use_add_tool", False))
        console_enabled: bool = bool(self.context.get("console_enabled", False))

        if not query:
            self.context.response.success = False
            self.context.response.answer = "Skipped: empty query"
            return self.context.response

        toolkit = Toolkit()
        if use_add_tool:
            toolkit.register_tool_function(_add)

        agent = ReActAgent(
            name=self.name,
            sys_prompt=sys_prompt,
            model=self.as_llm,
            formatter=self.as_llm_formatter,
            toolkit=toolkit,
        )
        agent.set_console_output_enabled(console_enabled)

        response: Msg = await agent.reply(
            Msg(name="user", role="user", content=query),
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
