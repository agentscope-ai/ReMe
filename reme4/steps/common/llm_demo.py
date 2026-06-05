"""Demo step that drives an Agent via the agent_wrapper component."""

from typing import Type

from agentscope.tool import FunctionTool, Toolkit
from pydantic import BaseModel

from ..base_step import BaseStep
from ...components import R


def add(a: float, b: float) -> str:
    """Add two numbers and return the sum.

    Args:
        a: first addend
        b: second addend
    """
    return str(a + b)


@R.register("llm_demo_step")
class LLMDemoStep(BaseStep):
    """Drive an Agent powered by the ``agent_wrapper`` component.

    Inputs (from RuntimeContext):
        query     (str, required): user message content.
        sys_prompt (str, optional): system prompt for the agent.
        use_add_tool (bool, optional): register the ``add`` tool when True.

    Output (written to context.response.answer):
        The agent's final reply text.
    """

    DEFAULT_SYS_PROMPT = "You are a helpful assistant. Provide clear and detailed responses."

    async def execute(self):
        assert self.context is not None
        query: str = self.context.get("query", "")
        sys_prompt: str = self.context.get("sys_prompt") or self.DEFAULT_SYS_PROMPT
        use_add_tool: bool = bool(self.context.get("use_add_tool", False))
        structured_model: Type[BaseModel] | None = self.context.get("structured_model")

        if not query:
            self.context.response.success = False
            self.context.response.answer = "Skipped: empty query"
            return self.context.response

        toolkit = Toolkit(tools=[FunctionTool(add)]) if use_add_tool else Toolkit()

        wrapper_kwargs = {
            "system_prompt": sys_prompt,
            "toolkit": toolkit,
        }
        if structured_model is not None:
            wrapper_kwargs["output_schema"] = structured_model

        _, result = await self.agent_wrapper.reply(query, **wrapper_kwargs)

        structured_content: dict | None = None
        if isinstance(result, dict) and "message" in result:
            msg = result["message"]
            structured_content = result["structured_output"]
        else:
            msg = result

        text = (msg.get_text_content() or "").strip()
        self.logger.info(f"[{self.name}] response: {text!r}")

        self.context.response.success = True
        self.context.response.answer = text
        self.context.response.metadata.update(
            {
                "query": query,
                "sys_prompt": sys_prompt,
                "use_add_tool": use_add_tool,
                "response": text,
                "structured_output": structured_content,
            },
        )
        return self.context.response
