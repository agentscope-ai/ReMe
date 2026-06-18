"""Demo step that drives an Agent via the agent_wrapper component."""

from typing import Type

from pydantic import BaseModel

from ..base_step import BaseStep
from ...components import R


@R.register("llm_demo_step")
class LLMDemoStep(BaseStep):
    """Drive an Agent powered by the ``agent_wrapper`` component.

    Inputs (from RuntimeContext):
        query     (str, required): user message content.
        sys_prompt (str, optional): system prompt for the agent.

    Output (written to context.response.answer):
        The agent's final reply text.
    """

    DEFAULT_SYS_PROMPT = "You are a helpful assistant. Provide clear and detailed responses."

    async def execute(self):
        assert self.context is not None
        query: str = self.context.get("query", "")
        sys_prompt: str = self.context.get("sys_prompt") or self.DEFAULT_SYS_PROMPT
        structured_model: Type[BaseModel] | None = self.context.get("structured_model")

        if not query:
            self.context.response.success = False
            self.context.response.answer = "Skipped: empty query"
            return self.context.response

        wrapper_kwargs = {
            "system_prompt": sys_prompt,
            "job_tools": ["add"],
        }
        if structured_model is not None:
            wrapper_kwargs["output_schema"] = structured_model

        result = await self.agent_wrapper.reply(query, **wrapper_kwargs)

        structured_content = result.get("structured_output")
        text = (result.get("result") or "").strip()
        self.logger.info(f"[{self.name}] response: {text!r}")

        self.context.response.success = True
        self.context.response.answer = text
        self.context.response.metadata.update(
            {
                "query": query,
                "sys_prompt": sys_prompt,
                "response": text,
                "structured_output": structured_content,
            },
        )
        return self.context.response
