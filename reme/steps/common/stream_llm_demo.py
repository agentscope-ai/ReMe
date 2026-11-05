"""Demo step that drives an Agent via the agent_wrapper component with streaming output."""

from ..base_step import BaseStep
from ...components import R
from ...schema import collect_reply


@R.register("stream_llm_demo_step")
class StreamLLMDemoStep(BaseStep):
    """Drive an Agent powered by the ``agent_wrapper`` component with streaming output.

    When streaming is enabled on the context, text/thinking/tool events are
    pushed chunk-by-chunk via ``self.context.add_stream_chunk``.
    When streaming is not enabled, falls back to non-streaming reply.

    Inputs (from RuntimeContext):
        query      (str, required): user message content.
        sys_prompt (str, optional): system prompt for the agent.

    Output (written to context.response.answer):
        The agent's final reply text.
    """

    DEFAULT_SYS_PROMPT = "You are a helpful assistant. Provide clear and detailed responses."

    async def execute(self):
        assert self.context is not None
        query: str = self.context.get("query", "")
        sys_prompt: str = self.context.get("sys_prompt") or self.DEFAULT_SYS_PROMPT

        if not query:
            self.context.response.success = False
            self.context.response.answer = "Skipped: empty query"
            return self.context.response

        wrapper_kwargs = {
            "system_prompt": sys_prompt,
            "job_tools": ["add"],
        }

        if self.context.stream:
            result = await collect_reply(
                self.agent_wrapper.reply_stream(query, **wrapper_kwargs),
                on_chunk=self.context.add_stream_chunk,
            )
        else:
            result = await self.agent_wrapper.reply(query, **wrapper_kwargs)
        text = result["result"].strip()
        self.context.response.metadata.update(result)

        self.logger.debug(f"[{self.name}] response: {text!r}")

        self.context.response.success = result["status"] == "completed" and self.context.stream_error is None
        self.context.response.answer = result["error"] or text
        self.context.response.metadata.update(
            {
                "query": query,
                "sys_prompt": sys_prompt,
                "response": text,
            },
        )
        return self.context.response
