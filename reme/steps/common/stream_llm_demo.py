"""Demo step that drives an Agent via the agent_wrapper component with streaming output."""

from ..base_step import BaseStep
from ...components import R
from ...enumeration import ChunkEnum


@R.register("stream_llm_demo_step")
class StreamLLMDemoStep(BaseStep):
    """Drive an Agent powered by the ``agent_wrapper`` component with streaming output.

    When streaming is enabled on the context, text/thinking/tool events are
    pushed chunk-by-chunk via ``self.context.add_stream_string``.
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
            text = await self._stream_reply(query, **wrapper_kwargs)
        else:
            result = await self.agent_wrapper.reply(query, **wrapper_kwargs)
            text = (result.get("result") or "").strip()

        self.logger.debug(f"[{self.name}] response: {text!r}")

        self.context.response.success = True
        self.context.response.answer = text
        self.context.response.metadata.update(
            {
                "query": query,
                "sys_prompt": sys_prompt,
                "response": text,
            },
        )
        return self.context.response

    async def _stream_reply(self, query: str, **wrapper_kwargs) -> str:
        """Stream unified chunks to the context stream queue."""
        assert self.context is not None
        text_parts: list[str] = []

        async for chunk in self.agent_wrapper.reply_stream(query, **wrapper_kwargs):
            await self.context.add_stream_string(chunk.chunk, chunk.chunk_type)

            if chunk.chunk_type == ChunkEnum.CONTENT and isinstance(chunk.chunk, str):
                text_parts.append(chunk.chunk)

            if chunk.session_id:
                self.context.response.metadata["session_id"] = chunk.session_id

        return "".join(text_parts).strip()
