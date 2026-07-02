"""Benchmark query step – ReAct agent that answers questions using the search tool."""

from ..base_step import BaseStep
from ...components import R
from ...enumeration import ChunkEnum


@R.register("bench_query_step")
class BenchQueryStep(BaseStep):
    """Answer a benchmark query via ReAct agent with access to the search tool.

    The agent uses the ``agent_wrapper`` component in ReAct mode, calling the
    ``search`` job tool to retrieve relevant memory chunks before generating
    a final answer.

    Inputs (from RuntimeContext):
        query       (str, required): The question to answer.
        query_time  (str, optional): ISO timestamp representing the query time,
                    used to ground the agent's temporal context.

    Output (written to context.response.answer):
        The agent's final answer text.
    """

    DEFAULT_SYS_PROMPT = (
        "You are a memory retrieval assistant. You MUST use the search tool to find information before answering.\n\n"
        "## Search Strategy\n"
        "- You can search multiple times with different queries to gather comprehensive information.\n"
        "## Answer Rules\n"
        "- Answer based ONLY on retrieved context.\n"
        "- Output ONLY the direct factual answer — no reasoning, no search process, no elaboration.\n"
        "- If information is insufficient after multiple searches, reply: 'Information not found.'"
    )

    TEMPORAL_HINT = "\nCurrent time context: {query_time}\n"

    async def execute(self):
        assert self.context is not None
        query: str = self.context.get("query", "")
        query_time: str | None = self.context.get("query_time")

        if not query:
            self.context.response.success = False
            self.context.response.answer = "Skipped: empty query"
            return self.context.response

        # Build system prompt with optional temporal context
        sys_prompt = self.DEFAULT_SYS_PROMPT
        if query_time:
            sys_prompt += self.TEMPORAL_HINT.format(query_time=query_time)

        wrapper_kwargs = {
            "system_prompt": sys_prompt,
            "job_tools": ["search"],
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
                "query_time": query_time,
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
