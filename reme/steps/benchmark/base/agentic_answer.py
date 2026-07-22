"""Shared base class for benchmark agentic-answer steps."""

import os

from ...base_step import BaseStep
from ...index._dedup import _ToolContextDedupMixin
from ....enumeration import ChunkEnum


class BaseAgenticAnswerStep(BaseStep):
    """Base ReAct-agent answer step shared by BEAM and LongMemEval benchmarks.

    Subclasses only need to set:
        TOOL_CONTEXT_PREFIX (str): prefix used to build the unique tool_context_id.

    And apply their own ``@R.register(...)`` decorator and docstring.

    Inputs (from RuntimeContext):
        query       (str, required): The question to answer.
        query_time  (str, optional): ISO timestamp representing the query time,
                    used to ground the agent's temporal context.

    Output (written to context.response.answer):
        The agent's final answer text.
    """

    MAX_ITERATION = 10
    TOOL_CONTEXT_PREFIX: str = "content_agentic_answer"

    async def execute(self):
        assert self.context is not None
        query: str = self.context.get("query", "")
        query_time: str | None = self.context.get("query_time")

        if not query:
            self.context.response.success = False
            self.context.response.answer = "Skipped: empty query"
            return self.context.response

        # Build system prompt with optional temporal context
        sys_prompt = self.get_prompt("system_prompt")
        if query_time:
            sys_prompt += "\n" + self.prompt_format("temporal_hint", query_time=query_time)

        if self.app_context is not None:
            tool_context_id = (
                f"{self.TOOL_CONTEXT_PREFIX}_{os.getpid()}_"
                f"{self.app_context.global_counter_next([self.TOOL_CONTEXT_PREFIX])}"
            )
        else:
            tool_context_id = f"{self.TOOL_CONTEXT_PREFIX}_{os.getpid()}_local"
        wrapper_kwargs = {
            "system_prompt": sys_prompt,
            "job_tools": ["search", "add_draft", "read_all_draft"],
            "react_config": {"max_iters": self.MAX_ITERATION},
            "tool_context_id": tool_context_id,
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

        if self.app_context is not None:
            self.app_context.metadata.get(_ToolContextDedupMixin.TOOL_CONTEXTS_KEY, {}).pop(tool_context_id, None)
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
