"""LongMemEval agentic answer step – ReAct agent that answers questions using the search tool."""

import os
import threading

from ...base_step import BaseStep
from ....components import R
from ....enumeration import ChunkEnum

# ---------------------------------------------------------------------------
# Process-safe & thread-safe counter for unique tool_context_id.
# PID guarantees cross-process uniqueness (multiprocessing Pool);
# threading.Lock + monotonic counter guarantees thread safety within a process.
# ---------------------------------------------------------------------------
_TOOL_CTX_LOCK = threading.Lock()
_TOOL_CTX_SEQ = 0


def _next_tool_context_id() -> str:
    global _TOOL_CTX_SEQ
    with _TOOL_CTX_LOCK:
        _TOOL_CTX_SEQ += 1
        seq = _TOOL_CTX_SEQ
    return f"lme_agentic_answer_{os.getpid()}_{seq}"


@R.register("lme_agentic_answer_step")
class LmeAgenticAnswerStep(BaseStep):
    """Answer a LongMemEval query via ReAct agent with access to the search tool.

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

    MAX_ITERATION = 10

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

        wrapper_kwargs = {
            "system_prompt": sys_prompt,
            "job_tools": ["search", "add_draft", "read_all_draft"],
            "react_config": {"max_iters": self.MAX_ITERATION},
            "tool_context_id": _next_tool_context_id(),
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
