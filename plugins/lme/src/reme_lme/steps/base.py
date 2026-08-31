"""Plugin-local base class for benchmark agentic-answer steps."""

import os

from reme.steps.base_step import BaseStep
from reme.steps.index._dedup import _ToolContextDedupMixin
from reme.enumeration import ChunkEnum
from reme.utils.counter import global_counter_inc


class BaseAgenticAnswerStep(BaseStep):
    """ReAct-agent answer implementation owned by this benchmark plugin.

    Subclasses only need to set:
        TOOL_CONTEXT_PREFIX (str): prefix used to build the unique tool_context_id.
        JOB_TOOLS (list[str]): tools exposed to the ReAct agent; override to customize.
        INJECTED_JOB_KWARGS (dict): server-owned kwargs injected into every job
            tool call via ``injected_job_kwargs``; override the attribute or the
            ``_injected_job_kwargs`` hook to customize.

    Concrete subclasses are registered by the plugin manifest.

    Inputs (from RuntimeContext):
        query       (str, required): The question to answer.
        query_time  (str, optional): ISO timestamp representing the query time,
                    used to ground the agent's temporal context.

    Output (written to context.response.answer):
        The agent's final answer text.
    """

    # Reasoning-round budget. The AgentScope wrapper converts it to the
    # backend's iteration-counting semantics; other backends ignore it.
    MAX_ITERATION = 10
    TOOL_CONTEXT_PREFIX: str = "content_agentic_answer"
    JOB_TOOLS: list[str] = ["search", "add_draft", "read_all_draft", "read"]
    INJECTED_JOB_KWARGS: dict = {"read_step_format_session": True}

    def _injected_job_kwargs(self, query: str) -> dict:  # pylint: disable=unused-argument
        """Server-owned kwargs injected into every job tool call.

        Overridable hook: subclasses can extend the static
        ``INJECTED_JOB_KWARGS`` with per-request values derived from ``query``.

        When the runtime context carries a truthy ``compress_session`` flag,
        session-transcript compression is enabled in ``search_v2_step`` by
        injecting a ``_search._compress.session`` marker plus the current
        ``query`` as the query-aware relevance filter. Default (falsy) leaves
        session chunks uncompressed.
        """
        injected = dict(self.INJECTED_JOB_KWARGS)
        if self.context is not None and self.context.get("compress_session"):
            injected["_search"] = {
                "_compress": {"session": "true"},
                "queries": [query],
                "type": "query-aware",
            }
        return injected

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
                f"{global_counter_inc(self.app_context.metadata, [self.TOOL_CONTEXT_PREFIX])}"
            )
        else:
            tool_context_id = f"{self.TOOL_CONTEXT_PREFIX}_{os.getpid()}_local"
        wrapper_kwargs = {
            "system_prompt": sys_prompt,
            "job_tools": list(self.JOB_TOOLS),
            "react_config": {"max_iters": self.MAX_ITERATION},
            "tool_context_id": tool_context_id,
        }
        if injected_job_kwargs := self._injected_job_kwargs(f"{query}(query time: {query_time})"):
            wrapper_kwargs["injected_job_kwargs"] = injected_job_kwargs

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
