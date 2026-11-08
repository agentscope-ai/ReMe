"""LongMemEval agentic-answer step."""

import os

from reme.schema import collect_reply
from reme.steps.base_step import BaseStep
from reme.steps.index._dedup import _ToolContextDedupMixin
from reme.utils.counter import global_counter_inc


class LmeAgenticAnswerStep(BaseStep):
    """Answer a LongMemEval query with the ReAct agent and workspace tools."""

    # Reasoning-round budget. The AgentScope wrapper converts it to the
    # backend's iteration-counting semantics; other backends ignore it.
    MAX_ITERATION = 10
    TOOL_CONTEXT_PREFIX: str = "lme_agentic_answer"
    JOB_TOOLS: list[str] = ["search", "add_draft", "read_all_draft", "read"]
    INJECTED_JOB_KWARGS: dict = {"read_step_format_session": True}

    def _injected_job_kwargs(self, query: str) -> dict:
        """Enable query-aware session compression when requested."""
        injected = dict(self.INJECTED_JOB_KWARGS)
        if self.context is not None and self.context.get("compress_session"):
            injected["_search"] = {
                "_compress": {"session": "true"},
                "queries": [query],
                "type": "query-aware",
            }
        return injected

    async def execute(self):
        """Answer the current query and store the result in the response."""
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

        try:
            if self.context.stream:
                result = await collect_reply(
                    self.agent_wrapper.reply_stream(query, **wrapper_kwargs),
                    on_chunk=self.context.add_stream_chunk,
                )
            else:
                result = await self.agent_wrapper.reply(query, **wrapper_kwargs)
        finally:
            if self.app_context is not None:
                self.app_context.metadata.get(_ToolContextDedupMixin.TOOL_CONTEXTS_KEY, {}).pop(tool_context_id, None)
        text = result["result"].strip()
        self.context.response.metadata.update(result)

        self.logger.debug(f"[{self.name}] response: {text!r}")

        self.context.response.success = result["status"] == "completed" and self.context.stream_error is None
        self.context.response.answer = result["error"] or text
        self.context.response.metadata.update(
            {
                "query": query,
                "query_time": query_time,
                "sys_prompt": sys_prompt,
                "response": text,
            },
        )
        return self.context.response
