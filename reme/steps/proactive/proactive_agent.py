"""Prepare Main context once, then run seven queries within the proactive window."""

import asyncio
from contextlib import nullcontext

from .base_pa_step import BasePaStep
from ...components import R
from ...utils.async_utils import finish_cleanup


@R.register("proactive_agent_step")
class ProactiveAgentStep(BasePaStep):
    """Refresh Main context and carry unfinished work through a TODO list."""

    async def execute(self):
        self._validate()
        runtime = self.pa
        max_turns = self.kwargs.get("max_turns", 10)
        if type(max_turns) is not int or not 1 <= max_turns <= 100:
            raise ValueError("max_turns must be an integer from 1 to 100")
        async with runtime.proactive_round() as current:
            return await self._run_round(runtime, max_turns, current)

    async def _run_round(self, runtime, max_turns, current):
        pending = await asyncio.to_thread(runtime.inbox.pending)
        try:
            config = await runtime.get_proactive_config()
        except (OSError, ValueError):
            if not pending:
                raise
            config, reason = None, "invalid_config"
        else:
            reason = await runtime.proactive_skip_reason(config)
        settings_only = bool(reason and pending)
        if reason and not settings_only:
            self.context.response.answer = {"sent": False, "reason": reason}
            return self.context.response
        await self._ensure_memory()
        session_id, round_id = current.session_id, current.id
        agent_session_id = None
        query_name = "query_context"
        try:
            if session_id is None:
                self.context.response.answer = {"sent": False, "reason": "no_session"}
                return self.context.response
            source = await self.agent_wrapper.read_session(session_id)
            chat_context = self.get_prompt("no_main_context")
            if source.last_turn_id is not None:
                context_reply = await self._reply(
                    self._handoff_prompt("query_context", now=self._now().isoformat()),
                    resume=session_id,
                    fork_session=True,
                    activity_source="proactive",
                    activity_session_id=session_id,
                    immediate=settings_only,
                )
                if context_reply is None:
                    self.context.response.answer = {"sent": False, "reason": "outside_active_hours"}
                    return self.context.response
                if context_reply["status"] != "completed" or not context_reply["result"].strip():
                    raise RuntimeError(context_reply.get("error") or "Could not prepare proactive chat context")
                chat_context = context_reply["result"]
            await runtime.checkpoint_proactive(round_id, "chat_context", chat_context)
            system_prompt = self._system_prompt(
                now=self._now().isoformat(),
                recovery_path=runtime.proactive_context_path,
                chat_context=chat_context,
                todo_path=runtime.proactive_todo_path,
                todo=await runtime.read_proactive_todo(),
            )
            queries = (
                ("query_memory", "query_config", "query_todo")
                if settings_only
                else (
                    "query_memory",
                    "query_tasks",
                    "query_history",
                    "query_research",
                    "query_message",
                    "query_config",
                    "query_todo",
                )
            )
            for query_name in queries:
                if self.context.stop_event is not None and self.context.stop_event.is_set():
                    raise asyncio.CancelledError()
                if not settings_only:
                    config = await runtime.get_proactive_config()
                now = self._now()
                query = self.prompt_format(
                    query_name,
                    memory_path=self._memory_path(),
                    max_turns=max_turns,
                    proactive_config_path=runtime.proactive_config_path,
                    now=now.isoformat(),
                    interval_minutes=config.interval_minutes if config is not None else None,
                    todo_path=runtime.proactive_todo_path,
                )
                if query_name == "query_message":
                    try:
                        hint = await runtime.hints.select(config, now)
                    except (OSError, ValueError) as exc:
                        self.logger.warning(f"Could not sample optional proactive hint: {exc}")
                        hint = None
                    if hint is not None:
                        query += "\n\n" + self.prompt_format(
                            "optional_hint",
                            now=now.isoformat(),
                            hint=self.get_prompt(hint),
                            period=current.hint_period,
                        )
                        self.context.response.metadata["proactive_hint"] = hint
                reason = None if settings_only else await runtime.proactive_skip_reason()
                if reason:
                    self.context.response.answer = {**current.result, "reason": reason}
                    return self.context.response
                current.phase = query_name
                try:
                    lock = runtime.memory_lock if query_name == "query_memory" else nullcontext()
                    async with lock:
                        reply = await self._reply(
                            query,
                            resume=agent_session_id,
                            system_prompt=system_prompt,
                            job_tools=list(self.kwargs.get("job_tools") or []),
                            injected_job_kwargs={
                                "session_id": session_id,
                                "pa_round_id": round_id,
                                "pa_caller": "proactive",
                            },
                            activity_source="proactive",
                            activity_session_id=session_id,
                            immediate=settings_only,
                        )
                finally:
                    current.phase = None
                if reply is None:
                    self.context.response.answer = {**current.result, "reason": "outside_active_hours"}
                    return self.context.response
                agent_session_id = reply["session_id"]
                if reply["status"] != "completed":
                    raise RuntimeError(reply.get("error") or f"Proactive agent {reply['status']}")
                if query_name == "query_config":
                    config = await runtime.get_proactive_config()
                await runtime.checkpoint_proactive(round_id, query_name, reply["result"])
                if query_name == "query_message" and current.request is None:
                    raise RuntimeError(
                        "The message query must submit a send_chat_message decision, including empty content",
                    )
                if query_name == "query_todo":
                    await runtime.save_proactive_todo(reply["result"])
            await runtime.clear_proactive_context()
            self.context.response.answer = dict(current.result)
            self.context.response.metadata["proactive_session_id"] = agent_session_id
        except BaseException as exc:
            try:
                await finish_cleanup(
                    runtime.checkpoint_proactive(
                        round_id,
                        "failure",
                        {
                            "query": query_name,
                            "session_id": agent_session_id,
                            "error": str(exc) or type(exc).__name__,
                        },
                    ),
                )
            except Exception as checkpoint_error:
                self.logger.warning(f"Could not save proactive failure context: {checkpoint_error}")
            raise
        return self.context.response
