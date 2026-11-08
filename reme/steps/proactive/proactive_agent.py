"""Run six sequential queries in one fresh proactive conversation per wakeup."""

import asyncio

from .base_pa_step import BasePaStep
from ...components import R
from ...utils.async_utils import finish_cleanup


@R.register("proactive_agent_step")
class ProactiveAgentStep(BasePaStep):
    """Keep preferences, task analysis, reminder decisions and handoff in one context."""

    async def execute(self):
        self._validate()
        runtime = self.pa
        max_turns = self.kwargs.get("max_turns", 20)
        if type(max_turns) is not int or not 1 <= max_turns <= 100:
            raise ValueError("max_turns must be an integer from 1 to 100")
        async with runtime.proactive_round() as current:
            return await self._run_round(runtime, max_turns, current)

    async def _run_round(self, runtime, max_turns, current):
        config = await runtime.get_proactive_config()
        if not config.is_active(self._now()):
            self.context.response.answer = {"sent": False, "reason": "outside_active_hours"}
            return self.context.response
        await self._ensure_memory()
        session_id, round_id = current.session_id, current.id
        agent_session_id = None
        query_name = "query_memory"
        try:
            system_prompt = self._system_prompt(
                now=self._now().isoformat(),
                handoff=await runtime.read_handoff(),
                recovery_path=runtime.proactive_context_path,
            )
            for query_name in (
                "query_memory",
                "query_conversation",
                "query_tasks",
                "query_reminder",
                "query_config",
                "query_handoff",
            ):
                if self.context.stop_event is not None and self.context.stop_event.is_set():
                    raise asyncio.CancelledError()
                config = await runtime.get_proactive_config()
                now = self._now()
                query = self.prompt_format(
                    query_name,
                    memory_path=self._memory_path(),
                    max_turns=max_turns,
                    proactive_config_path=runtime.proactive_config_path,
                    interval_minutes=config.interval_minutes,
                    next_wakeup=config.next_wakeup(now).isoformat(),
                    now=now.isoformat(),
                )
                if query_name == "query_reminder":
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
                        )
                        self.context.response.metadata["proactive_hint"] = hint
                reply = await self.agent_wrapper.reply(
                    query,
                    resume=agent_session_id,
                    system_prompt=system_prompt,
                    job_tools=list(self.kwargs.get("job_tools") or []),
                    injected_job_kwargs={"session_id": session_id, "pa_round_id": round_id},
                    activity_source="proactive",
                    activity_session_id=session_id,
                )
                agent_session_id = reply["session_id"]
                if reply["status"] != "completed":
                    raise RuntimeError(reply.get("error") or f"Proactive agent {reply['status']}")
                await runtime.checkpoint_proactive(round_id, query_name, reply["result"])
                if query_name == "query_reminder" and current.request is None:
                    raise RuntimeError(
                        "The reminder query must submit a send_reminder decision, including empty content",
                    )
            await runtime.write_handoff(reply["result"])
            await runtime.clear_proactive_context()
            self.context.response.answer = {**current.result, "handoff": reply["result"]}
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
