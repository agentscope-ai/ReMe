"""Fork the chat session with proactive instructions, compact, then run at most one query."""

import asyncio

from .base import BasePaAgentStep
from ....components import R
from ....utils.async_utils import finish_cleanup


@R.register("proactive_agent_step")
class ProactiveAgentStep(BasePaAgentStep):
    """Carry chat context and unfinished work through one proactive turn."""

    async def execute(self):
        self._validate()
        max_turns = self.kwargs.get("max_turns", 10)
        if type(max_turns) is not int or not 1 <= max_turns <= 100:
            raise ValueError("max_turns must be an integer from 1 to 100")
        async with self.pa.proactive_round() as current:
            reason = await self.pa.proactive_skip_reason()
            if reason or current.session_id is None:
                self.context.response.answer = {"sent": False, "reason": reason or "no_session"}
                return self.context.response
            result = await self._within_active_hours(
                lambda: self._run_round(current, max_turns),
            )
            if result is None:
                self.context.response.answer = {**current.result, "reason": "outside_active_hours"}
            return self.context.response

    async def _run_round(self, current, max_turns):
        runtime = self.pa
        agent_session_id = None
        try:
            if self.context.stop_event is not None and self.context.stop_event.is_set():
                raise asyncio.CancelledError()
            await self._ensure_memory()
            source = await self.agent_wrapper.read_session(current.session_id)
            handoff = await runtime.read_proactive_handoff()
            options = self._agent_options(
                injected={
                    "session_id": current.session_id,
                    "pa_round_id": current.id,
                    "pa_caller": "proactive",
                },
                handoff=handoff,
                has_handoff=bool(handoff),
            )
            fork = {"resume": current.session_id, "fork_session": True} if source.last_turn_id is not None else {}
            agent_session_id = await self.agent_wrapper.create_session(**fork, **options)
            if source.last_turn_id is not None:
                await self.agent_wrapper.compact_session(agent_session_id)
            config = await runtime.get_proactive_config()
            try:
                hint = await runtime.hints.select(config, self._now())
            except (OSError, ValueError) as exc:
                self.logger.warning(f"Could not sample optional proactive hint: {exc}")
                hint = None
            optional_hint = ""
            if hint is not None:
                self.context.response.metadata["proactive_hint"] = hint
                hint_prompt, hint_index = hint.rsplit("_", 1)
                optional_hint = self.get_prompt(hint_prompt).strip().splitlines()[int(hint_index) - 1]
            if reason := await runtime.proactive_skip_reason():
                self.context.response.answer = {**current.result, "reason": reason}
                return self.context.response
            if self.context.stop_event is not None and self.context.stop_event.is_set():
                raise asyncio.CancelledError()
            query = self.prompt_format(
                "query",
                max_turns=max_turns,
                memory_path=self._memory_path(),
                proactive_config_path=runtime.proactive_config_path,
                optional_hint=optional_hint,
                has_hint=bool(optional_hint),
                hint_period=current.hint_period or "normal",
            )
            current.allow_message = True
            try:
                reply = await self.agent_wrapper.reply(
                    query,
                    resume=agent_session_id,
                    **options,
                    activity_source="proactive",
                    activity_session_id=current.session_id,
                )
            finally:
                current.allow_message = False
            if reply["status"] != "completed":
                raise RuntimeError(reply.get("error") or f"Proactive agent {reply['status']}")
            if current.request is None:
                raise RuntimeError(
                    "The proactive query must submit a send_chat_message decision, including empty content",
                )
            await runtime.get_proactive_config()
            await runtime.save_proactive_handoff(reply["result"])
            self.context.response.answer = dict(current.result)
            self.context.response.metadata["proactive_session_id"] = agent_session_id
            return self.context.response
        finally:
            if agent_session_id is not None:
                await finish_cleanup(self.agent_wrapper.release_session(agent_session_id))
