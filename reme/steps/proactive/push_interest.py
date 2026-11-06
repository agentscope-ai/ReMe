"""Bind Remy's reminder tools to the current foreground conversation."""

import asyncio
from uuid import uuid4

from ._interest import _InterestStep
from .loop import _minutes
from ...components import R


@R.register("push_interest_step")
class PushInterestStep(_InterestStep):
    """Let the agent choose its reminder; only its bound Job can deliver it."""

    async def execute(self):
        self._validate("default_wakeup_minutes")
        self.context.response.answer = {"sent": False}
        await self._ensure_memory()
        notes = await self._notes()
        session_id = self._state["session_id"]
        if session_id is None or not any(post["status"] == "unsent" for post in notes.values()):
            return self.context.response
        session_path = await self.agent_wrapper.get_session_path(session_id)
        snapshot = await self.agent_wrapper.read_session(session_id)
        round_id = uuid4().hex
        rounds = self._state.setdefault("rounds", {})
        current = rounds[round_id] = {
            "session_id": session_id,
            "turn_id": snapshot.last_turn_id or "",
            "versions": {name: self._version(post) for name, post in notes.items()},
            "lock": asyncio.Lock(),
            "request": None,
            "result": {"sent": False},
        }
        try:
            delay = self.context.get("default_wakeup_minutes", self.kwargs.get("default_wakeup_minutes"))
            await self._agent(
                ["session_status", "send_reminder", "set_next_wakeup"],
                injected={"session_id": session_id, "pa_round_id": round_id},
                session_id=session_id,
                session_path=session_path or "",
                default_wakeup_minutes=_minutes(delay),
            )
            self.context.response.answer = current["result"]
        finally:
            rounds.pop(round_id, None)
        return self.context.response
