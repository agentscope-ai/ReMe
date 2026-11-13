"""Consume recent chat rounds within a server-bound proactive round."""

from ..base import BasePaStep
from ....components import R


@R.register("read_conversation_step")
class ReadConversationStep(BasePaStep):
    """Return completed chat text and advance each session's cursor."""

    async def execute(self):
        round_id = self.context.get("pa_round_id")
        current = self.pa.current_round
        if self.context.get("pa_caller") != "proactive" or current is None or current.id != round_id:
            raise RuntimeError("read_conversation requires a live proactive round")
        self.require_paths(self.pa.session_ids_path)
        self.context.response.answer = await self.pa.sessions.read_conversation(
            max_turns=self.context.get("max_turns", 10),
        )
        return self.context.response
