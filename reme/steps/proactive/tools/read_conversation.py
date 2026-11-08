"""Consume recent Main chat rounds within a server-bound proactive round."""

from ...base_step import BaseStep, Ref
from ...file_io._path import _check_path_permission
from ....components import PaRuntime, R


@R.register("read_conversation_step")
class ReadConversationStep(BaseStep):
    """Checkpoint completed chat text before advancing each session's cursor."""

    pa: PaRuntime = Ref(PaRuntime, "pa")

    async def execute(self):
        round_id = self.context.get("pa_round_id")
        current = self.pa.current_round
        if self.context.get("pa_caller") != "proactive" or current is None or current.id != round_id:
            raise RuntimeError("read_conversation requires a live proactive round")
        for path in (self.pa.session_ids_path, self.pa.proactive_context_path):
            if not _check_path_permission(self.workspace_path, path, self.context.get("_allowed_paths")):
                raise PermissionError("no permission to consume conversation")
        self.context.response.answer = await self.pa.read_conversation(
            max_turns=self.context.get("max_turns", 10),
            round_id=round_id,
        )
        return self.context.response
