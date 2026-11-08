"""Read incremental conversation text through the PA component."""

from ...base_step import BaseStep, Ref
from ...file_io._path import _check_path_permission
from ....components import PaRuntime, R


@R.register("read_conversation_step")
class ReadConversationStep(BaseStep):
    """Return timestamped unread user/assistant text across recent chat sessions."""

    pa: PaRuntime = Ref(PaRuntime, "pa")

    async def execute(self):
        if not _check_path_permission(
            self.workspace_path,
            self.pa.session_ids_path,
            self.context.get("_allowed_paths"),
        ):
            raise PermissionError("no permission to access conversation cursors")
        if self.context.get("pa_round_id") is not None and not _check_path_permission(
            self.workspace_path,
            self.pa.proactive_context_path,
            self.context.get("_allowed_paths"),
        ):
            raise PermissionError("no permission to save conversation recovery context")
        self.context.response.answer = await self.pa.read_conversation(
            max_turns=self.context.get("max_turns", 20),
            max_message_chars=self.context.get("max_message_chars", 2000),
            max_chars=self.context.get("max_chars", 20000),
            round_id=self.context.get("pa_round_id"),
        )
        return self.context.response
