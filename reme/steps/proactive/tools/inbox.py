"""Persistent feedback between the chat and proactive agents."""

from ..base import BasePaStep
from ....components import R


@R.register("write_proactive_inbox_step")
class WriteProactiveInboxStep(BasePaStep):
    """Append feedback without starting a proactive round."""

    async def execute(self):
        if self.context.get("pa_caller") != "chat":
            raise ValueError("write_proactive_inbox is bound to Main")
        self.require_paths(self.pa.proactive_inbox_path, self.pa.inbox.receipts_path)
        self.context.response.answer = await self.pa.write_proactive_inbox(self.context.get("message"))
        return self.context.response


@R.register("read_proactive_inbox_step")
class ReadProactiveInboxStep(BasePaStep):
    """Return unread inbox text and advance its consumption state."""

    async def execute(self):
        current = self.pa.current_round
        if current is None or current.id != self.context.get("pa_round_id"):
            raise ValueError("read_proactive_inbox requires a live proactive round")
        self.require_paths(
            self.pa.proactive_inbox_path,
            self.pa.inbox.cursor_path,
            self.pa.inbox.receipts_path,
        )
        self.context.response.answer = await self.pa.read_proactive_inbox(
            max_bytes=self.max_file_bytes(),
            round_id=self.context.get("pa_round_id"),
        )
        return self.context.response
