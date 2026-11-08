"""Thin, permission-aware reminder tools; all submission rules belong to PaRuntime."""

from ...base_step import BaseStep, Ref
from ...file_io._path import _check_path_permission
from ....components import PaRuntime, R


@R.register("send_reminder_step")
class SendReminderStep(BaseStep):
    """Submit a decision bound to the active proactive round."""

    pa: PaRuntime = Ref(PaRuntime, "pa")

    async def execute(self):
        if not _check_path_permission(self.workspace_path, self.pa.reminders_path, self.context.get("_allowed_paths")):
            raise PermissionError("no permission to write reminder history")
        self.context.response.answer = await self.pa.send_reminder(
            self.context.get("content"),
            self.context.get("kind"),
            round_id=self.context.get("pa_round_id"),
            session_id=self.context.get("session_id"),
        )
        return self.context.response


@R.register("get_reminder_step")
class GetReminderStep(BaseStep):
    """Read current reminder states without consuming history."""

    pa: PaRuntime = Ref(PaRuntime, "pa")

    async def execute(self):
        if not _check_path_permission(self.workspace_path, self.pa.reminders_path, self.context.get("_allowed_paths")):
            raise PermissionError("no permission to read reminder history")
        self.context.response.answer = await self.pa.get_reminder(
            limit=self.context.get("limit", 10),
            offset=self.context.get("offset", 0),
        )
        return self.context.response
