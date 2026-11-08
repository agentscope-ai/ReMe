"""Persistent feedback between the chat and proactive agents."""

from ...base_step import BaseStep, Ref
from ...file_io._path import _check_path_permission
from ....components import PaRuntime, R


@R.register("write_proactive_inbox_step")
class WriteProactiveInboxStep(BaseStep):
    """Append feedback without starting a proactive round."""

    pa: PaRuntime = Ref(PaRuntime, "pa")

    async def execute(self):
        if self.context.get("pa_caller") != "chat":
            raise ValueError("write_proactive_inbox is bound to Main")
        for path in (self.pa.proactive_inbox_path, self.pa.inbox.receipts_path):
            if not _check_path_permission(self.workspace_path, path, self.context.get("_allowed_paths")):
                raise PermissionError("no permission to write proactive inbox")
        self.context.response.answer = await self.pa.write_proactive_inbox(self.context.get("message"))
        return self.context.response


@R.register("read_proactive_inbox_step")
class ReadProactiveInboxStep(BaseStep):
    """Checkpoint feedback before consuming the text inbox."""

    pa: PaRuntime = Ref(PaRuntime, "pa")

    async def execute(self):
        current = self.pa.current_round
        if current is None or current.id != self.context.get("pa_round_id"):
            raise ValueError("read_proactive_inbox requires a live proactive round")
        paths = [self.pa.proactive_inbox_path, self.pa.inbox.cursor_path, self.pa.inbox.receipts_path]
        if self.context.get("pa_round_id") is not None:
            paths.append(self.pa.proactive_context_path)
        if any(
            not _check_path_permission(self.workspace_path, path, self.context.get("_allowed_paths")) for path in paths
        ):
            raise PermissionError("no permission to consume proactive feedback and save recovery context")
        self.context.response.answer = await self.pa.read_proactive_inbox(
            max_bytes=self.max_file_bytes(),
            round_id=self.context.get("pa_round_id"),
        )
        return self.context.response
