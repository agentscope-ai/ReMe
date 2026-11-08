"""Persistent feedback between the chat and proactive agents."""

from ...base_step import BaseStep, Ref
from ...file_io._path import _check_path_permission
from ....components import PaRuntime, R


@R.register("send_to_proactive_step")
class SendToProactiveStep(BaseStep):
    """Append feedback without starting a proactive round."""

    pa: PaRuntime = Ref(PaRuntime, "pa")

    async def execute(self):
        target = self.pa.proactive_inbox_path
        if not _check_path_permission(self.workspace_path, target, self.context.get("_allowed_paths")):
            raise PermissionError("no permission to access the proactive inbox")
        await self.pa.send_to_proactive(self.context.get("message"))
        self.context.response.answer = {"queued": True, "path": self.to_workspace_relative(target)}
        return self.context.response


@R.register("get_proactive_messages_step")
class GetProactiveMessagesStep(BaseStep):
    """Checkpoint feedback before consuming the text inbox."""

    pa: PaRuntime = Ref(PaRuntime, "pa")

    async def execute(self):
        paths = [self.pa.proactive_inbox_path]
        if self.context.get("pa_round_id") is not None:
            paths.append(self.pa.proactive_context_path)
        if any(
            not _check_path_permission(self.workspace_path, path, self.context.get("_allowed_paths")) for path in paths
        ):
            raise PermissionError("no permission to consume proactive feedback and save recovery context")
        self.context.response.answer = await self.pa.get_proactive_messages(
            max_bytes=self.max_file_bytes(),
            round_id=self.context.get("pa_round_id"),
        )
        return self.context.response
