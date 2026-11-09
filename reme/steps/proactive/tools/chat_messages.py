"""Thin, permission-aware reminder tools; all submission rules belong to PaRuntime."""

from ...base_step import BaseStep, Ref
from ...file_io._path import _check_path_permission
from ....components import PaRuntime, R


@R.register("send_chat_message_step")
class SendChatMessageStep(BaseStep):
    """Submit a decision bound to the active proactive round."""

    pa: PaRuntime = Ref(PaRuntime, "pa")

    async def execute(self):
        if not _check_path_permission(self.workspace_path, self.pa.chat_inbox_path, self.context.get("_allowed_paths")):
            raise PermissionError("no permission to write reminder history")
        if self.context.get("pa_caller") != "proactive":
            raise ValueError("send_chat_message requires a server-bound sending agent")
        self.context.response.answer = await self.pa.send_chat_message(
            self.context.get("content"),
            self.context.get("kind", "normal"),
            round_id=self.context.get("pa_round_id"),
            session_id=self.context.get("session_id"),
        )
        return self.context.response


@R.register("send_chat_message_immediate_step")
class SendChatMessageImmediateStep(BaseStep):
    """Submit an immediate result bound to a cron decision or temporary task."""

    pa: PaRuntime = Ref(PaRuntime, "pa")

    async def execute(self):
        caller = self.context.get("pa_caller")
        if caller == "cron":
            run_id = self.context.get("cron_run_id")
            current = self.pa.tasks.message_context(run_id)
            for path in (
                self.pa.chat_inbox_path,
                self.pa.tasks.recovery_path(current.saved.task.id),
                self.workspace_path / current.saved.run.result_path,
            ):
                if not _check_path_permission(self.workspace_path, path, self.context.get("_allowed_paths")):
                    raise PermissionError("no permission to publish cron messages")
            self.context.response.answer = await self.pa.tasks.send(run_id, self.context.get("content"))
            return self.context.response
        if caller != "temporary":
            raise ValueError("Immediate chat messages require a server-bound task")
        task_id = self.context.get("temp_task_id")
        for path in (
            self.pa.chat_inbox_path,
            self.pa.temp_tasks.path,
            self.pa.temp_tasks.document_path(task_id, "result.md"),
        ):
            if not _check_path_permission(self.workspace_path, path, self.context.get("_allowed_paths")):
                raise PermissionError("no permission to publish temporary task results")
        self.context.response.answer = await self.pa.temp_tasks.send(
            task_id,
            self.context.get("temp_revision"),
            self.context.get("content"),
        )
        return self.context.response


@R.register("get_chat_messages_step")
class GetChatMessagesStep(BaseStep):
    """Read current reminder states without consuming history."""

    pa: PaRuntime = Ref(PaRuntime, "pa")

    async def execute(self):
        if not _check_path_permission(self.workspace_path, self.pa.chat_inbox_path, self.context.get("_allowed_paths")):
            raise PermissionError("no permission to read reminder history")
        self.context.response.answer = await self.pa.get_chat_messages(
            limit=self.context.get("limit", 10),
            offset=self.context.get("offset", 0),
            round_id=self.context.get("pa_round_id") if self.context.get("pa_caller") == "proactive" else None,
            cron_run_id=self.context.get("cron_run_id") if self.context.get("pa_caller") == "cron" else None,
        )
        return self.context.response
