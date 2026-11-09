"""Thin, permission-aware reminder tools; all submission rules belong to PaRuntime."""

from ..base import BasePaStep
from ....components import R
from ....components.pa.reminders import message_content


@R.register("send_chat_message_step")
class SendChatMessageStep(BasePaStep):
    """Submit a decision bound to the active proactive round."""

    async def execute(self):
        self.require_paths(self.pa.chat_inbox_path)
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
class SendChatMessageImmediateStep(BasePaStep):
    """Send a temporary result or a finalized cron task's urgent finding."""

    async def execute(self):
        caller = self.context.get("pa_caller")
        if caller == "cron":
            task_id = self.context.get("cron_task_id")
            if not isinstance(task_id, str):
                raise ValueError("Cron messages require a server-bound task identity")
            state = self.pa.tasks.execution(task_id).state
            self.require_paths(
                self.pa.chat_inbox_path,
                self.pa.tasks.task_path(task_id),
                self.pa.tasks.staging_path(state.run),
            )
            content = message_content(self.context.get("content"))
            self.context.response.answer = await self.pa.tasks.complete(task_id, content)
            return self.context.response
        if caller != "temporary":
            raise ValueError("Immediate chat messages require a server-bound task")
        task_id = self.context.get("temp_task_id")
        self.require_paths(
            self.pa.chat_inbox_path,
            self.pa.temp_tasks.path,
            self.pa.temp_tasks.document_path(task_id),
        )
        self.context.response.answer = await self.pa.temp_tasks.send(
            task_id,
            self.context.get("temp_revision"),
            self.context.get("content"),
        )
        return self.context.response


@R.register("get_chat_messages_step")
class GetChatMessagesStep(BasePaStep):
    """Read current reminder states without consuming history."""

    async def execute(self):
        self.require_paths(self.pa.chat_inbox_path)
        if self.context.get("pa_caller") == "cron":
            task_id = self.context.get("cron_task_id")
            if not isinstance(task_id, str):
                raise ValueError("Cron history requires a server-bound task identity")
            self.pa.tasks.execution(task_id)
        self.context.response.answer = await self.pa.get_chat_messages(
            limit=self.context.get("limit", 10),
            offset=self.context.get("offset", 0),
            round_id=self.context.get("pa_round_id") if self.context.get("pa_caller") == "proactive" else None,
            cron_task_id=self.context.get("cron_task_id") if self.context.get("pa_caller") == "cron" else None,
        )
        return self.context.response
