"""Evaluate reminders in a disposable fork of the current Chat conversation."""

import asyncio
import math

from .chat import BaseChatAgentStep
from ....components import R
from ....schema.agent_runtime import InboxWakeRequest


@R.register("inbox_chat_step")
class InboxChatStep(BaseChatAgentStep):
    """Keep automatic inference and tool traces outside the foreground session."""

    def __init__(self, timeout_seconds=120, **kwargs):
        super().__init__(**kwargs)
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a finite positive number")
        self.timeout_seconds = timeout_seconds

    async def execute(self):
        request = InboxWakeRequest.model_validate(self.parameters())
        self.require_paths(self.workspace_path)
        async with asyncio.timeout(self.timeout_seconds):
            return await self._execute(request)

    async def _execute(self, request: InboxWakeRequest):
        runtime = self.agent_runtime
        if not any(identifier in runtime.notifications.pending for identifier in request.msg_ids):
            self.context.response.answer = {"skipped": "already_read"}
            return self.context.response
        if runtime.chat_lock.locked():
            self.context.response.answer = {"skipped": "busy"}
            return self.context.response
        async with runtime.chat_lock:
            if runtime.closed:
                raise RuntimeError("Agent runtime is closed")
            idle = max(0, (runtime.now() - runtime.notifications.last_activity_at).total_seconds())
            if request.priority != "P0" and idle < getattr(
                runtime.notifications.config,
                f"{request.priority.lower()}_idle_seconds",
            ):
                self.context.response.answer = {"skipped": "not_idle"}
                return self.context.response
            options = self.session_options()
            session_id = await self.ensure_session(options)
            status = await runtime.agent_wrapper.get_session_status(session_id)
            if status.status not in {"idle", "not_loaded"}:
                self.context.response.answer = {"skipped": "busy"}
                return self.context.response
        async with runtime.notifications.turn(session_id, request.priority, request.msg_ids) as turn:
            reply = await runtime.agent_wrapper.reply(
                self.prompt_format("hint", urgent=request.priority == "P0"),
                **{
                    **options,
                    **self.web_options(False),
                    "resume": session_id,
                    "fork_session": True,
                    "ephemeral": True,
                    "sandbox": "read-only",
                    "job_tools": ["read_inbox", "send_chat_msg"],
                    "injected_job_kwargs": {"_agent_id": "chat", "_inbox_turn": turn.id},
                },
            )
            if not turn.done:
                raise RuntimeError(reply.get("error") or "Inbox fork ended without submitting a reminder decision")
            self.context.response.answer = {"sent": bool(turn.content)}
        return self.context.response
