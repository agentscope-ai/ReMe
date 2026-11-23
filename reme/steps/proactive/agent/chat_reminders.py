"""Evaluate reminders in a disposable fork of the current Chat conversation."""

import asyncio
from contextlib import nullcontext
import math

from ....constants import CHAT_AGENT_NAME
from .chat import BaseChatAgentStep
from ....core.registry import R
from ....schema.proactive_runtime import ProcessChatRemindersRequest


@R.register("process_chat_reminders_step")
class ProcessChatRemindersStep(BaseChatAgentStep):
    """Keep automatic inference and tool traces outside the foreground session."""

    def __init__(self, timeout_seconds=120, **kwargs):
        super().__init__(**kwargs)
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a finite positive number")
        self.timeout_seconds = timeout_seconds

    async def execute(self):
        request = ProcessChatRemindersRequest.model_validate(self.parameters())
        self.require_paths(self.workspace_path)
        async with asyncio.timeout(self.timeout_seconds):
            return await self._execute(request)

    async def _execute(self, request: ProcessChatRemindersRequest):
        runtime = self.proactive_runtime
        if not any(identifier in runtime.notifications.pending for identifier in request.msg_ids):
            self.context.response.answer = {"skipped": "already_read"}
            return self.context.response
        active_fork = request.priority == "P0"
        if not active_fork and runtime.chat_lock.locked():
            self.context.response.answer = {"skipped": "busy"}
            return self.context.response
        async with nullcontext() if active_fork else runtime.chat_lock:
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
            current = await runtime.sessions.current(CHAT_AGENT_NAME) if active_fork else None
            session_id = current.session_id if current else await self.ensure_session(options)
            status = await runtime.agent_wrapper.get_session_status(session_id)
            if status.status not in ({"idle", "not_loaded", "active"} if active_fork else {"idle", "not_loaded"}):
                self.context.response.answer = {"skipped": "busy"}
                return self.context.response
            if active_fork and (
                len(runtime.chat_tasks) > 1 or (runtime.chat_lock.locked() and status.status != "active")
            ):
                # Queued user input is not necessarily part of the native snapshot yet.
                self.context.response.answer = {"skipped": "busy"}
                return self.context.response
        async with runtime.notifications.turn(session_id, request.priority, request.msg_ids) as turn:
            if not turn.done and (inbox_items := await runtime.notifications.read_turn(turn)):
                reply = await runtime.agent_wrapper.reply(
                    self.prompt_format("hint", inbox_items=inbox_items),
                    system_prompt=options["system_prompt"],
                    job_tools=["submit_chat_reminder"],
                    resume=session_id,
                    fork_session=True,
                    ephemeral=True,
                    sandbox="read-only",
                    injected_job_kwargs={"_agent_name": CHAT_AGENT_NAME, "_inbox_turn": turn.id},
                )
                if not turn.done:
                    raise RuntimeError(reply.get("error") or "Inbox fork ended without submitting a reminder decision")
            self.context.response.answer = {"sent": bool(turn.content)}
        return self.context.response
