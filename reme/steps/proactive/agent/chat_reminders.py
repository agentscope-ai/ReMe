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
        manager = self.proactive_manager
        if not any(identifier in manager.notifications.pending for identifier in request.msg_ids):
            self.context.response.answer = {"skipped": "already_read"}
            return self.context.response
        active_fork = request.priority == "P0"
        if not active_fork and manager.chat_lock.locked():
            self.context.response.answer = {"skipped": "busy"}
            return self.context.response
        async with nullcontext() if active_fork else manager.chat_lock:
            if manager.closed:
                raise RuntimeError("Proactive Manager is closed")
            idle = max(0, (manager.now() - manager.notifications.last_activity_at).total_seconds())
            if request.priority != "P0" and idle < getattr(
                manager.notifications.config,
                f"{request.priority.lower()}_idle_seconds",
            ):
                self.context.response.answer = {"skipped": "not_idle"}
                return self.context.response
            options = self.session_options()
            current = await manager.sessions.current(CHAT_AGENT_NAME) if active_fork else None
            session_id = current.session_id if current else await self.ensure_session(options)
            status = await manager.agent_wrapper.get_session_status(session_id)
            if status.status not in ({"idle", "not_loaded", "active"} if active_fork else {"idle", "not_loaded"}):
                self.context.response.answer = {"skipped": "busy"}
                return self.context.response
            if active_fork and (
                len(manager.chat_tasks) > 1 or (manager.chat_lock.locked() and status.status != "active")
            ):
                # Queued user input is not necessarily part of the native snapshot yet.
                self.context.response.answer = {"skipped": "busy"}
                return self.context.response
        async with manager.notifications.turn(session_id, request.priority, request.msg_ids) as turn:
            if not turn.done and (inbox_items := await manager.notifications.read_turn(turn)):
                reply = await manager.agent_wrapper.reply(
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
