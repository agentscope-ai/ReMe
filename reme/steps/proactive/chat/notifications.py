"""Single-batch reminder decisions and their BackgroundJob-owned scheduling."""

import asyncio
import math
from datetime import timedelta, timezone

from .chat import BaseChatAgentStep
from ....core.registry import R
from ....schema.proactive_runtime import ProcessChatNotificationsRequest


async def notifications_allowed(manager, items):
    """Recheck sender activation and notification windows for a selected batch."""
    for item in items:
        try:
            definition = await manager.definition(item["from_agent_name"])
        except ValueError:
            return False
        if (
            not definition.enabled
            or definition.notify_window is not None
            and not definition.notify_window.allows(manager.now())
        ):
            return False
    return True


class BaseChatNotificationsStep(BaseChatAgentStep):
    """Share localized inbox rendering between batch selection and inference."""

    def format_item(self, item):
        """Render one reminder with the invocation's language and prompt overrides."""
        return self.prompt_format("inbox_item", **item)


@R.register("process_chat_notifications_step")
class ProcessChatNotificationsStep(BaseChatNotificationsStep):
    """Decision inference is invocation-scoped; Runtime owns the batch capability."""

    def __init__(self, timeout_seconds=120, **kwargs):
        super().__init__(**kwargs)
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        self.timeout_seconds = timeout_seconds

    async def execute(self):
        request = ProcessChatNotificationsRequest.model_validate(self.parameters())
        runtime = self.agent_runtime
        self.require_paths(self.workspace_path)
        async with runtime.track(), asyncio.timeout(self.timeout_seconds):
            reserved = await runtime.recover_deliveries()
            if set(request.msg_ids) & reserved:
                raise RuntimeError("Notification delivery recovery is pending")
            if not set(request.msg_ids) & await runtime.inbox.pending_ids():
                self.context.response.answer = {"skipped": "already_processed"}
                return self.context.response
            if request.priority != "P0" and runtime.busy:
                self.context.response.answer = {"skipped": "busy"}
                return self.context.response
            session_id = await self.ensure_session()
            async with runtime.notification_turn(
                session_id,
                request.priority,
                request.msg_ids,
                format_item=self.format_item,
                eligible=lambda items: notifications_allowed(self.agent_manager, items),
            ) as turn:
                hint = self.prompt_format(
                    "hint",
                    inbox_items="\n\n---\n\n".join(self.format_item(item) for item in turn.items),
                )
                reply = await runtime.agent_wrapper.reply(
                    hint,
                    system_prompt=self.session_options()["system_prompt"],
                    job_tools=["submit_chat_reply"],
                    skills=[],
                    resume=session_id,
                    fork_session=True,
                    ephemeral=True,
                    sandbox="read-only",
                    injected_job_kwargs={"_agent_name": runtime.name, "_inbox_turn": turn.id},
                )
                if not turn.done:
                    raise RuntimeError(reply.get("error") or "Notification fork ended without a send/skip decision")
                self.context.response.answer = {"sent": bool(turn.content)}
        return self.context.response


@R.register("chat_notifications_loop_step")
class ChatNotificationsLoopStep(BaseChatNotificationsStep):
    """Select eligible batches without keeping state on a reusable Step instance."""

    async def _process(self, priority, identifiers):
        runtime = self.agent_runtime
        runtime.last_started_at = runtime.now()
        try:
            response = await self.run_job(
                self.kwargs.get("process_job", "process_chat_notifications"),
                priority=priority,
                msg_ids=identifiers,
            )
            if not response.success:
                error = response.metadata.get("reme.job_error", {})
                runtime.blocked = error.get("type") in {
                    "TypeError",
                    "ImportError",
                    "ModuleNotFoundError",
                    "NotImplementedError",
                }
                raise RuntimeError(str(response.answer))
            runtime.error = runtime.retry_at = None
            runtime.failures = 0
            runtime.last_completed_at = runtime.now()
        except Exception as exc:
            runtime.error = str(exc)[:2000]
            runtime.failures += 1
            delay = min(300, 5 * 2 ** min(runtime.failures - 1, 8))
            runtime.retry_at = runtime.now().astimezone(timezone.utc) + timedelta(seconds=delay)
            self.logger.error(f"Notification processing failed; retry in {delay}s: {exc}")
        finally:
            runtime.notification_worker = None
            runtime.live_turns.notify()

    async def tick(self):
        """Run one scheduling pass without storing state on the Step."""
        runtime = self.agent_runtime
        if runtime.closed or not runtime.enabled or runtime.blocked or runtime.notification_worker is not None:
            return
        reserved = await runtime.recover_deliveries()
        if runtime.foreground_waiting:
            return
        now = runtime.now()
        if runtime.retry_at is not None and runtime.retry_at.timestamp() > now.timestamp():
            return
        idle = max(0, now.timestamp() - runtime.last_activity_at.timestamp())
        choices = {}
        for reference in await runtime.inbox.pending_references():
            if reference.id in reserved:
                continue
            try:
                item = await runtime.reminder_item(reference)
                if not await notifications_allowed(self.agent_manager, [item]):
                    continue
                priority = item["priority"]
                if priority == "P0":
                    if (
                        now.timestamp() - reference.created_at.timestamp()
                        < runtime.notification_config.p0_delay_seconds
                    ):
                        continue
                elif runtime.busy or idle < getattr(runtime.notification_config, f"{priority.lower()}_idle_seconds"):
                    continue
                size = len(self.format_item(item).encode())
                if size > runtime.inbox.claim_max_bytes:
                    runtime.error = f"Reminder is too large for one batch: {reference.record_path}"
                    continue
                choices.setdefault(priority, []).append((reference.created_at, reference.id, size))
            except (OSError, ValueError) as exc:
                runtime.error = str(exc)
        for priority, values in sorted(choices.items()):
            selected, size = [], 0
            for _, identifier, length in sorted(values):
                length += 7 if selected else 0
                if len(selected) >= runtime.inbox.claim_max_items or size + length > runtime.inbox.claim_max_bytes:
                    break
                selected.append(identifier)
                size += length
            if selected:
                runtime.notification_worker = asyncio.create_task(
                    self._process(priority, selected),
                    name="notifications:Remy",
                )
                break

    async def execute(self):
        stop = self.context.stop_event
        if not isinstance(stop, asyncio.Event):
            raise RuntimeError("Notification scheduling requires a BackgroundJob stop_event")
        runtime = self.agent_runtime
        try:
            while not stop.is_set() and not runtime.closed:
                try:
                    await self.tick()
                except (OSError, ValueError, RuntimeError) as exc:
                    runtime.error = str(exc)
                    self.logger.error(f"Notification scheduler: {exc}")
                try:
                    await asyncio.wait_for(stop.wait(), timeout=1)
                except TimeoutError:
                    pass
        finally:
            if runtime.notification_worker is not None:
                runtime.notification_worker.cancel()
                await asyncio.gather(runtime.notification_worker, return_exceptions=True)
        return self.context.response
