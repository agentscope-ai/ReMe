"""One event- or time-triggered Agent Runtime and its reliable commit boundary."""

import asyncio
from contextlib import nullcontext
from datetime import timedelta, timezone

from .base import BaseAgentRuntime
from ..inbox import EventInbox
from ..outbox import NotificationOutbox
from ....core.registry import R
from ....schema.agent_inbox import NotificationRef
from ....schema.proactive_agent import ProactiveRuntimeConfig, ProactiveRuntimeDefinition
from ....utils.async_utils import complete_in_thread
from ....utils.cron_utils import next_cron_time
from ....utils.files.path_lock import get_path_lock
from ....utils.proactive_record import is_date_directory, is_round_record, split_record
from ....schema.proactive_agent import ProactiveRecordDecision
from ....utils.timezone_utils import resolve_timezone


@R.register("proactive")
class ProactiveAgentRuntime(BaseAgentRuntime):
    """Hold per-task state; a Job executes each round under execution()."""

    def __init__(
        self,
        agent_type,
        notification_target="Remy",
        enabled=True,
        schedule=None,
        min_trigger_interval_seconds=0,
        max_batch_events=20,
        notify_window=None,
        initial_task=None,
        max_local_search_calls=10,
        recent_history_hours=24,
        **kwargs,
    ):
        kwargs.setdefault("skills", ["proactive-record"])
        super().__init__(**kwargs)
        if initial_task is not None and (not isinstance(initial_task, str) or not initial_task.strip()):
            raise ValueError("initial_task must not be blank")
        self.initial_task = initial_task
        self.initial_config = ProactiveRuntimeConfig(
            **self.binding.model_dump(),
            agent_name=self.name,
            agent_type=agent_type,
            enabled=enabled,
            schedule=schedule,
            min_trigger_interval_seconds=min_trigger_interval_seconds,
            max_batch_events=max_batch_events,
            notify_window=notify_window,
            max_local_search_calls=max_local_search_calls,
            recent_history_hours=recent_history_hours,
        )
        self.get_skills()
        self.notification_target = (
            self.bind(notification_target, BaseAgentRuntime, optional=False)
            if isinstance(notification_target, str)
            else notification_target
        )
        self.schedule = None
        self.retry_at = None
        self.retry_items = None
        self.retry_delay = 0
        self.delivery_error = None
        self.retiring = False
        self._handoffs = {}

    def _check_open(self):
        super()._check_open()
        if self.retiring:
            raise RuntimeError("Proactive Runtime is retiring")

    @property
    def config_path(self):
        """Locate this task's durable runtime definition."""
        return self.files.checked(self.directory / "config.json")

    @property
    def task_path(self):
        """Locate the original user-editable task requirements."""
        return self.files.checked(self.directory / "TASK.md")

    @property
    def daily_directory(self):
        """Locate the shared file-native daily record root."""
        name = self.app_context.app_config.daily_dir if self.app_context else "daily"
        return self.files.path(name)

    async def _start(self):
        value = await asyncio.to_thread(self.files.read_json, self.config_path, None)
        if value is not None:
            definition = ProactiveRuntimeDefinition.model_validate(value)
            self.check_definition(definition)
        await self.initialize_task()
        if value is None:
            definition = ProactiveRuntimeDefinition(**self.initial_config.model_dump(), created_at=self.now())
            await complete_in_thread(self.files.write_json, self.config_path, definition.model_dump(mode="json"))
        self.inbox = EventInbox(
            name=f"{self.name}_inbox",
            directory=self.to_workspace_relative(self.directory / "inbox"),
            app_context=self.app_context,
        )
        self.outbox = NotificationOutbox(
            name=f"{self.name}_outbox",
            directory=self.to_workspace_relative(NotificationOutbox.directory_for(self.directory)),
            target=self.notification_target,
            app_context=self.app_context,
        )
        await self.start_children()
        for receipt in await self.inbox.receipts():
            if receipt.metadata.get("notification"):
                self._handoffs[receipt.metadata["notification"]["id"]] = receipt
                await self.inbox.retain(receipt.item_ids)

    async def definition(self, *, include_deleted=False):
        """Read the definition and reject changes to immutable runtime wiring."""
        value = await asyncio.to_thread(self.files.read_json, self.config_path, None)
        if value is None:
            raise ValueError(f"Task definition is missing: {self.name}")
        definition = ProactiveRuntimeDefinition.model_validate(value)
        self.check_definition(definition)
        if (definition.deleted_at is not None or self.retiring) and not include_deleted:
            from ..manager.base import TaskNotFoundError

            raise TaskNotFoundError(f"Task is deleted: {self.name}")
        return definition

    def check_definition(self, definition):
        """Reject edits to identity, input type or session capabilities while live."""
        if definition.wiring() != self.initial_config.wiring():
            raise ValueError("Persisted task definition conflicts with immutable Runtime wiring")

    async def status(self):
        """Read operational settings and state without resolving this Runtime again."""
        definition = await self.definition()
        snapshot = await self.inbox.snapshot()
        receipt = snapshot["receipt"]
        next_trigger = self.next_trigger(definition)
        return {
            "agent_name": self.name,
            "config": definition.model_dump(mode="json"),
            "task_path": self.to_workspace_relative(self.task_path),
            "running": self.worker is not None or self.execution_lock.locked(),
            "pending_events": snapshot["pending_count"],
            "in_flight_events": snapshot["in_flight_count"],
            "last_started_at": self.last_started_at.isoformat() if self.last_started_at else None,
            "last_completed_at": receipt.consume_ts.isoformat() if receipt else None,
            "next_trigger_at": next_trigger.isoformat() if next_trigger else None,
            "error": self.error or self.delivery_error,
        }

    async def initialize_task(self, initial_task=None):
        """Create missing task requirements without rewriting user-owned content."""
        content = self.initial_task if initial_task is None else initial_task
        if content is None:
            content = self.prompt.get_prompt("default_task")
        if not content.strip():
            raise ValueError("Task description must not be blank")
        await complete_in_thread(self.files.write, self.task_path, content, create_only=True)

    def owns_record(self, relative):
        """Check that a source belongs to this Runtime's daily record namespace."""
        path = self.files.path(relative)
        return (
            path.parent.parent == self.daily_directory
            and is_date_directory(path.parent)
            and is_round_record(path, self.name)
        )

    def has_record_registration(self, path):
        """Reserve a record path even when its delivery file is damaged."""
        reference_id = NotificationRef.identity(self.to_workspace_relative(path))
        return self.outbox.reference_path(reference_id).exists()

    async def receipt_record_paths(self):
        """Keep committed record names reserved after a file is moved."""
        return frozenset(
            receipt.metadata["reminder_path"]
            for receipt in await self.inbox.receipts()
            if isinstance(receipt.metadata.get("reminder_path"), str)
        )

    async def append(self, events):
        """Persist an idempotent batch, optionally waiting for pending capacity."""
        definition = await self.definition()
        if definition.agent_type != "event":
            from ..manager.base import TimeTriggeredTaskDeliveryError

            raise TimeTriggeredTaskDeliveryError(f"Time-triggered task cannot receive observations: {self.name}")
        return await self.inbox.append(events, wait=False)

    async def commit_round(self, batch, reminder, session_id, record):
        """The durable input receipt precedes asynchronous reminder transport."""
        if reminder is not None and record is None:
            raise ValueError("A reminder requires a round record")
        reference = None
        delivery = None
        async with await get_path_lock(record.path) if record is not None else nullcontext():
            if record is not None:
                await asyncio.to_thread(record.check_version, record.path)
            if reminder is not None:
                if not self.owns_record(reminder.id) or self.files.path(reminder.id) != record.path:
                    raise ValueError("Reminder does not identify this round's record")
                decision = ProactiveRecordDecision.parse(split_record(self.files.read(record.path))[0])
                if decision.reminder(reminder.id) != reminder:
                    raise ValueError("Reminder changed before commit")
                reference = NotificationRef.from_record(self.name, reminder.id, self.now())
            metadata = {
                "session_id": session_id,
                "reminder_path": reminder.id if reminder else None,
                "notification": reference.model_dump(mode="json") if reference else None,
            }
            await self.inbox.ack(batch["token"], metadata=metadata, retain=reference is not None)
            if reference is not None:
                self._handoffs[reference.id] = (await self.inbox.snapshot())["receipt"]
            self.retry_items = None
            self.last_completed_at = self.now()
            if reference is not None:
                try:
                    delivery = await self.outbox.prepare_reference(reference)
                except (OSError, ValueError) as exc:
                    self.delivery_error = str(exc)
            if delivery is not None:
                try:
                    await self.outbox.deliver(delivery.id)
                    await self.inbox.retain([item["id"] for item in batch["items"]], release=True)
                    self._handoffs.pop(reference.id, None)
                    self.delivery_error = None
                except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
                    self.delivery_error = str(exc)
        return {
            "agent_name": self.name,
            "processed": len(batch["items"]),
            "notifications": [reminder.model_dump(mode="json")] if reminder else [],
        }

    async def recover_handoffs(self):
        """Retry committed results independently of model execution and input capacity."""
        for receipt in list(self._handoffs.values()):
            value = receipt.metadata.get("notification")
            if value is None:
                continue
            await self.inbox.retain(receipt.item_ids)
            try:
                reference = NotificationRef.model_validate(value)
                if reference.source_agent != self.name or not self.owns_record(reference.record_path):
                    raise ValueError("Invalid retained reminder reference")
                delivery = await self.outbox.prepare_reference(reference)
                if delivery.status != "delivered":
                    path = self.files.path(reference.record_path)
                    async with await get_path_lock(path):
                        decision = ProactiveRecordDecision.parse(split_record(self.files.read(path))[0])
                        if decision.reminder(reference.record_path) is None:
                            raise ValueError("Committed reminder no longer requests delivery")
                        await self.outbox.deliver(delivery.id)
                await self.inbox.retain(receipt.item_ids, release=True)
                self._handoffs.pop(reference.id, None)
                self.delivery_error = None
            except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
                self.delivery_error = str(exc)
        self.live_turns.notify()

    def next_trigger(self, definition):
        """Cache the next occurrence for the current enabled schedule expression."""
        if not definition.enabled or definition.agent_type != "time":
            self.schedule = None
            return None
        key = (definition.schedule, self.app_context.app_config.timezone if self.app_context else None)
        if self.schedule is None or self.schedule[0] != key:
            self.schedule = (key, self._next_cron_time(definition.schedule, self.now()))
        return self.schedule[1]

    def _next_cron_time(self, expression, now):
        from croniter import CroniterBadDateError

        timezone_name = self.app_context.app_config.timezone if self.app_context else None
        try:
            return next_cron_time(expression, now.astimezone(resolve_timezone(timezone_name)))
        except CroniterBadDateError:
            return None

    def scheduled(self, definition, now):
        """Return a due occurrence without advancing it before queue admission."""
        scheduled = self.next_trigger(definition)
        return scheduled if scheduled is not None and scheduled.timestamp() <= now.timestamp() else None

    def advance_schedule(self, definition, now):
        """Advance only after the planned Event has been durably accepted."""
        self.schedule = (self.schedule[0], self._next_cron_time(definition.schedule, now))

    def failed_round(self, error, retry_seconds=5):
        """Record an execution failure and capped exponential retry time."""
        self.error = str(error)[:2000]
        self.retry_delay = min(max(300, retry_seconds), self.retry_delay * 2 if self.retry_delay else retry_seconds)
        self.retry_at = self.now().astimezone(timezone.utc) + timedelta(seconds=self.retry_delay)

    def completed_round(self):
        """Clear execution backoff after a successful round."""
        self.error = self.retry_at = None
        self.retry_delay = 0
