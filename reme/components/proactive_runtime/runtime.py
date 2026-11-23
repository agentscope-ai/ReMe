"""Execute and schedule one named Proactive Runtime."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from ...schema.agent_activity import AgentActivity
from ...schema.agent_inbox import InboxReceipt
from ...schema.event import Event
from ...schema.proactive_agent import CRON_EVENT_SOURCE, ProactiveRuntimeDefinition
from ...utils.async_utils import finish_cleanup

if TYPE_CHECKING:
    from ..proactive_manager.local_proactive_manager import LocalProactiveManager


class ProactiveRuntime:
    """Own one named inbox, active worker, execution lock and retry state."""

    def __init__(self, manager: LocalProactiveManager, agent_name: str):
        self.manager = manager
        self.agent_name = agent_name
        self.worker: asyncio.Task | None = None
        self.tasks: set[asyncio.Task] = set()
        self.lock = asyncio.Lock()
        self.schedule: tuple[tuple[str, str | None], datetime | None] | None = None
        self.started_at: datetime | None = None
        self.error: str | None = None
        self.retry_at: datetime | None = None
        self.retry_items: list[str] | None = None
        self.retry_delay = 0.0

    async def definition(self) -> ProactiveRuntimeDefinition:
        """Read the current definition with the manager's mutation lock held."""
        if self.manager.runtimes.get(self.agent_name) is not self:
            raise ValueError(f"Proactive Runtime is no longer registered: {self.agent_name}")
        return await self.manager.definition(self.agent_name)

    async def mount(self, definition: ProactiveRuntimeDefinition):
        """Route this runtime's events to its workspace inbox directory."""
        manager = self.manager
        directory = manager.registry.directory_for(definition) / "inbox"
        await manager.inbox.mount(self.agent_name, directory.relative_to(manager.workspace_path).as_posix())

    async def append(self, events: list[Event]) -> list[str]:
        """Append to this inbox and wake the scheduler."""
        identifiers = await self.manager.inbox.append(self.agent_name, events, wait=False)
        self.manager.changed.set()
        return identifiers

    @asynccontextmanager
    async def track(self):
        """Track in-flight requests before they can block."""
        if self.manager.closed:
            raise RuntimeError("Proactive Manager is closed")
        task = asyncio.current_task()
        self.tasks.add(task)
        try:
            yield
        finally:
            self.tasks.discard(task)

    async def stop(self):
        """Cancel this runtime's worker and requests before unregistering it."""
        tasks = self.tasks.copy()
        if self.worker is not None:
            tasks.add(self.worker)
        for task in tasks:
            task.cancel()
        if tasks:
            await finish_cleanup(asyncio.gather(*tasks, return_exceptions=True))

    def configure(self, definition: ProactiveRuntimeDefinition):
        """Forget a scheduled trigger when it ceases to apply."""
        if not definition.enabled or (self.schedule is not None and self.schedule[0][0] != definition.cron):
            self.schedule = None

    def next_trigger(self, definition: ProactiveRuntimeDefinition) -> datetime | None:
        """Read the next scheduled trigger without mutating execution state."""
        if definition.enabled and definition.cron is not None:
            return self._cron_schedule(definition.cron, self.manager.now())[1]
        return None

    def recover_receipt(self, receipt: InboxReceipt | None) -> None:
        """Clear a retry selection if its acknowledgement was already committed."""
        if self.retry_items and receipt is not None and set(self.retry_items).issubset(receipt.item_ids):
            self.retry_items = None

    async def run(self, execute, *, record=None):
        """Commit successful execution before handing asynchronous delivery to the outbox."""
        manager = self.manager
        agent_name = self.agent_name
        async with self.track(), self.lock:
            async with manager.management_lock:
                agent = await self.definition()
                if not agent.enabled:
                    return {"agent_name": agent_name, "skipped": "disabled"}
                await self.mount(agent)
                if self.retry_items:
                    self.recover_receipt((await manager.inbox.snapshot(agent_name))["receipt"])
                batch = await manager.inbox.claim(
                    agent_name,
                    limit=agent.max_batch_events,
                    item_ids=self.retry_items,
                    batch_key=lambda item: item["id"] if item["source"] == CRON_EVENT_SOURCE else None,
                )
                if batch is None:
                    return {"agent_name": agent_name, "processed": 0}
                self.retry_items = [item["id"] for item in batch["items"]]
                owns_worker = self.worker is None
                if owns_worker:
                    self.worker = asyncio.current_task()
            try:
                self.started_at = manager.now()
                manager.activities[agent_name] = AgentActivity(
                    session_id="",
                    turn_id=batch["token"],
                    source="task",
                    task_id=agent_name,
                    started_at=self.started_at,
                    phase="working",
                )
                reminder, session_id = await execute(agent, batch["items"])
                manager.activities[agent_name] = manager.activities[agent_name].model_copy(
                    update={"session_id": session_id},
                )

                async def commit():
                    async with manager.outbox.handoff(agent_name, reminder, record) as ref:
                        await manager.inbox.ack(
                            agent_name,
                            batch["token"],
                            metadata={"reminder_path": ref, "session_id": session_id},
                            retain=ref is not None,
                        )
                        self.retry_items = None

                await finish_cleanup(commit())
                manager.logger.info(
                    f"Proactive agent={agent_name} session={session_id} result=committed "
                    f"processed={len(batch['items'])} notifications={int(reminder is not None)}",
                )
                manager.activities[agent_name] = manager.activities[agent_name].model_copy(
                    update={"phase": "completed"},
                )
                manager.changed.set()
                return {
                    "agent_name": agent_name,
                    "processed": len(batch["items"]),
                    "notifications": [reminder.model_dump()] if reminder is not None else [],
                }
            except BaseException as exc:
                phase = "interrupted" if isinstance(exc, asyncio.CancelledError) else "failed"
                manager.activities[agent_name] = manager.activities[agent_name].model_copy(update={"phase": phase})
                raise
            finally:
                try:
                    await finish_cleanup(manager.inbox.nack(agent_name, batch["token"]))
                except ValueError:
                    pass
                except RuntimeError as exc:
                    manager.logger.warning(f"Could not release failed inbox claim for {agent_name}: {exc}")
                finally:
                    if owns_worker:
                        self.worker = None

    def _next_cron_time(self, expression: str, now: datetime) -> datetime | None:
        """Return no trigger for an exhausted schedule without hiding its retained task."""
        from croniter import CroniterBadDateError, croniter

        anchor = now if self.manager.app_context.app_config.timezone else now.replace(tzinfo=None)
        try:
            next_fire = croniter(expression, anchor).get_next(datetime)
        except CroniterBadDateError:
            return None
        if next_fire.tzinfo is None:
            next_fire = next_fire.astimezone()
        return next_fire

    def _cron_schedule(self, expression: str, now: datetime):
        key = (expression, self.manager.app_context.app_config.timezone)
        if self.schedule is not None and self.schedule[0] == key:
            return self.schedule
        return key, self._next_cron_time(expression, now)

    async def _schedule_cron(self, agent):
        now = self.manager.now()
        self.schedule = self._cron_schedule(agent.cron, now)
        key, scheduled = self.schedule
        if scheduled is None or scheduled.timestamp() > now.timestamp():
            return
        await self.append(
            [
                Event(
                    id=f"cron-{self.agent_name}-{int(scheduled.timestamp())}",
                    event_ts=scheduled,
                    source=CRON_EVENT_SOURCE,
                    description=f"Scheduled trigger: {scheduled.isoformat()}",
                ),
            ],
        )
        self.schedule = key, self._next_cron_time(agent.cron, now)

    async def tick(self, definition, run_job):
        """Queue a due trigger and start at most one eligible worker."""
        self.configure(definition)
        if not definition.enabled:
            return
        await self.mount(definition)
        manager = self.manager
        if definition.cron is not None:
            try:
                await self._schedule_cron(definition)
            except TimeoutError:
                manager.logger.warning(f"Cron input deferred for {self.agent_name}: inbox capacity is full")
        if self.worker is not None or (self.retry_at is not None and self.retry_at > manager.now()):
            return
        snapshot = await manager.inbox.snapshot(self.agent_name)
        if not snapshot["pending_count"]:
            return
        receipt = snapshot["receipt"]
        if (
            receipt is not None
            and definition.min_trigger_interval_seconds
            and (manager.now() - receipt.consume_ts).total_seconds() < definition.min_trigger_interval_seconds
        ):
            self.recover_receipt(receipt)
            item = await manager.inbox.peek(self.agent_name, item_ids=self.retry_items)
            if item is None or item["source"] != CRON_EVENT_SOURCE:
                return
        self.worker = asyncio.create_task(self._work(run_job), name=f"agent:{self.agent_name}")

    async def _work(self, run_job):
        manager = self.manager
        agent_name = self.agent_name
        try:
            response = await run_job("run_proactive_agent", agent_name=agent_name)
            if not response.success:
                raise RuntimeError(str(response.answer))
            self.error = self.retry_at = None
            self.retry_delay = 0
        except Exception as exc:
            self.error = str(exc)[:1000]
            delay = min(
                max(300, manager.retry_seconds),
                self.retry_delay * 2 if self.retry_delay else manager.retry_seconds,
            )
            self.retry_delay = delay
            self.retry_at = manager.now() + timedelta(seconds=delay)
            manager.logger.exception(f"Proactive Runtime {agent_name} failed; retry in {delay:g}s: {exc}")
        finally:
            self.worker = None
            manager.changed.set()
