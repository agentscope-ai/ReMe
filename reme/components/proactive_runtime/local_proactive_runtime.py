"""Application-owned agent coordination with file-native recovery."""

import asyncio
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
from zoneinfo import ZoneInfo

from watchfiles import awatch

from ...constants import CHAT_AGENT_NAME
from .base_proactive_runtime import BaseProactiveRuntime
from ...utils.files.workspace_files import WorkspaceFiles
from ._internal.notifications import ChatNotifications
from ._internal.outbox import ReminderOutbox
from ._internal.registry import AgentRegistry
from ._internal.sessions import AgentSessions
from ._internal.live_turn import LiveTurns
from ..agent_wrapper import BaseAgentWrapper
from ..agent_inbox import BaseAgentInbox
from ...core.registry import R
from ...schema.proactive_runtime import (
    ChatAgentConfig,
    NotificationConfig,
)
from ...schema.agent_activity import AgentActivity
from ...schema.agent_inbox import InboxReceipt
from ...schema.application_config import ApplicationConfig
from ...schema.event import Event
from ...schema.proactive_agent import CRON_EVENT_SOURCE, CreateProactiveAgentConfig
from ...utils.async_utils import complete_in_thread, finish_cleanup
from ...utils.cron_utils import next_cron_time
from ...utils.timezone_utils import system_timezone


@dataclass
class _Execution:
    worker: asyncio.Task | None = None
    tasks: set[asyncio.Task] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    schedule: tuple[tuple[str, str | None], datetime | None] | None = None
    started_at: datetime | None = None
    error: str | None = None
    retry_at: datetime | None = None
    retry_items: list[str] | None = None
    retry_delay: float = 0

    def recover_receipt(self, receipt: InboxReceipt | None) -> None:
        """Clear a retry selection if its acknowledgement was already committed."""
        if self.retry_items and receipt is not None and set(self.retry_items).issubset(receipt.item_ids):
            self.retry_items = None


@R.register("local")
class LocalProactiveRuntime(BaseProactiveRuntime):
    """Own task lifetimes and rebuild execution facts from their source logs."""

    def __init__(
        self,
        agent_wrapper="default",
        inbox="default",
        notifications=None,
        retry_seconds=5,
        agents=None,
        chat=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if type(retry_seconds) not in (int, float) or not math.isfinite(retry_seconds) or retry_seconds < 0:
            raise ValueError("retry_seconds must be a finite nonnegative number")
        if agents is not None and not isinstance(agents, list):
            raise ValueError("agents must be a list of initial task configurations")
        self.initial_agents = [CreateProactiveAgentConfig.model_validate(value) for value in agents or []]
        if len({agent.agent_name.casefold() for agent in self.initial_agents}) != len(self.initial_agents):
            raise ValueError("initial Agent names must be unique, including case")
        self.initial_chat = ChatAgentConfig.model_validate(chat if chat is not None else {})
        self.agent_wrapper = self.bind(agent_wrapper, BaseAgentWrapper, optional=False)
        self.inbox = self.bind(inbox, BaseAgentInbox, optional=False)
        self.files = WorkspaceFiles(self.workspace_path)
        config = self.app_context.app_config if self.app_context else ApplicationConfig()
        self.agents = AgentRegistry(self.files, config.reme_agent_dir)
        self.sessions = AgentSessions(self.agents)
        self.live_turns = LiveTurns()
        self.outbox = ReminderOutbox(self, config.daily_dir or "daily")
        self.notifications = ChatNotifications(
            self,
            NotificationConfig.model_validate(notifications if notifications is not None else {}),
        )
        self.retry_seconds = retry_seconds
        self.chat_lock = asyncio.Lock()
        self.chat_tasks = set()
        self.activities = {}
        self.executions = {}
        self.changed = asyncio.Event()
        self.management_lock = asyncio.Lock()
        self.watch_stop = asyncio.Event()
        self.watcher = None
        self.closed = True

    @property
    def workers(self):
        """Read active workers from their owning execution states."""
        return {name: state.worker for name, state in self.executions.items() if state.worker is not None}

    @property
    def agent_tasks(self):
        """Read request ownership without maintaining another mutable registry."""
        return {task: name for name, state in self.executions.items() for task in state.tasks}

    def execution(self, agent_name):
        """Own one task's locks, work and retry state together."""
        if agent_name not in self.executions:
            self.executions[agent_name] = _Execution()
        return self.executions[agent_name]

    async def stop_workers(self):
        """Stop scheduled work without closing foreground sessions or the component."""
        await self.notifications.close()
        tasks = set(self.workers.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await finish_cleanup(asyncio.gather(*tasks, return_exceptions=True))

    def now(self):
        """Use the application timezone for dates and trigger calculation."""
        timezone = self.app_context.app_config.timezone if self.app_context else None
        return datetime.now(ZoneInfo(timezone)) if timezone else datetime.now().astimezone()

    async def _start(self):
        if self.app_context is None:
            raise RuntimeError("LocalProactiveRuntime requires an ApplicationContext")
        await complete_in_thread(
            self.files.write,
            self.agents.chat_memory_path(),
            self.initial_chat.initial_memory,
            create_only=True,
        )
        existing = {name.casefold() for name in await asyncio.to_thread(self.agents.load, include_deleted=True)}
        for initial in self.initial_agents:
            if initial.agent_name.casefold() not in existing:
                await self.create(initial)
        for agent in (await asyncio.to_thread(self.agents.load)).values():
            await self._mount_inbox(agent)
        await self.outbox.sync()
        self.notifications.reset()
        self.closed = False
        self.changed.set()
        self.watch_stop.clear()
        self.watcher = asyncio.create_task(self._watch(), name="agents:watch")

    async def _close(self):
        self.closed = True
        self.live_turns.clear()
        self.watch_stop.set()
        self.changed.set()
        await self.stop_workers()
        tasks = set(self.agent_tasks) | self.chat_tasks
        if self.watcher is not None:
            tasks.add(self.watcher)
        for task in tasks:
            task.cancel()
        if tasks:
            await finish_cleanup(asyncio.gather(*tasks, return_exceptions=True))
        self.activities.clear()
        self.chat_tasks.clear()
        self.executions.clear()
        self.watcher = None

    async def _watch(self):
        """One watcher plus timeout rescans covers edits, replacement and missed notifications."""
        async with aclosing(
            awatch(
                self.agents.directory,
                stop_event=self.watch_stop,
                yield_on_timeout=True,
                rust_timeout=1000,
                debounce=100,
                step=50,
            ),
        ) as watcher:
            async for _ in watcher:
                if self.closed:
                    break
                try:
                    await self.outbox.sync()
                except Exception as exc:
                    self.logger.error(f"Could not reconcile agent files: {exc}")
                self.changed.set()

    async def _mount_inbox(self, agent):
        directory = (self.agents.directory_for(agent) / "inbox").relative_to(self.workspace_path).as_posix()
        await self.inbox.mount(agent.agent_name, directory)

    async def definition(self, agent_name):
        """Read current user-owned configuration rather than a stale runtime copy."""
        return await asyncio.to_thread(self.agents.get, agent_name)

    async def create(self, config):
        """Persist a task and its inbox directory before returning success."""
        async with self.management_lock:
            agent, restored = await complete_in_thread(
                self.agents.create,
                config,
                self.now(),
            )
            status = await self._status(agent)
            self.changed.set()
            return {**status, "restored": restored}

    async def update(self, agent_name, config):
        """Apply settings and invalidate the timer when its schedule changes."""
        async with self.management_lock:
            agent = await complete_in_thread(self.agents.update, agent_name, config, self.now())
            execution = self.execution(agent_name)
            if not agent.enabled or (execution.schedule is not None and execution.schedule[0][0] != agent.cron):
                execution.schedule = None
            self.changed.set()
            return await self._status(agent)

    async def delete(self, agent_name):
        """Stop owned work while retaining a restorable identity and all source files."""
        async with self.management_lock:
            tasks = {task for task, owner in self.agent_tasks.items() if owner == agent_name}
            if agent_name in self.workers:
                tasks.add(self.workers[agent_name])
            for task in tasks:
                task.cancel()
            if tasks:
                await finish_cleanup(asyncio.gather(*tasks, return_exceptions=True))
            session = await self.sessions.current(agent_name)
            if session is not None:
                await finish_cleanup(self.agent_wrapper.release_session(session.session_id))
            agent = await complete_in_thread(self.agents.delete, agent_name, self.now())
            self.execution(agent_name).schedule = None
            self.changed.set()
            return {"agent_name": agent.agent_name, "deleted": True, "files_retained": True}

    async def append_events(self, agent_name: str, events: list[Event]) -> dict:
        """Accept external observations for a background task."""
        if any(event.source == CRON_EVENT_SOURCE for event in events):
            raise ValueError("cron is a reserved framework source")
        async with self.management_lock:
            agent = await self.definition(agent_name)
            await self._mount_inbox(agent)
        # Register the validated task before append can suspend on a full inbox.
        async with self._agent_task(agent_name):
            identifiers = await self._append(agent, events)
        return {"agent_name": agent_name, "event_ids": identifiers}

    async def _append(self, agent, events: list[Event], *, wait=True) -> list[str]:
        identifiers = await self.inbox.append(agent.agent_name, events, wait=wait)
        self.changed.set()
        return identifiers

    async def status(self, agent_name):
        """Inspect without acquiring a model-generation lock."""
        return await self._status(await self.definition(agent_name))

    async def _status(self, agent):
        agent_name = agent.agent_name
        await self._mount_inbox(agent)
        snapshot = await self.inbox.snapshot(agent_name)
        receipt = snapshot["receipt"]
        execution = self.execution(agent_name)
        task_path = self.agents.task_path(agent)
        error = execution.error
        next_trigger = None
        if agent.enabled and agent.cron is not None:
            _, next_trigger = self._cron_schedule(agent.cron, execution, self.now())
            if next_trigger is None:
                error = f"Cron schedule has no future trigger: {agent.cron}"
        return {
            "agent_name": agent_name,
            "config": agent.model_dump(mode="json", exclude={"created_at", "deleted_at"}),
            "created_at": agent.created_at.isoformat(),
            "task_path": task_path.relative_to(self.workspace_path).as_posix(),
            "running": execution.worker is not None,
            "pending_events": snapshot["pending_count"],
            "in_flight_events": snapshot["in_flight_count"],
            "last_started_at": (execution.started_at.isoformat() if execution.started_at else None),
            "last_completed_at": receipt.consume_ts.isoformat() if receipt else None,
            "next_trigger_at": next_trigger.isoformat() if next_trigger is not None else None,
            "error": error,
        }

    async def list_agents(self):
        """List task identities independently of chat and native session counts."""
        agents = await asyncio.to_thread(self.agents.load)
        items = [await self._status(agent) for agent in agents.values()]
        return {"items": items, "total": len(items)}

    @asynccontextmanager
    async def chat_turn(self):
        """Serialize Chat and explicit session rotation while owning shutdown cancellation."""
        if self.closed:
            raise RuntimeError("Agent runtime is closed")
        task = asyncio.current_task()
        self.notifications.foreground()
        self.chat_tasks.add(task)
        try:
            async with self.chat_lock:
                try:
                    yield
                finally:
                    self.notifications.activity()
        finally:
            self.chat_tasks.discard(task)

    async def list_views(self):
        """Inspect all visible identities, including Remy, without creating sessions."""
        from ._internal.views import AgentViews

        return await AgentViews(self).list()

    async def view_agent(self, agent_name):
        """Project settings and execution facts without reading message history."""
        from ._internal.views import AgentViews

        return (await AgentViews(self).view(agent_name)).model_dump(mode="json")

    async def read_agent(self, agent_name):
        """Read the current formal session without activating a temporary query fork."""
        from ._internal.views import AgentViews

        return await AgentViews(self).read(agent_name)

    async def watch_agent(self, agent_name):
        """Stream reconstructible snapshots and changes; reconnect starts with a snapshot."""
        from ._internal.views import AgentViews

        async with aclosing(AgentViews(self).events(agent_name)) as events:
            async for event in events:
                yield event

    async def list_inputs(self, agent_name, **kwargs):
        """Page a task's input events or Chat's reminder view."""
        if agent_name == CHAT_AGENT_NAME:
            return await self.outbox.list(**kwargs)
        await self._mount_inbox(await self.definition(agent_name))
        page = await self.inbox.read(agent_name, **kwargs)
        page["items"] = [{"event": item.pop("item"), **item} for item in page["items"]]
        return page

    async def list_chat_sessions(self, offset=0, limit=10):
        """Listing is side-effect-free, including before the first Chat turn."""
        return await self.sessions.list(CHAT_AGENT_NAME, offset, limit)

    @asynccontextmanager
    async def _agent_task(self, agent_name):
        """Own Agent calls before they wait for locks so shutdown and deletion can cancel them."""
        if self.closed:
            raise RuntimeError("Agent runtime is closed")
        task = asyncio.current_task()
        execution = self.execution(agent_name)
        execution.tasks.add(task)
        try:
            yield
        finally:
            execution.tasks.discard(task)

    @asynccontextmanager
    async def query_session(self, agent_name, options):
        """Own a temporary fork of the latest task session, including during execution."""
        async with self._agent_task(agent_name):
            async with self.management_lock:
                await self.definition(agent_name)
                source = await self.sessions.current(agent_name)
                if source is None:
                    raise ValueError("Task has no queryable session yet")
            session_id = None
            try:

                async def fork():
                    nonlocal session_id
                    session_id = await self.agent_wrapper.create_session(
                        resume=source.session_id,
                        fork_session=True,
                        **options,
                    )
                    await self.sessions.record(agent_name, session_id, self.now(), temporary=True)

                await finish_cleanup(fork())
                yield session_id
            finally:
                if session_id is not None:
                    await finish_cleanup(self.agent_wrapper.release_session(session_id))

    async def run(self, agent_name, execute, *, record=None):
        """Commit successful execution before handing asynchronous delivery to the outbox."""
        async with (
            self._agent_task(agent_name),
            self.execution(agent_name).lock,
        ):
            async with self.management_lock:
                agent = await self.definition(agent_name)
                if not agent.enabled:
                    return {"agent_name": agent_name, "skipped": "disabled"}
                await self._mount_inbox(agent)
                execution = self.executions[agent_name]
                if execution.retry_items:
                    execution.recover_receipt((await self.inbox.snapshot(agent_name))["receipt"])
                batch = await self.inbox.claim(
                    agent_name,
                    limit=agent.max_batch_events,
                    item_ids=execution.retry_items,
                    batch_key=lambda item: item["id"] if item["source"] == CRON_EVENT_SOURCE else None,
                )
                if batch is None:
                    return {"agent_name": agent_name, "processed": 0}
                execution.retry_items = [item["id"] for item in batch["items"]]
                owns_worker = execution.worker is None
                if owns_worker:
                    execution.worker = asyncio.current_task()
            try:
                execution.started_at = self.now()
                self.activities[agent_name] = AgentActivity(
                    session_id="",
                    turn_id=batch["token"],
                    source="task",
                    task_id=agent_name,
                    started_at=execution.started_at,
                    phase="working",
                )
                reminder, session_id = await execute(agent, batch["items"])
                self.activities[agent_name] = self.activities[agent_name].model_copy(update={"session_id": session_id})

                async def commit():
                    async with self.outbox.handoff(agent_name, reminder, record) as ref:
                        await self.inbox.ack(
                            agent_name,
                            batch["token"],
                            metadata={"reminder_path": ref, "session_id": session_id},
                            retain=ref is not None,
                        )
                        execution.retry_items = None

                await finish_cleanup(commit())
                self.logger.info(
                    f"Proactive agent={agent_name} session={session_id} result=committed "
                    f"processed={len(batch['items'])} notifications={int(reminder is not None)}",
                )
                self.activities[agent_name] = self.activities[agent_name].model_copy(update={"phase": "completed"})
                self.changed.set()
                return {
                    "agent_name": agent_name,
                    "processed": len(batch["items"]),
                    "notifications": [reminder.model_dump()] if reminder is not None else [],
                }
            except BaseException as exc:
                phase = "interrupted" if isinstance(exc, asyncio.CancelledError) else "failed"
                self.activities[agent_name] = self.activities[agent_name].model_copy(update={"phase": phase})
                raise
            finally:
                try:
                    await finish_cleanup(self.inbox.nack(agent_name, batch["token"]))
                except ValueError:
                    pass
                except RuntimeError as exc:
                    self.logger.warning(f"Could not release failed inbox claim for {agent_name}: {exc}")
                finally:
                    if owns_worker:
                        execution.worker = None

    def _next_cron_time(self, expression: str, now: datetime) -> datetime | None:
        """Return no trigger for an exhausted schedule without hiding its retained task."""
        from croniter import CroniterBadDateError

        if not self.app_context.app_config.timezone:
            now = datetime.fromtimestamp(now.timestamp(), system_timezone())
        try:
            return next_cron_time(expression, now)
        except CroniterBadDateError:
            return None

    def _cron_schedule(self, expression: str, execution: _Execution, now: datetime):
        key = (expression, self.app_context.app_config.timezone)
        if execution.schedule is not None and execution.schedule[0] == key:
            return execution.schedule
        return key, self._next_cron_time(expression, now)

    async def _schedule_cron(self, agent):
        now = self.now()
        execution = self.execution(agent.agent_name)
        execution.schedule = self._cron_schedule(agent.cron, execution, now)
        key, scheduled = execution.schedule
        if scheduled is None or scheduled.timestamp() > now.timestamp():
            return
        await self._append(
            agent,
            [
                Event(
                    id=f"cron-{agent.agent_name}-{int(scheduled.timestamp())}",
                    event_ts=scheduled,
                    source=CRON_EVENT_SOURCE,
                    description=f"Scheduled trigger: {scheduled.isoformat()}",
                    metadata={"scheduled_at": scheduled.isoformat()},
                ),
            ],
            wait=False,
        )
        execution.schedule = key, self._next_cron_time(agent.cron, now)

    async def tick(self, run_job):
        """Check both consumers even when one scheduler fails."""
        results = await asyncio.gather(
            self._tick_tasks(run_job),
            self.notifications.tick(run_job),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def _tick_tasks(self, run_job):
        """Schedule independent serial tasks under the management lock."""
        async with self.management_lock:
            agents = await asyncio.to_thread(self.agents.load)
            for agent in agents.values():
                execution = self.execution(agent.agent_name)
                if not agent.enabled or agent.cron is None:
                    execution.schedule = None
                if not agent.enabled:
                    continue
                await self._mount_inbox(agent)
                if agent.cron is not None:
                    try:
                        await self._schedule_cron(agent)
                    except TimeoutError:
                        self.logger.warning(f"Cron input deferred for {agent.agent_name}: inbox capacity is full")
                if execution.worker is not None:
                    continue
                if execution.retry_at is not None and execution.retry_at > self.now():
                    continue
                snapshot = await self.inbox.snapshot(agent.agent_name)
                if not snapshot["pending_count"]:
                    continue
                receipt = snapshot["receipt"]
                if (
                    receipt is not None
                    and agent.min_trigger_interval_seconds
                    and (self.now() - receipt.consume_ts).total_seconds() < agent.min_trigger_interval_seconds
                ):
                    execution.recover_receipt(receipt)
                    item = await self.inbox.peek(agent.agent_name, item_ids=execution.retry_items)
                    if item is None or item["source"] != CRON_EVENT_SOURCE:
                        continue
                execution.worker = asyncio.create_task(
                    self._work(agent.agent_name, run_job),
                    name=f"agent:{agent.agent_name}",
                )

    async def _work(self, agent_name, run_job):
        execution = self.executions[agent_name]
        try:
            response = await run_job("run_proactive_agent", agent_name=agent_name)
            if not response.success:
                raise RuntimeError(str(response.answer))
            execution.error = execution.retry_at = None
            execution.retry_delay = 0
        except Exception as exc:
            execution.error = str(exc)[:1000]
            delay = min(
                max(300, self.retry_seconds),
                execution.retry_delay * 2 if execution.retry_delay else self.retry_seconds,
            )
            execution.retry_delay = delay
            execution.retry_at = self.now() + timedelta(seconds=delay)
            self.logger.exception(f"Proactive agent {agent_name} failed; retry in {delay:g}s: {exc}")
        finally:
            execution.worker = None
            self.changed.set()

    async def wait(self, stop, seconds=1):
        """Wake for external edits or new inputs while continuing timer checks."""
        tasks = [
            asyncio.create_task(stop.wait()),
            asyncio.create_task(self.changed.wait()),
        ]
        try:
            await asyncio.wait(tasks, timeout=seconds, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.changed.clear()
