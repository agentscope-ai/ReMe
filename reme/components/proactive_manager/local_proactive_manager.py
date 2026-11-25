"""Application-owned agent coordination with file-native recovery."""

import asyncio
from contextlib import aclosing, asynccontextmanager
from datetime import datetime
import math

from watchfiles import awatch

from ...constants import CHAT_AGENT_NAME
from .base_proactive_manager import BaseProactiveManager, TimeTriggeredTaskDeliveryError
from ...utils.files.workspace_files import WorkspaceFiles
from ._internal.notifications import ChatNotifications
from ._internal.outbox import ReminderOutbox
from ._internal.registry import RuntimeRegistry
from ._internal.sessions import AgentSessions
from ._internal.live_turn import LiveTurns
from ..agent_wrapper import BaseAgentWrapper
from ..agent_inbox import BaseAgentInbox
from ..proactive_runtime import ProactiveRuntime
from ...core.registry import R
from ...schema.proactive_runtime import (
    ChatAgentConfig,
    NotificationConfig,
)
from ...schema.application_config import ApplicationConfig
from ...schema.event import Event
from ...schema.proactive_agent import CRON_EVENT_SOURCE, CreateProactiveRuntimeConfig, ProactiveRuntimeDefinition
from ...utils.async_utils import complete_in_thread, finish_cleanup
from ...utils.timezone_utils import resolve_timezone


@R.register("local")
class LocalProactiveManager(BaseProactiveManager):
    """Discover workspace runtimes and coordinate shared services."""

    def __init__(
        self,
        agent_wrapper="default",
        inbox="default",
        notifications=None,
        retry_seconds=5,
        runtimes=None,
        chat=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if type(retry_seconds) not in (int, float) or not math.isfinite(retry_seconds) or retry_seconds < 0:
            raise ValueError("retry_seconds must be a finite nonnegative number")
        if runtimes is not None and not isinstance(runtimes, list):
            raise ValueError("runtimes must be a list of initial Runtime configurations")
        self.initial_runtimes = [CreateProactiveRuntimeConfig.model_validate(value) for value in runtimes or []]
        if len({runtime.agent_name.casefold() for runtime in self.initial_runtimes}) != len(self.initial_runtimes):
            raise ValueError("initial Runtime names must be unique, including case")
        self.initial_chat = ChatAgentConfig.model_validate(chat if chat is not None else {})
        self.agent_wrapper = self.bind(agent_wrapper, BaseAgentWrapper, optional=False)
        self.inbox = self.bind(inbox, BaseAgentInbox, optional=False)
        self.files = WorkspaceFiles(self.workspace_path)
        config = self.app_context.app_config if self.app_context else ApplicationConfig()
        self.registry = RuntimeRegistry(self.files, config.reme_agent_dir, self.logger)
        self.sessions = AgentSessions(self.registry)
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
        self.runtimes: dict[str, ProactiveRuntime] = {}
        self.changed = asyncio.Event()
        self.management_lock = asyncio.Lock()
        self.watch_stop = asyncio.Event()
        self.watcher = None
        self.closed = True

    @property
    def workers(self):
        """Read active workers from their owning execution states."""
        return {name: state.worker for name, state in self.runtimes.items() if state.worker is not None}

    def _runtime(self, definition: ProactiveRuntimeDefinition) -> ProactiveRuntime:
        """Own one live instance per persisted definition."""
        name = definition.agent_name
        if name not in self.runtimes:
            self.runtimes[name] = ProactiveRuntime(self, name)
        return self.runtimes[name]

    async def _find_runtime(self, agent_name: str) -> ProactiveRuntime:
        runtime = self.runtimes.get(agent_name)
        if runtime is not None:
            return runtime
        async with self.management_lock:
            return self._runtime(await self.definition(agent_name))

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
        return datetime.now(resolve_timezone(timezone))

    async def _start(self):
        if self.app_context is None:
            raise RuntimeError("LocalProactiveManager requires an ApplicationContext")
        await complete_in_thread(
            self.files.write,
            self.registry.chat_memory_path(),
            self.initial_chat.initial_memory,
            create_only=True,
        )
        existing = await asyncio.to_thread(self.registry.configured_names)
        for initial in self.initial_runtimes:
            if initial.agent_name.casefold() not in existing:
                await self.create(initial)
        for agent in (await asyncio.to_thread(self.registry.load)).values():
            await self._runtime(agent).mount(agent)
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
        await asyncio.gather(*(runtime.stop() for runtime in self.runtimes.values()))
        tasks = self.chat_tasks.copy()
        if self.watcher is not None:
            tasks.add(self.watcher)
        for task in tasks:
            task.cancel()
        if tasks:
            await finish_cleanup(asyncio.gather(*tasks, return_exceptions=True))
        self.activities.clear()
        self.chat_tasks.clear()
        self.runtimes.clear()
        self.watcher = None

    async def _watch(self):
        """One watcher plus timeout rescans covers edits, replacement and missed notifications."""
        async with aclosing(
            awatch(
                self.registry.directory,
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

    async def definition(self, agent_name):
        """Read current user-owned configuration rather than a stale runtime copy."""
        return await asyncio.to_thread(self.registry.get, agent_name)

    async def create(self, config):
        """Persist a task and its inbox directory before returning success."""
        async with self.management_lock:
            agent, restored = await complete_in_thread(
                self.registry.create,
                config,
                self.now(),
            )
            status = await self._status(agent)
            self.changed.set()
            return {**status, "restored": restored}

    async def update(self, agent_name, config):
        """Apply settings and invalidate the timer when its schedule changes."""
        async with self.management_lock:
            agent = await complete_in_thread(self.registry.update, agent_name, config, self.now())
            self._runtime(agent).configure(agent)
            self.changed.set()
            return await self._status(agent)

    async def delete(self, agent_name):
        """Stop owned work while retaining a restorable identity and all source files."""
        async with self.management_lock:
            runtime = self._runtime(await self.definition(agent_name))
            await runtime.stop()
            session = await self.sessions.current(agent_name)
            if session is not None:
                await finish_cleanup(self.agent_wrapper.release_session(session.session_id))
            agent = await complete_in_thread(self.registry.delete, agent_name, self.now())
            self.runtimes.pop(agent_name)
            self.activities.pop(agent_name, None)
            self.changed.set()
            return {"agent_name": agent.agent_name, "deleted": True, "files_retained": True}

    async def append_events(self, agent_name: str, events: list[Event]) -> dict:
        """Accept external observations for an event-triggered task."""
        if any(event.source == CRON_EVENT_SOURCE for event in events):
            raise ValueError("cron is a reserved framework source")
        async with self.management_lock:
            if self.closed:
                raise RuntimeError("Proactive Manager is closed")
            agent = await self.definition(agent_name)
            if agent.cron is not None:
                raise TimeTriggeredTaskDeliveryError(f"Time-triggered task cannot receive observations: {agent_name}")
            runtime = self._runtime(agent)
            await runtime.mount(agent)
            identifiers = await runtime.append(events)
        return {"agent_name": agent_name, "event_ids": identifiers}

    async def status(self, agent_name):
        """Inspect without acquiring a model-generation lock."""
        async with self.management_lock:
            return await self._status(await self.definition(agent_name))

    async def _status(self, agent, *, allow_unavailable=False):
        """Read execution state and retain per-task errors in the shared roster."""
        agent_name = agent.agent_name
        runtime = self._runtime(agent)
        task_path = None
        snapshot = {}
        error = runtime.scheduler_error or runtime.error
        try:
            task_path = self.registry.task_path(agent)
            await runtime.mount(agent)
            snapshot = await self.inbox.snapshot(agent_name)
        except ValueError as exc:
            if not allow_unavailable:
                raise
            error = str(exc)
        receipt = snapshot.get("receipt")
        next_trigger = runtime.next_trigger(agent)
        if not error and agent.enabled and agent.cron is not None and next_trigger is None:
            error = f"Cron schedule has no future trigger: {agent.cron}"
        return {
            "agent_name": agent_name,
            "config": agent.model_dump(mode="json", exclude={"created_at", "deleted_at"}),
            "created_at": agent.created_at.isoformat(),
            "task_path": task_path.relative_to(self.workspace_path).as_posix() if task_path is not None else None,
            "running": runtime.worker is not None,
            "pending_events": snapshot.get("pending_count", 0),
            "in_flight_events": snapshot.get("in_flight_count", 0),
            "last_started_at": (runtime.started_at.isoformat() if runtime.started_at else None),
            "last_completed_at": receipt.consume_ts.isoformat() if receipt else None,
            "next_trigger_at": next_trigger.isoformat() if next_trigger is not None else None,
            "error": error,
        }

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

    async def task_statuses(self):
        """Snapshot registered task state without repeatedly scanning their definitions."""
        async with self.management_lock:
            definitions = await asyncio.to_thread(self.registry.load)
            return {name: await self._status(agent, allow_unavailable=True) for name, agent in definitions.items()}

    async def view_agent(self, agent_name, *, status=None):
        """Project settings and execution facts without reading message history."""
        from ._internal.views import AgentViews

        return (await AgentViews(self).view(agent_name, task_status=status)).model_dump(mode="json")

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
        async with self.management_lock:
            definition = await self.definition(agent_name)
            await self._runtime(definition).mount(definition)
            page = await self.inbox.read(agent_name, **kwargs)
        page["items"] = [{"event": item.pop("item"), **item} for item in page["items"]]
        return page

    async def list_chat_sessions(self, offset=0, limit=10):
        """Listing is side-effect-free, including before the first Chat turn."""
        return await self.sessions.list(CHAT_AGENT_NAME, offset, limit)

    @asynccontextmanager
    async def query_session(self, agent_name, options):
        """Own a temporary fork of the latest task session, including during execution."""
        runtime = await self._find_runtime(agent_name)
        async with runtime.track():
            async with self.management_lock:
                await runtime.definition()
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

    async def run_round(self, agent_name, execute, *, record=None):
        """Dispatch one input batch to its named runtime."""
        runtime = await self._find_runtime(agent_name)
        return await runtime.run(execute, record=record)

    async def run(self, stop, run_job):
        """Drive the scheduler under the owning BackgroundJob's lifecycle."""
        try:
            while not stop.is_set():
                await self.tick(run_job)
                await self.wait(stop)
        finally:
            await self.stop_workers()

    async def tick(self, run_job):
        """Check both consumers even when one scheduler fails."""
        results = await asyncio.gather(
            self._tick_runtimes(run_job),
            self.notifications.tick(run_job),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def _tick_runtimes(self, run_job):
        """Reconcile workspace definitions and dispatch eligible runtimes."""
        async with self.management_lock:
            definitions = await asyncio.to_thread(self.registry.load)
            for name in self.runtimes.keys() - definitions.keys():
                await self.runtimes.pop(name).stop()
                self.activities.pop(name, None)
            for definition in definitions.values():
                runtime = self._runtime(definition)
                try:
                    await runtime.tick(definition, run_job)
                except ValueError as exc:
                    error = str(exc)
                    if runtime.scheduler_error != error:
                        self.logger.error(f"Could not schedule Proactive Runtime {definition.agent_name}: {error}")
                    runtime.scheduler_error = error
                else:
                    runtime.scheduler_error = None

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
