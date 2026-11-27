"""File-native task discovery with explicit ownership of dynamic components."""

import asyncio

from .base import BaseAgentManager, TaskNotFoundError
from ._registry import RuntimeRegistry
from ..runtime.base import BaseAgentRuntime
from ..runtime.task import ProactiveAgentRuntime
from ...agent_wrapper import BaseAgentWrapper
from ....core.registry import R
from ....schema.event import Event
from ....schema.proactive_agent import TIME_EVENT_SOURCE, ProactiveRuntimeConfig, validate_runtime_names
from ....utils.async_utils import complete_in_thread
from ....utils.files.workspace_files import WorkspaceFiles


@R.register("local")
class LocalAgentManager(BaseAgentManager):
    """Borrow static runtimes and create, stop and close only dynamic children."""

    def __init__(
        self,
        runtimes=None,
        agent_wrapper="default",
        notification_target="Remy",
        allowed_run_jobs=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.agent_wrapper = (
            self.bind(agent_wrapper, BaseAgentWrapper, optional=False)
            if isinstance(agent_wrapper, str)
            else agent_wrapper
        )
        self.notification_target = (
            self.bind(notification_target, BaseAgentRuntime, optional=False)
            if isinstance(notification_target, str)
            else notification_target
        )
        self.static_names = validate_runtime_names(runtimes) if runtimes else []
        for index, name in enumerate(self.static_names):
            setattr(self, f"_static_{index}", self.bind(name, BaseAgentRuntime, optional=False))
        choices = set(ProactiveRuntimeConfig.model_fields["run_job"].annotation.__args__)
        self.allowed_run_jobs = frozenset(allowed_run_jobs if allowed_run_jobs is not None else choices)
        if not self.allowed_run_jobs or not self.allowed_run_jobs <= choices:
            raise ValueError("allowed_run_jobs must select supported domain execution Jobs")
        self.files = WorkspaceFiles(self.workspace_path)
        directory = self.app_context.app_config.reme_agent_dir if self.app_context else "reme_agents"
        self.registry = RuntimeRegistry(self.files, directory, self.logger)
        self.runtimes = {}
        self._dynamic = {}
        self._deleting = set()
        self.lock = asyncio.Lock()
        self.changed = asyncio.Event()
        self.closed = True

    def now(self):
        """Return application-local time using the configured timezone rules."""
        return self.notification_target.now()

    async def _start(self):
        self.runtimes = {name: getattr(self, f"_static_{index}") for index, name in enumerate(self.static_names)}
        if any(not isinstance(value, ProactiveAgentRuntime) for value in self.runtimes.values()):
            raise ValueError("AgentManager can only manage background Runtimes")
        self.closed = False
        try:
            await self.discover()
        except BaseException:
            await self._close()
            raise

    async def _close(self):
        self.closed = True
        self.changed.set()
        errors = []
        for runtime in reversed(list(self._dynamic.values())):
            try:
                await runtime.close()
            except Exception as exc:
                errors.append(exc)
        self._dynamic.clear()
        if errors:
            raise ExceptionGroup("Could not close dynamic Runtimes", errors)

    async def definition(self, agent_name, *, include_deleted=False):
        """Read the definition and reject changes to immutable runtime wiring."""
        if agent_name in self._deleting and not include_deleted:
            raise TaskNotFoundError(f"Task is being deleted: {agent_name}")
        return await asyncio.to_thread(self.registry.get, agent_name, include_deleted=include_deleted)

    async def _resolve_locked(self, definition):
        if self.closed:
            raise RuntimeError("AgentManager is closed")
        if definition.run_job not in self.allowed_run_jobs:
            raise ValueError("Task selects a disallowed execution Job")
        if definition.agent_name not in self.runtimes:
            config = definition.model_dump(exclude={"agent_name", "created_at", "deleted_at"})
            runtime = ProactiveAgentRuntime(
                name=definition.agent_name,
                agent_wrapper=self.agent_wrapper,
                notification_target=self.notification_target,
                app_context=self.app_context,
                **config,
            )
            try:
                await runtime.start()
            except BaseException:
                await runtime.close()
                raise
            self._dynamic[definition.agent_name] = runtime
            self.runtimes[definition.agent_name] = runtime
        runtime = self.runtimes[definition.agent_name]
        initial = runtime.initial_config
        if definition.run_job != initial.run_job or definition.agent_type != initial.agent_type:
            raise ValueError("Persisted task definition conflicts with immutable Runtime wiring")
        if getattr(runtime, "retiring", False):
            raise TaskNotFoundError(f"Task is being deleted: {definition.agent_name}")
        return runtime

    async def resolve(self, agent_name):
        async with self.lock:
            definition = await self.definition(agent_name)
            return await self._resolve_locked(definition)

    async def discover(self):
        """Resolve valid file-owned definitions while isolating damaged identities."""
        definitions = await asyncio.to_thread(self.registry.load)
        for name, definition in definitions.items():
            if name in self._deleting:
                continue
            try:
                async with self.lock:
                    await self._resolve_locked(definition)
            except (OSError, ValueError, RuntimeError) as exc:
                self.registry.issues[name] = str(exc)
        return definitions

    async def create(self, config):
        async with self.lock:
            previous = (await asyncio.to_thread(self.registry.load, include_deleted=True)).get(config.get("agent_name"))
            job = config.get("run_job", previous.run_job if previous else "run_memory_agent")
            if job not in self.allowed_run_jobs:
                raise ValueError("Task selects a disallowed execution Job")
            agent, restored = await complete_in_thread(self.registry.create, config, self.now())
            runtime = self.runtimes.get(agent.agent_name)
            if runtime is not None:
                runtime.retiring = False
            runtime = await self._resolve_locked(agent)
            self.changed.set()
        return {
            "agent_name": agent.agent_name,
            "restored": restored,
            "task_path": self.to_workspace_relative(runtime.task_path),
        }

    async def update(self, agent_name, settings):
        async with self.lock:
            await self.definition(agent_name)
            definition = await complete_in_thread(self.registry.update, agent_name, settings, self.now())
            runtime = await self._resolve_locked(definition)
            runtime.next_trigger(definition)
            self.changed.set()
        return await self.status(agent_name)

    async def delete(self, agent_name):
        async with self.lock:
            runtime = await self._resolve_locked(await self.definition(agent_name))
            self._deleting.add(agent_name)
            runtime.retiring = True
        try:
            await runtime.stop_work()
            async with self.lock:
                await complete_in_thread(self.registry.delete, agent_name, self.now())
                if agent_name in self._dynamic:
                    await self._dynamic.pop(agent_name).close()
                    self.runtimes.pop(agent_name, None)
                self.changed.set()
        except BaseException:
            runtime.retiring = False
            raise
        finally:
            self._deleting.discard(agent_name)
        return {"agent_name": agent_name, "deleted": True, "files_retained": True}

    async def append_events(self, agent_name, events):
        events = [Event.model_validate(value) for value in events]
        if any(value.source == TIME_EVENT_SOURCE for value in events):
            raise ValueError("time is a reserved framework source")
        async with self.lock:
            runtime = await self._resolve_locked(await self.definition(agent_name))
            identifiers = await runtime.append(events)
            self.changed.set()
        return {"agent_name": agent_name, "event_ids": identifiers}

    async def status(self, agent_name):
        """Read operational settings and state from the Runtime owners."""
        runtime = await self.resolve(agent_name)
        definition = await runtime.definition()
        snapshot = await runtime.inbox.snapshot()
        receipt = snapshot["receipt"]
        next_trigger = runtime.next_trigger(definition)
        return {
            "agent_name": agent_name,
            "config": definition.model_dump(mode="json"),
            "task_path": self.to_workspace_relative(runtime.task_path),
            "running": runtime.worker is not None or runtime.execution_lock.locked(),
            "pending_events": snapshot["pending_count"],
            "in_flight_events": snapshot["in_flight_count"],
            "last_started_at": runtime.last_started_at.isoformat() if runtime.last_started_at else None,
            "last_completed_at": receipt.consume_ts.isoformat() if receipt else None,
            "next_trigger_at": next_trigger.isoformat() if next_trigger else None,
            "error": runtime.error or runtime.delivery_error,
        }
