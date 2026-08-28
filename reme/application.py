"""Main application entry point."""

import asyncio
import heapq
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, AsyncGenerator, TypeVar

from . import __version__
from .components import ApplicationContext, BaseComponent
from .components.job import BackgroundJob, BaseJob, CronJob, StreamJob
from .components.service import BaseService
from .enumeration import ComponentEnum, ComponentType, component_type_name
from .plugin import resolve_plugin_runtime
from .schema import ComponentConfig, Response, StreamChunk
from .utils import execute_stream_task, print_logo, get_logger

T = TypeVar("T", bound=BaseComponent)
_NodeKey = tuple[str, str]


class Application(BaseComponent):
    """Wires components from config and runs jobs against them."""

    def __init__(self, **kwargs) -> None:
        runtime = resolve_plugin_runtime(kwargs)
        self.context = ApplicationContext(registry=runtime.registry, **runtime.config)
        self._started_components: list[BaseComponent] = []
        self._component_mutation_lock = asyncio.Lock()

        self._setup_workspace_directories()
        logger = get_logger(
            log_to_console=self.config.log_to_console,
            log_to_file=self.config.log_to_file,
            force_init=True,
        )
        super().__init__()
        self._init_service()

        if self.config.enable_logo:
            print_logo(self.config, self.context.service)
        logger.info(f"Initializing {self.config.app_name} Application v{__version__}")
        self._init_components()
        self._init_jobs()

    @property
    def config(self):
        """Typed view onto the application config held by the context."""
        return self.context.app_config

    # ----- Wiring (called once during __init__) --------------------------

    def _setup_workspace_directories(self) -> None:
        """Ensure the workspace root and configured subdirectories exist on disk."""
        cfg = self.config
        workspace_path = Path(cfg.workspace_dir).absolute()
        workspace_path.mkdir(parents=True, exist_ok=True)
        for subdir in [
            cfg.metadata_dir,
            cfg.session_dir,
            cfg.mem_session_dir,
            cfg.resource_dir,
            cfg.daily_dir,
            cfg.digest_dir,
        ]:
            if subdir:
                (workspace_path / subdir).mkdir(parents=True, exist_ok=True)

    def _init_service(self) -> None:
        """Instantiate the single service backend declared in config.service."""
        self.context.service = self._instantiate(
            ComponentEnum.SERVICE,
            self.config.service,
            label="Service",
            expected_type=BaseService,
        )

    def _init_components(self) -> None:
        """Instantiate every component declared under config.components."""
        for ctype, group in self.config.components.items():
            self.context.components[ctype] = {}
            for name, cfg in group.items():
                self.context.components[ctype][name] = self._instantiate(
                    ctype,
                    cfg,
                    label=f"Component '{name}'",
                    expected_type=BaseComponent,
                    name=name,
                )

    def _init_jobs(self) -> None:
        """Instantiate every job declared under config.jobs."""
        for name, cfg in self.config.jobs.items():
            self.context.jobs[name] = self._instantiate(
                ComponentEnum.JOB,
                cfg,
                label=f"Job '{name}'",
                expected_type=BaseJob,
                name=name,
            )

    def _instantiate(
        self,
        ctype: ComponentType,
        cfg: ComponentConfig,
        *,
        label: str,
        expected_type: type[T],
        name: str | None = None,
    ) -> T:
        """Resolve cfg.backend through the registry and construct the instance.

        `label` is the human-readable identifier used only in error messages.
        `expected_type` narrows the return type and guards against a backend
        registered under the wrong component type.
        `name` is forwarded to the constructor for named components/jobs;
        leave it None for the service, which is keyed solely by type.
        """
        if not cfg.backend:
            raise ValueError(f"{label} is missing the required 'backend' field")
        backend_cls = self.context.registry.get(ctype, cfg.backend)
        if backend_cls is None:
            raise ValueError(f"Unregistered backend '{cfg.backend}' for {label}")

        params = cfg.model_dump()
        params["app_context"] = self.context
        if name is not None:
            params.setdefault("name", name)
        instance = backend_cls(**params)
        if not isinstance(instance, expected_type):
            got, want = type(instance).__name__, expected_type.__name__
            raise TypeError(f"{label} backend '{cfg.backend}' produced {got}, expected {want} subclass")
        return instance

    # ----- Dependency ordering ------------------------------------------

    def _topological_order(
        self,
        replacement: tuple[_NodeKey, BaseComponent] | None = None,
    ) -> list[BaseComponent]:
        """Return components in dependency order via Kahn's algorithm; raise on missing dep or cycle."""
        nodes: dict[_NodeKey, BaseComponent] = {
            (ctype, name): comp for ctype, group in self.context.components.items() for name, comp in group.items()
        }
        if replacement is not None:
            nodes[replacement[0]] = replacement[1]
        in_degree, dependents = self._build_dependency_graph(nodes)

        ready = [k for k, d in in_degree.items() if d == 0]
        heapq.heapify(ready)
        ordered: list[BaseComponent] = []
        while ready:
            key = heapq.heappop(ready)
            ordered.append(nodes[key])
            for downstream in dependents[key]:
                in_degree[downstream] -= 1
                if in_degree[downstream] == 0:
                    heapq.heappush(ready, downstream)

        if len(ordered) != len(nodes):
            unresolved = [f"{k[0]}:{k[1]}" for k, d in in_degree.items() if d > 0]
            raise ValueError(f"Circular dependency detected among: {unresolved}")
        return ordered

    @staticmethod
    def _build_dependency_graph(
        nodes: dict[_NodeKey, BaseComponent],
    ) -> tuple[dict[_NodeKey, int], dict[_NodeKey, list[_NodeKey]]]:
        """Compute in-degree and adjacency lists; raise if a required dep is missing."""
        in_degree: dict[_NodeKey, int] = dict.fromkeys(nodes, 0)
        dependents: dict[_NodeKey, list[_NodeKey]] = {k: [] for k in nodes}
        for key, comp in nodes.items():
            for dep in comp.dependencies:
                dep_key = (dep.ctype, dep.name)
                if dep_key in nodes:
                    dependents[dep_key].append(key)
                    in_degree[key] += 1
                elif not dep.optional:
                    raise ValueError(
                        f"Component {key[0]}:{key[1]} depends on unregistered {dep.ctype}:{dep.name}",
                    )
        return in_degree, dependents

    # ----- Lifecycle -----------------------------------------------------

    async def _start(self) -> None:
        """Start components, then jobs as base > stream > background > cron."""
        async with self._component_mutation_lock:
            pool_size = self.config.thread_pool_max_workers
            if pool_size > 0:
                self.context.thread_pool = ThreadPoolExecutor(max_workers=pool_size)
                self.logger.info(f"Thread pool created with max_workers={pool_size}")
            try:
                components = self._topological_order()
                jobs = list(self.context.jobs.values())
                base_jobs = [j for j in jobs if not isinstance(j, (StreamJob, BackgroundJob))]
                stream_jobs = [j for j in jobs if isinstance(j, StreamJob)]
                background_jobs = [j for j in jobs if isinstance(j, BackgroundJob) and not isinstance(j, CronJob)]
                cron_jobs = [j for j in jobs if isinstance(j, CronJob)]
                for c in components + base_jobs + stream_jobs + background_jobs + cron_jobs:
                    await self._start_one(c)
            except Exception:
                await self._close_started_components()
                raise

    async def _start_one(self, c: BaseComponent) -> None:
        """Start one component and record it for ordered shutdown."""
        try:
            if isinstance(c, BackgroundJob):
                self.logger.info(f"Starting background job: {c.name}")
            await c.start()
            self._started_components.append(c)
        except Exception as e:
            self.logger.exception(f"Failed to start {component_type_name(c.component_type)}:{c.name}: {e}")
            raise

    async def _close(self) -> None:
        """Close in reverse start order so every peer outlives its dependents."""
        async with self._component_mutation_lock:
            await self._close_started_components()

    async def _close_started_components(self) -> None:
        """Close resources while the caller serializes component mutations."""
        for c in reversed(self._started_components):
            try:
                await c.close()
            except Exception as e:
                self.logger.exception(f"Failed to close {component_type_name(c.component_type)}:{c.name}: {e}")
        self._started_components.clear()
        if self.context.thread_pool is not None:
            self.context.thread_pool.shutdown(wait=True)
            self.context.thread_pool = None

    async def update_component(self, component_enum: ComponentType, name: str, /, **kwargs) -> BaseComponent:
        """Update an existing component by type/name; never creates missing components."""
        async with self._component_mutation_lock:
            component_type = component_type_name(component_enum)
            group = self.context.components.get(component_type)
            if not group or name not in group:
                raise KeyError(f"Component '{name}' not found in {component_type}")

            component = group[name]
            for key in kwargs:
                if not hasattr(component, key):
                    raise AttributeError(f"Component {component_type}:{name} has no attribute '{key}'")
            for key, value in kwargs.items():
                setattr(component, key, value)
            return component

    async def replace_component(
        self,
        component_enum: ComponentType,
        name: str,
        /,
        *,
        config: ComponentConfig | Mapping[str, Any],
        runtime_updates: Mapping[str, Any] | None = None,
    ) -> BaseComponent:
        """Atomically replace an existing component and its bind() references.

        ``config`` is the complete declarative configuration for the new
        component. ``runtime_updates`` injects non-serializable live objects,
        such as an already verified model, before the replacement is started.

        The new component is constructed, validated, and started while the old
        component remains visible. The context, dependent bindings, persisted
        application config, and shutdown order are then switched without an
        await boundary. A start failure leaves the old generation untouched.
        Hosts must still quiesce calls that may retain component references
        across this operation.
        """
        async with self._component_mutation_lock:
            component_type = component_type_name(component_enum)
            group = self.context.components.get(component_type)
            config_group = self.config.components.get(component_type)
            if not group or name not in group or config_group is None:
                raise KeyError(f"Component '{name}' not found in {component_type}")

            old_component = group[name]
            replacement_config = (
                config.model_copy(deep=True)
                if isinstance(config, ComponentConfig)
                else ComponentConfig.model_validate(dict(config))
            )
            replacement = self._instantiate(
                component_type,
                replacement_config,
                label=f"Component '{name}'",
                expected_type=BaseComponent,
                name=name,
            )
            for key, value in (runtime_updates or {}).items():
                if not hasattr(replacement, key):
                    raise AttributeError(
                        f"Replacement {component_type}:{name} has no attribute '{key}'",
                    )
                setattr(replacement, key, value)

            node_key = (component_type, name)
            replacement_order = Application._topological_order(self, replacement=(node_key, replacement))
            was_started = old_component.is_started
            if was_started:
                await replacement.start()

            consumers: list[tuple[BaseComponent, str]] = []
            candidates: list[BaseComponent] = [
                component for components in self.context.components.values() for component in components.values()
            ]
            candidates.extend(self.context.jobs.values())
            if self.context.service is not None:
                candidates.append(self.context.service)
            for consumer in candidates:
                if consumer is old_component:
                    continue
                for attr, dependency in consumer.dependency_bindings.items():
                    if (
                        dependency.ctype == component_type
                        and dependency.name == name
                        and consumer.__dict__.get(attr) is old_component
                    ):
                        consumers.append((consumer, attr))

            # Commit the new generation synchronously so observers cannot see
            # a context with only some dependency references updated.
            group[name] = replacement
            config_group[name] = replacement_config
            for consumer, attr in consumers:
                consumer.__dict__[attr] = replacement
            if was_started:
                started = set(self._started_components)
                started.discard(old_component)
                started.add(replacement)
                component_set = {
                    component for components in self.context.components.values() for component in components.values()
                }
                non_components = [
                    component
                    for component in self._started_components
                    if component is not old_component and component not in component_set
                ]
                self._started_components = [
                    component for component in replacement_order if component in started
                ] + non_components

            if was_started:
                try:
                    await old_component.close()
                except Exception as exc:  # The committed replacement remains authoritative.
                    self.logger.exception(
                        f"Failed to close replaced component {component_type}:{name}: {exc}",
                    )
            return replacement

    # ----- Job execution -------------------------------------------------

    async def run_job(self, name: str, /, **kwargs) -> Response:
        """Execute a registered job by name and return its final Response."""
        if name not in self.context.jobs:
            raise KeyError(f"Job '{name}' not found")
        return await self.context.jobs[name](**kwargs)

    async def run_stream_job(self, name: str, /, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """Execute a streaming job, yielding chunks as they are produced."""
        if name not in self.context.jobs:
            raise KeyError(f"Job '{name}' not found")
        stream_queue: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(self.context.jobs[name](stream_queue=stream_queue, **kwargs))
        async for chunk in execute_stream_task(
            stream_queue=stream_queue,
            task=task,
            task_name=name,
            output_format="chunk",
        ):
            assert isinstance(chunk, StreamChunk)
            yield chunk

    def run_app(self):
        """Serve the application through the configured service backend."""
        assert isinstance(self.context.service, BaseService)
        self.context.service.run_app(app=self)
