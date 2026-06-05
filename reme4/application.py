"""Main application entry point."""

import asyncio
import heapq
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator, TypeVar

from .components import BaseComponent, ApplicationContext
from .components.job import BaseJob
from .components.service import BaseService
from .enumeration import ComponentEnum
from .schema import ComponentConfig, Response, StreamChunk
from .utils import execute_stream_task, print_logo, get_logger

T = TypeVar("T", bound=BaseComponent)
_NodeKey = tuple[ComponentEnum, str]


class Application(BaseComponent):
    """Wires components from config and runs jobs against them."""

    def __init__(self, **kwargs) -> None:
        self.context = ApplicationContext(**kwargs)
        self._started_components: list[BaseComponent] = []

        self._setup_vault_directories()

        if self.config.enable_logo:
            print_logo(self.config)
        logger = get_logger(log_to_console=self.config.log_to_console, log_to_file=self.config.log_to_file)
        logger.info(f"Initializing {self.config.app_name} Application")
        super().__init__()

        self._init_service()
        self._init_components()
        self._init_jobs()

    @property
    def config(self):
        """Typed view onto the application config held by the context."""
        return self.context.app_config

    # ----- Wiring (called once during __init__) --------------------------

    def _setup_vault_directories(self) -> None:
        """Ensure the vault root and configured subdirectories exist on disk."""
        cfg = self.config
        vault_path = Path(cfg.vault_dir).absolute()
        vault_path.mkdir(parents=True, exist_ok=True)
        for subdir in [cfg.metadata_dir, cfg.resource_dir, cfg.daily_dir, cfg.digest_dir]:
            if subdir:
                (vault_path / subdir).mkdir(parents=True, exist_ok=True)

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
        ctype: ComponentEnum,
        cfg: ComponentConfig,
        *,
        label: str,
        expected_type: type[T],
        name: str | None = None,
    ) -> T:
        """Resolve cfg.backend through the registry and construct the instance.

        `label` is the human-readable identifier used only in error messages.
        `expected_type` narrows the return type and guards against a backend
        registered under the wrong ComponentEnum.
        `name` is forwarded to the constructor for named components/jobs;
        leave it None for the service, which is keyed solely by type.
        """
        # Lazy import: the registry self-populates as component modules load.
        from .components import R

        if not cfg.backend:
            raise ValueError(f"{label} is missing the required 'backend' field")
        backend_cls = R.get(ctype, cfg.backend)
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

    def _topological_order(self) -> list[BaseComponent]:
        """Return components in dependency order via Kahn's algorithm; raise on missing dep or cycle."""
        nodes: dict[_NodeKey, BaseComponent] = {
            (ctype, name): comp for ctype, group in self.context.components.items() for name, comp in group.items()
        }
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
            unresolved = [f"{k[0].value}:{k[1]}" for k, d in in_degree.items() if d > 0]
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
                        f"Component {key[0].value}:{key[1]} depends on unregistered {dep.ctype.value}:{dep.name}",
                    )
        return in_degree, dependents

    # ----- Lifecycle -----------------------------------------------------

    async def _start(self) -> None:
        """Start components in dependency order, then jobs (background last)."""
        pool_size = self.config.thread_pool_max_workers
        if pool_size > 0:
            self.context.thread_pool = ThreadPoolExecutor(max_workers=pool_size)
            self.logger.info(f"Thread pool created with max_workers={pool_size}")
        components = self._topological_order()
        jobs = list(self.context.jobs.values())
        # Background jobs come last so they observe a fully wired system.
        foreground = [j for j in jobs if j.backend != "background"]
        background = [j for j in jobs if j.backend == "background"]
        for c in components + foreground + background:
            await self._start_one(c)

    async def _start_one(self, c: BaseComponent) -> None:
        """Start one component and record it for ordered shutdown; log and swallow failures."""
        try:
            if c.backend == "background":
                self.logger.info(f"Starting background job: {c.name}")
            await c.start()
            self._started_components.append(c)
        except Exception as e:
            self.logger.exception(f"Failed to start {c.component_type.value}:{c.name}: {e}")

    async def _close(self) -> None:
        """Close in reverse start order so every peer outlives its dependents."""
        for c in reversed(self._started_components):
            try:
                await c.close()
            except Exception as e:
                self.logger.exception(f"Failed to close {c.component_type.value}:{c.name}: {e}")
        self._started_components.clear()
        if self.context.thread_pool is not None:
            self.context.thread_pool.shutdown(wait=True)
            self.context.thread_pool = None

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
