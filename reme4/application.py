"""Main application entry point."""

import asyncio
import heapq
from pathlib import Path
from typing import AsyncGenerator

from .components import BaseComponent, ApplicationContext
from .enumeration import ComponentEnum
from .schema import Response, StreamChunk
from .utils import execute_stream_task, print_logo, get_logger


class Application(BaseComponent):
    """Main application: initializes components, resolves dependencies, runs jobs."""

    def __init__(self, **kwargs) -> None:
        self.context = ApplicationContext(**kwargs)
        self._started_components: list[BaseComponent] = []

        working_path = Path(self.config.working_dir).absolute()
        working_path.mkdir(parents=True, exist_ok=True)
        (working_path / self.config.metadata_dir).mkdir(parents=True, exist_ok=True)
        (working_path / self.config.daily_dir).mkdir(parents=True, exist_ok=True)
        (working_path / self.config.digest_dir).mkdir(parents=True, exist_ok=True)

        if self.config.enable_logo:
            print_logo(self.config)

        logger = get_logger(log_to_console=self.config.log_to_console, log_to_file=self.config.log_to_file)
        logger.info(f"Initializing {self.config.app_name} Application")
        super().__init__()

        from .components import R

        # Service
        service_config = self.config.service
        if not service_config.backend:
            raise ValueError("Service configuration is missing the required 'backend' field")
        service_cls = R.get(ComponentEnum.SERVICE, service_config.backend)
        if not service_cls:
            raise ValueError(f"Unregistered service backend '{service_config.backend}'")
        params = service_config.model_dump()
        params["app_context"] = self.context
        self.context.service = service_cls(**params)

        # Components
        for component_type, component_configs in self.config.components.items():
            self.context.components[component_type] = {}
            for name, config in component_configs.items():
                if not config.backend:
                    raise ValueError(f"Component '{name}' is missing the required 'backend' field")
                backend_cls = R.get(component_type, config.backend)
                if not backend_cls:
                    raise ValueError(f"Unregistered backend '{config.backend}' for component '{name}'")
                params = config.model_dump()
                params.setdefault("name", name)
                params["app_context"] = self.context
                self.context.components[component_type][name] = backend_cls(**params)

        # Jobs
        for job_config in self.config.jobs:
            if not job_config.backend:
                raise ValueError(f"Job '{job_config.name}' is missing the required 'backend' field")
            job_cls = R.get(ComponentEnum.JOB, job_config.backend)
            if not job_cls:
                raise ValueError(f"Unregistered backend '{job_config.backend}' for job '{job_config.name}'")
            params = job_config.model_dump()
            params["app_context"] = self.context
            self.context.jobs[job_config.name] = job_cls(**params)

    @property
    def config(self):
        """Application configuration."""
        return self.context.app_config

    def _topological_order(self) -> list[BaseComponent]:
        """Kahn's algorithm. Raises on missing required dep or cycle."""
        nodes: dict[tuple[ComponentEnum, str], BaseComponent] = {
            (ctype, name): comp for ctype, group in self.context.components.items() for name, comp in group.items()
        }

        in_degree: dict[tuple[ComponentEnum, str], int] = dict.fromkeys(nodes, 0)
        dependents: dict[tuple[ComponentEnum, str], list[tuple[ComponentEnum, str]]] = {k: [] for k in nodes}
        for key, comp in nodes.items():
            for dep in comp.dependencies:
                dep_key = (dep.ctype, dep.name)
                if dep_key in nodes:
                    dependents[dep_key].append(key)
                    in_degree[key] += 1
                elif not dep.optional:
                    raise ValueError(
                        f"Component {key[0].value}:{key[1]} depends on {dep.ctype.value}:{dep.name}, not registered",
                    )

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

    async def _start(self) -> None:
        """Start components, then regular jobs, then background jobs; record order for reverse close."""
        components = self._topological_order()
        jobs = list(self.context.jobs.values())
        sequence = (
            components + [j for j in jobs if j.backend != "background"] + [j for j in jobs if j.backend == "background"]
        )

        for c in sequence:
            try:
                if c.backend == "background":
                    self.logger.info(f"Starting background job: {c.name}")
                await c.start()
                self._started_components.append(c)
            except Exception as e:
                self.logger.exception(f"Failed to start {c.component_type.value}:{c.name}: {e}")

    async def _close(self) -> None:
        """Close in reverse order of successful start."""
        for c in reversed(self._started_components):
            try:
                await c.close()
            except Exception as e:
                self.logger.exception(f"Failed to close {c.component_type.value}:{c.name}: {e}")
        self._started_components.clear()

    async def run_job(self, name: str, /, **kwargs) -> Response:
        """Execute a registered job by name."""
        if name not in self.context.jobs:
            raise KeyError(f"Job '{name}' not found")
        return await self.context.jobs[name](**kwargs)

    async def run_stream_job(self, name: str, /, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """Execute a streaming job and yield chunks."""
        if name not in self.context.jobs:
            raise KeyError(f"Job '{name}' not found")
        job = self.context.jobs[name]
        stream_queue = asyncio.Queue()
        task = asyncio.create_task(job(stream_queue=stream_queue, **kwargs))
        async for chunk in execute_stream_task(
            stream_queue=stream_queue,
            task=task,
            task_name=name,
            output_format="chunk",
        ):
            assert isinstance(chunk, StreamChunk)
            yield chunk

    def run_app(self):
        """Start the service and serve the application."""
        from .components.service import BaseService

        assert isinstance(self.context.service, BaseService)
        self.context.service.run_app(app=self)
