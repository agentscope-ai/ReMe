"""Application module for managing the main application lifecycle."""

import asyncio
from pathlib import Path
from typing import AsyncGenerator

from .component import BaseComponent, ApplicationContext
from .enumeration import ComponentEnum
from .schema import Response, StreamChunk
from .utils import execute_stream_task, print_logo, get_logger


class Application(BaseComponent):
    """Application component for managing the main application."""

    def __init__(self, **kwargs) -> None:
        self.context = ApplicationContext(**kwargs)

        working_path = Path(self.config.working_dir).absolute()
        working_path.mkdir(parents=True, exist_ok=True)

        if self.config.enable_logo:
            print_logo(self.config)

        logger = get_logger(
            log_to_console=self.config.log_to_console,
            log_to_file=self.config.log_to_file,
            force_init=True,
        )
        logger.info(f"Initializing {self.config.app_name} Application")

        super().__init__()

        from .component import R

        # Initialize the service
        service_config = self.config.service
        if not service_config.backend:
            raise ValueError("Service configuration is missing the required 'backend' field")
        service_cls = R.get(ComponentEnum.SERVICE, service_config.backend)
        if not service_cls:
            raise ValueError(
                f"Service references an unregistered backend '{service_config.backend}' "
                f"of type '{ComponentEnum.SERVICE}'",
            )
        params = service_config.model_dump()
        params["app_context"] = self.context
        self.context.service = service_cls(**params)

        # Initialize all components grouped by type and name
        for component_type, component_configs in self.config.components.items():
            self.context.components[component_type] = {}
            for name, config in component_configs.items():
                if not config.backend:
                    raise ValueError(f"Component '{name}' is missing the required 'backend' field")
                backend_cls = R.get(component_type, config.backend)
                if not backend_cls:
                    raise ValueError(
                        f"Component '{name}' references an unregistered backend '{config.backend}' "
                        f"of type '{component_type}'",
                    )
                params = config.model_dump()
                params.setdefault("name", name)
                params["app_context"] = self.context
                self.context.components[component_type][name] = backend_cls(**params)

        # Initialize all jobs
        for job_config in self.config.jobs:
            if not job_config.backend:
                raise ValueError(f"Job '{job_config.name}' is missing the required 'backend' field")

            job_cls = R.get(ComponentEnum.JOB, job_config.backend)
            if not job_cls:
                raise ValueError(
                    f"Job '{job_config.name}' references an unregistered backend '{job_config.backend}' "
                    f"of type '{ComponentEnum.JOB}'",
                )
            params = job_config.model_dump()
            params["app_context"] = self.context
            self.context.jobs[job_config.name] = job_cls(**params)

    @property
    def config(self):
        """Get application configuration."""
        return self.context.app_config

    async def _start(self) -> None:
        """Start the application."""
        for components in self.context.components.values():
            for component in components.values():
                try:
                    await component.start()
                except Exception as e:
                    self.logger.exception(f"Failed to start component {component.__class__.__name__}: {e}")

        for name, job in self.context.jobs.items():
            try:
                await job.start()
            except Exception as e:
                self.logger.exception(f"Failed to start job '{name}': {e}")

    async def _close(self) -> None:
        """Close the application."""
        for name, job in self.context.jobs.items():
            try:
                await job.close()
            except Exception as e:
                self.logger.exception(f"Failed to close job '{name}': {e}")

        for components in reversed(list(self.context.components.values())):
            for component in reversed(list(components.values())):
                try:
                    await component.close()
                except Exception as e:
                    self.logger.exception(f"Failed to close component {component.__class__.__name__}: {e}")

    async def run_job(self, name: str, **kwargs) -> Response:
        """Execute a registered job by name."""
        if name not in self.context.jobs:
            raise KeyError(f"Job '{name}' not found")
        job = self.context.jobs[name]
        return await job(**kwargs)

    async def run_stream_job(self, name: str, **kwargs) -> AsyncGenerator[StreamChunk, None]:
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
        """Run the application as a service."""
        service = self.context.service
        service.run_app(app=self)
