import asyncio
from typing import AsyncGenerator

from .component import BaseComponent, ApplicationContext
from .schema import Response, StreamChunk
from .utils import execute_stream_task


class Application(BaseComponent):
    """Application component for managing the main application."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.context = ApplicationContext(**kwargs)

    @property
    def config(self):
        return self.context.app_config

    async def _start(self, app_context=None) -> None:
        """Start the application."""
        for components in self.context.components.values():
            for component in components.values():
                try:
                    await component.start(self.context)
                except Exception as e:
                    self.logger.exception(f"Failed to start component {component.__class__.__name__}: {e}")

        for name, job in self.context.jobs.items():
            try:
                await job.start(self.context)
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
        if name not in self.context.jobs:
            raise KeyError(f"Job '{name}' not found")
        job = self.context.jobs[name]
        return await job(app_context=self.context, **kwargs)

    async def run_stream_job(self, name: str, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        if name not in self.context.jobs:
            raise KeyError(f"Job '{name}' not found")
        job = self.context.jobs[name]
        stream_queue = asyncio.Queue()
        task = asyncio.create_task(job(stream_queue=stream_queue, app_context=self.context, **kwargs))
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