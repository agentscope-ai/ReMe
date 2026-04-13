"""HTTP service implementation using FastAPI and uvicorn."""

import asyncio
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .base_service import BaseService
from ..job import BaseJob, StreamJob
from ...schema import Request, Response
from ...utils import execute_stream_task

# Environment variable name for ReMe port
REME_PORT_ENV = "REME_PORT"

if TYPE_CHECKING:
    from ...application import Application


class HttpService(BaseService):
    """HTTP service that exposes jobs as REST endpoints.

    Regular jobs return JSON responses, while StreamJobs return
    server-sent events (SSE) for real-time streaming.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Initialize the HTTP service.

        Args:
            host: Bind address for the server.
            port: Port number for the server.
            **kwargs: Additional arguments passed to uvicorn.
        """
        super().__init__(**kwargs)
        self.host: str = host
        self.port: int = port

    def _add_job(self, job: BaseJob) -> None:
        """Register a regular job as a POST endpoint.

        Args:
            job: The job to register.
        """

        async def execute_endpoint(request: Request) -> Response:
            return await job(**request.model_dump(exclude_none=True))

        self.service.post(
            path=f"/{job.name}",
            response_model=Response,
            description=job.description,
        )(execute_endpoint)

    def _add_stream_job(self, job: StreamJob) -> None:
        """Register a stream job as a SSE endpoint.

        Args:
            job: The stream job to register.
        """

        async def execute_stream_endpoint(request: Request) -> StreamingResponse:
            stream_queue = asyncio.Queue()
            task = asyncio.create_task(
                job(stream_queue=stream_queue, **request.model_dump(exclude_none=True))
            )

            async def generate_stream() -> AsyncGenerator[bytes, None]:
                async for chunk in execute_stream_task(
                        stream_queue=stream_queue,
                        task=task,
                        task_name=job.name,
                        output_format="bytes",
                ):
                    yield chunk

            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        self.service.post(f"/{job.name}")(execute_stream_endpoint)

    def add_job(self, job: BaseJob) -> None:
        """Register a job with the HTTP service.

        StreamJobs are registered as SSE endpoints, regular jobs as JSON endpoints.

        Args:
            job: The job to register.
        """
        if isinstance(job, StreamJob):
            self._add_stream_job(job)
        else:
            self._add_job(job)

    def build_service(self, app: "Application") -> None:
        """Build the FastAPI application with CORS middleware.

        Args:
            app: The application instance.
        """

        @asynccontextmanager
        async def lifespan(_: FastAPI):
            await app.start()
            yield
            await app.close()

        self.service = FastAPI(title=app.config.app_name, lifespan=lifespan)

        self.service.add_middleware(
            CORSMiddleware,  # type: ignore[arg-type]
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.service.get("/health")(lambda: {"status": "healthy"})

    def start_service(self, app: "Application") -> None:
        """Start the HTTP server.

        Args:
            app: The application instance.
        """
        os.environ[REME_PORT_ENV] = str(self.port)
        self.logger.info(f"Setting {REME_PORT_ENV}={self.port}")
        uvicorn.run(self.service, host=self.host, port=self.port, **self.kwargs)
