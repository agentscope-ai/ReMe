"""HTTP service: exposes jobs as FastAPI endpoints (JSON, or SSE for stream jobs)."""

import asyncio
import warnings
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .base_service import BaseService
from ..component_registry import R
from ..job import BaseJob, StreamJob
from ...constants import REME_DEFAULT_HOST, REME_DEFAULT_PORT
from ...schema import Request, Response
from ...utils import execute_stream_task, resolve_web_static_dir

if TYPE_CHECKING:
    from ...application import Application


# uvicorn 0.41 still imports these deprecated websockets symbols on startup,
# even though we don't use WebSocket. Silence just those specific warnings.
_WEBSOCKET_DEPRECATION_PATTERNS = (
    r".*websockets\.legacy is deprecated.*",
    r".*WebSocketServerProtocol is deprecated.*",
)


@R.register("http")
class HttpService(BaseService):
    """Map non-stream jobs to JSON POST endpoints and StreamJobs to SSE endpoints."""

    def __init__(
        self,
        host: str = REME_DEFAULT_HOST,
        port: int = REME_DEFAULT_PORT,
        web_enabled: bool = True,
        web_static_dir: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.host: str = host
        self.port: int = port
        self.web_enabled = web_enabled
        self.web_static_dir = web_static_dir

    # ----- BaseService contract ------------------------------------------

    def build_service(self, app: "Application") -> None:
        """Create the FastAPI app with permissive CORS and an app-managed lifespan."""
        self.service = FastAPI(
            title=app.config.app_name,
            lifespan=self._lifespan(app, self.host, self.port),
        )
        cors_origins = ["*"]
        self.service.add_middleware(
            CORSMiddleware,  # type: ignore[arg-type]
            allow_origins=cors_origins,
            allow_credentials="*" not in cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def add_job(self, job: BaseJob) -> bool:
        """Dispatch to streaming or non-streaming registration based on job type."""
        if isinstance(job, StreamJob):
            self._add_stream_job(job)
        else:
            self._add_json_job(job)
        return True

    def start_service(self, app: "Application") -> None:
        """Run uvicorn, suppressing unrelated websocket deprecation noise."""
        for pattern in _WEBSOCKET_DEPRECATION_PATTERNS:
            warnings.filterwarnings("ignore", category=DeprecationWarning, message=pattern)
        uvicorn.run(self.service, host=self.host, port=self.port, **self.kwargs)

    def finalize_service(self, app: "Application") -> None:
        """Serve the optional workspace UI after all job endpoints are registered."""
        del app
        if not self.web_enabled:
            return

        static_dir = resolve_web_static_dir(self.web_static_dir)
        if static_dir is None:
            self.logger.info("Web workspace is unavailable; no static build was found")
            return

        index_file = static_dir / "index.html"
        assets_dir = static_dir / "assets"
        if assets_dir.is_dir():
            self.service.mount(
                "/assets",
                StaticFiles(directory=str(assets_dir)),
                name="web-assets",
            )

        no_cache_headers = {"Cache-Control": "no-cache, no-store, must-revalidate"}

        @self.service.get("/{full_path:path}", include_in_schema=False)
        async def workspace_spa(full_path: str):
            if full_path in {"docs", "redoc", "openapi.json"}:
                raise HTTPException(status_code=404, detail="Not Found")

            if full_path and ".." not in full_path and not Path(full_path).is_absolute():
                static_file = static_dir / full_path
                if static_file.is_file():
                    return FileResponse(static_file)

            return FileResponse(index_file, headers=no_cache_headers)

    # ----- Endpoint factories --------------------------------------------

    def _add_json_job(self, job: BaseJob) -> None:
        """Register a job as POST /{job.name} returning a JSON Response."""

        async def endpoint(request: Request) -> Response:
            return await job(**request.model_dump(exclude_none=True))

        self.service.post(
            f"/{job.name}",
            response_model=Response,
            description=job.description,
        )(endpoint)

    def _add_stream_job(self, job: StreamJob) -> None:
        """Register a StreamJob as POST /{job.name} streaming chunks as text/event-stream."""

        async def endpoint(request: Request) -> StreamingResponse:
            stream_queue: asyncio.Queue = asyncio.Queue()
            task = asyncio.create_task(
                job(stream_queue=stream_queue, **request.model_dump(exclude_none=True)),
            )

            async def body() -> AsyncGenerator[bytes, None]:
                async for chunk in execute_stream_task(
                    stream_queue=stream_queue,
                    task=task,
                    task_name=job.name,
                    output_format="bytes",
                ):
                    assert isinstance(chunk, bytes)
                    yield chunk

            return StreamingResponse(body(), media_type="text/event-stream")

        self.service.post(f"/{job.name}")(endpoint)
