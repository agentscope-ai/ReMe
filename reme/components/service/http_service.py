"""HTTP service: expose jobs through JSON/SSE endpoints and MCP tools."""

import asyncio
import warnings
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.routing import Route

from .base_service import BaseService
from ..component_registry import R
from ..job import BaseJob, StreamJob
from ...constants import REME_DEFAULT_HOST, REME_DEFAULT_PORT
from ...schema import Request, Response
from ...utils import execute_stream_task, resolve_web_static_dir
from .mcp_tools import add_mcp_job

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
    """Expose jobs through JSON/SSE endpoints and streamable HTTP MCP."""

    def __init__(
        self,
        host: str = REME_DEFAULT_HOST,
        port: int = REME_DEFAULT_PORT,
        web_enabled: bool = True,
        web_static_dir: str | None = None,
        mcp_enabled: bool = True,
        mcp_path: str = "/mcp",
        mcp_stateless_http: bool = False,
        injected_job_kwargs: dict[str, Any] | None = None,
        tool_error_on_failure: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.host: str = host
        self.port: int = port
        self.web_enabled = web_enabled
        self.web_static_dir = web_static_dir
        self.mcp_enabled = mcp_enabled
        self.mcp_path = self._validate_mcp_path(mcp_path)
        self.mcp_stateless_http = mcp_stateless_http
        self.injected_job_kwargs = dict(injected_job_kwargs or {})
        self.tool_error_on_failure = tool_error_on_failure
        self.mcp_server = None
        self.mcp_app = None

    # ----- BaseService contract ------------------------------------------

    def build_service(self, app: "Application") -> None:
        """Create one FastAPI app containing JSON/SSE and optional MCP routes."""
        lifespan = self._lifespan(app, self.host, self.port)
        if self.mcp_enabled:
            from fastmcp import FastMCP
            from fastmcp.utilities.lifespan import combine_lifespans

            self.mcp_server = FastMCP(name=app.config.app_name)
            self.mcp_app = self.mcp_server.http_app(
                path=self.mcp_path,
                transport="streamable-http",
                stateless_http=self.mcp_stateless_http,
            )
            lifespan = combine_lifespans(lifespan, self.mcp_app.lifespan)

        self.service = FastAPI(
            title=app.config.app_name,
            lifespan=lifespan,
        )
        cors_origins = ["*"]
        self.service.add_middleware(
            CORSMiddleware,  # type: ignore[arg-type]
            allow_origins=cors_origins,
            allow_credentials="*" not in cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        if self.mcp_app is not None:
            # Forward the exact path to the complete FastMCP ASGI app. Copying
            # only its routes would bypass its middleware and application state;
            # mounting it would make the trailing-slash path canonical instead.
            self.service.router.routes.append(
                Route(
                    self.mcp_path,
                    endpoint=self.mcp_app,
                    include_in_schema=False,
                ),
            )

    def add_jobs(self, app: "Application") -> None:
        """Validate reserved routes before the shared tolerant registration loop."""
        if self.mcp_enabled:
            conflicts = sorted(
                job.name
                for name, job in app.context.jobs.items()
                if job.enable_serve and (self.jobs is None or name in self.jobs) and f"/{job.name}" == self.mcp_path
            )
            if conflicts:
                names = ", ".join(conflicts)
                raise ValueError(
                    f"Job name conflicts with the MCP endpoint {self.mcp_path!r}: {names}",
                )
        super().add_jobs(app)

    def add_job(self, job: BaseJob) -> bool:
        """Register HTTP routes for every job and MCP tools for non-stream jobs."""
        if self.mcp_enabled and f"/{job.name}" == self.mcp_path:
            raise ValueError(
                f"Job name '{job.name}' conflicts with the MCP endpoint {self.mcp_path!r}",
            )
        if isinstance(job, StreamJob):
            self._add_stream_job(job)
        else:
            self._add_json_job(job)
            if self.mcp_server is not None:
                add_mcp_job(
                    self.mcp_server,
                    job,
                    injected_job_kwargs=self.injected_job_kwargs,
                    tool_error_on_failure=self.tool_error_on_failure,
                )
        return True

    def start_service(self, app: "Application") -> None:
        """Run uvicorn, suppressing unrelated websocket deprecation noise."""
        for pattern in _WEBSOCKET_DEPRECATION_PATTERNS:
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message=pattern,
            )
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
        post_only_paths = {
            route.path
            for route in self.service.routes
            if "POST" in (getattr(route, "methods", None) or set())
            and "GET" not in (getattr(route, "methods", None) or set())
        }

        @self.service.get("/{full_path:path}", include_in_schema=False)
        async def workspace_spa(full_path: str):
            if full_path in {"docs", "redoc", "openapi.json"}:
                raise HTTPException(status_code=404, detail="Not Found")
            if f"/{full_path}" in post_only_paths:
                raise HTTPException(
                    status_code=405,
                    detail="Method Not Allowed",
                    headers={"Allow": "POST"},
                )

            if full_path and not Path(full_path).is_absolute():
                static_file = (static_dir / full_path).resolve()
                if static_file.is_relative_to(static_dir) and static_file.is_file():
                    return FileResponse(static_file)

            return FileResponse(index_file, headers=no_cache_headers)

    # ----- Endpoint factories --------------------------------------------

    @staticmethod
    def _validate_mcp_path(path: str) -> str:
        """Return a canonical, non-reserved absolute path for the MCP endpoint."""
        if not path.startswith("/") or path == "/" or path.endswith("/"):
            raise ValueError(
                "mcp_path must start with '/', must not be '/', and must not end with '/'",
            )
        if "//" in path or any(segment in {".", ".."} for segment in path.split("/")):
            raise ValueError("mcp_path must use non-empty literal path segments")
        if any(char in path for char in "{}?#\\") or any(
            char.isspace() or ord(char) < 32 or ord(char) == 127 for char in path
        ):
            raise ValueError("mcp_path must be a literal URL path without route, query, or fragment syntax")
        if path in {"/assets", "/docs", "/redoc", "/openapi.json"}:
            raise ValueError(f"mcp_path conflicts with reserved HTTP path {path!r}")
        return path

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
