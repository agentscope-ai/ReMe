"""Model Context Protocol (MCP) service implementation.

Optionally exposes an HTTP sidecar (off by default) on a local-only
loopback port. The sidecar shares the running Application instance,
so callers (lifecycle hooks, scripts) see the live file_store without
booting a duplicate process.
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI, Request as FastAPIRequest
from fastmcp import FastMCP
from fastmcp.server.server import Transport
from fastmcp.tools import FunctionTool

from .base_service import BaseService
from ..component_registry import R
from ..job import StreamJob
from ...constants import REME_DEFAULT_HOST, REME_DEFAULT_PORT, REME_SERVICE_INFO

if TYPE_CHECKING:
    from ...application import Application
    from ..job import BaseJob


def _build_sidecar_app(app: "Application") -> FastAPI:
    """A minimal FastAPI app that exposes each registered job as POST /<job>."""
    api = FastAPI(title=f"{app.config.app_name}-sidecar")

    @api.get("/health")
    async def health():
        return {"status": "ok", "jobs": list(app.context.jobs.keys())}

    for job_name in list(app.context.jobs.keys()):
        # bind job_name via default arg to avoid late-binding in the closure
        async def endpoint(request: FastAPIRequest, _name: str = job_name):
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            response = await app.run_job(_name, **(payload or {}))
            return {"answer": response.answer, "success": response.success}

        api.post(f"/{job_name}")(endpoint)

    return api


@R.register("mcp")
class MCPService(BaseService):
    """Expose jobs as Model Context Protocol (MCP) tools."""

    def __init__(
        self,
        transport: Transport = "sse",
        host: str = REME_DEFAULT_HOST,
        port: int = REME_DEFAULT_PORT,
        sidecar_http: bool = False,
        sidecar_http_host: str = "127.0.0.1",
        sidecar_http_port: int = 8765,
        sidecar_info_path: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transport: Transport = transport
        self.host: str = host
        self.port: int = port
        self.sidecar_http: bool = sidecar_http
        self.sidecar_http_host: str = sidecar_http_host
        self.sidecar_http_port: int = sidecar_http_port
        # Optional file path where {host, port} are dropped at startup so that
        # external scripts (lifecycle hooks) can discover the sidecar URL.
        self.sidecar_info_path: str = sidecar_info_path

    def build_service(self, app: "Application") -> None:

        @asynccontextmanager
        async def lifespan(_: FastMCP):
            await app.start()
            service_info = json.dumps({"host": self.host, "port": self.port})
            os.environ[REME_SERVICE_INFO] = service_info
            self.logger.info(f"ReMe MCP Service started: {REME_SERVICE_INFO}={service_info}")

            sidecar_task: asyncio.Task | None = None
            if self.sidecar_http:
                sidecar_task = asyncio.create_task(self._serve_sidecar(app))

            try:
                yield
            finally:
                if sidecar_task is not None:
                    sidecar_task.cancel()
                    try:
                        await sidecar_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    self._cleanup_sidecar_info()
                await app.close()

        self.service = FastMCP(name=app.config.app_name, lifespan=lifespan)

    async def _serve_sidecar(self, app: "Application") -> None:
        sidecar_app = _build_sidecar_app(app)
        config = uvicorn.Config(
            sidecar_app,
            host=self.sidecar_http_host,
            port=self.sidecar_http_port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        self._publish_sidecar_info()
        self.logger.info(
            f"MCP HTTP sidecar listening on " f"http://{self.sidecar_http_host}:{self.sidecar_http_port}",
        )
        await server.serve()

    def _publish_sidecar_info(self) -> None:
        if not self.sidecar_info_path:
            return
        info_path = Path(self.sidecar_info_path)
        info_path.parent.mkdir(parents=True, exist_ok=True)
        info_path.write_text(
            json.dumps({"host": self.sidecar_http_host, "port": self.sidecar_http_port}),
            encoding="utf-8",
        )

    def _cleanup_sidecar_info(self) -> None:
        if not self.sidecar_info_path:
            return
        try:
            Path(self.sidecar_info_path).unlink(missing_ok=True)
        except Exception:
            pass

    def add_job(self, job: "BaseJob") -> None:
        if isinstance(job, StreamJob):
            return

        async def execute_tool(**kwargs):
            response = await job(**kwargs)
            return response.answer

        self.service.add_tool(
            FunctionTool(
                name=job.name,
                description=job.description,
                fn=execute_tool,
                parameters=job.parameters if job.parameters else None,
            ),
        )

    def start_service(self, app: "Application") -> None:
        transport_kwargs = {}
        if self.transport != "stdio":
            transport_kwargs["host"] = self.host
            transport_kwargs["port"] = self.port
        self.service.run(transport=self.transport, show_banner=False, **transport_kwargs)
