"""Model Context Protocol (MCP) service implementation."""
import json
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

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


@R.register("mcp")
class MCPService(BaseService):
    """Expose jobs as Model Context Protocol (MCP) tools."""

    def __init__(
            self,
            transport: Transport = "sse",
            host: str = REME_DEFAULT_HOST,
            port: int = REME_DEFAULT_PORT,
            **kwargs):
        super().__init__(**kwargs)
        self.transport: Transport = transport
        self.host: str = host
        self.port: int = port

    def build_service(self, app: "Application") -> None:

        @asynccontextmanager
        async def lifespan(_: FastMCP):
            await app.start()
            service_info = json.dumps({"host": self.host, "port": self.port})
            os.environ[REME_SERVICE_INFO] = service_info
            self.logger.info(f"ReMe MCP Service started: {REME_SERVICE_INFO}={service_info}")
            yield
            await app.close()

        self.service = FastMCP(name=app.config.app_name, lifespan=lifespan)

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
