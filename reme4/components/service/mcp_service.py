"""MCP (Model Context Protocol) service: exposes jobs as MCP tools."""

from typing import TYPE_CHECKING

from fastmcp import FastMCP
from fastmcp.server.server import Transport
from fastmcp.tools import FunctionTool

from .base_service import BaseService
from ..component_registry import R
from ..job import BaseJob, StreamJob
from ...constants import REME_DEFAULT_HOST, REME_DEFAULT_PORT

if TYPE_CHECKING:
    from ...application import Application


@R.register("mcp")
class MCPService(BaseService):
    """Expose non-stream jobs as MCP tools over stdio, SSE, or other supported transports."""

    def __init__(
        self,
        transport: Transport = "sse",
        host: str = REME_DEFAULT_HOST,
        port: int = REME_DEFAULT_PORT,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transport: Transport = transport
        self.host: str = host
        self.port: int = port

    # ----- BaseService contract ------------------------------------------

    def build_service(self, app: "Application") -> None:
        """Create the FastMCP server with an app-managed lifespan."""
        self.service = FastMCP(
            name=app.config.app_name,
            lifespan=self._lifespan(app, self.host, self.port),
        )

    def add_job(self, job: BaseJob) -> None:
        """Register a non-stream job as an MCP tool; StreamJobs are skipped (not supported)."""
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
                parameters=job.parameters or None,
            ),
        )

    def start_service(self, app: "Application") -> None:
        """Run the MCP server; bind host/port only when the transport is network-based."""
        transport_kwargs: dict = {}
        if self.transport != "stdio":
            transport_kwargs["host"] = self.host
            transport_kwargs["port"] = self.port
        self.service.run(transport=self.transport, show_banner=False, **transport_kwargs)
