"""MCP service: expose jobs as MCP tools."""

from typing import TYPE_CHECKING, Any

from .base_service import BaseService
from ..component_registry import R
from ..job import BaseJob
from ...constants import REME_DEFAULT_HOST, REME_DEFAULT_PORT
from .mcp_tools import add_mcp_job

if TYPE_CHECKING:
    from fastmcp.server.server import Transport
    from ...application import Application


@R.register("mcp")
class MCPService(BaseService):
    """Expose non-stream jobs as MCP tools over stdio, SSE, or streamable-http."""

    def __init__(
        self,
        transport: "Transport" = "sse",
        host: str = REME_DEFAULT_HOST,
        port: int = REME_DEFAULT_PORT,
        injected_job_kwargs: dict[str, Any] | None = None,
        tool_error_on_failure: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transport: Transport = transport
        self.host: str = host
        self.port: int = port
        self.injected_job_kwargs = dict(injected_job_kwargs or {})
        self.tool_error_on_failure = tool_error_on_failure

    # ----- BaseService contract ------------------------------------------

    def build_service(self, app: "Application") -> None:
        """Construct the FastMCP server."""
        from fastmcp import FastMCP

        self.service = FastMCP(
            name=app.config.app_name,
            lifespan=self._lifespan(app, self.host, self.port),
        )

    def add_job(self, job: BaseJob) -> bool:
        """Register a non-stream job as an MCP tool; StreamJobs are unsupported."""
        return add_mcp_job(
            self.service,
            job,
            injected_job_kwargs=self.injected_job_kwargs,
            tool_error_on_failure=self.tool_error_on_failure,
        )

    def start_service(self, app: "Application") -> None:
        """Run the MCP server; bind host/port only for network transports."""
        transport_kwargs: dict = {}
        if self.transport != "stdio":
            transport_kwargs["host"] = self.host
            transport_kwargs["port"] = self.port
        self.service.run(
            transport=self.transport,
            show_banner=False,
            **transport_kwargs,
        )
