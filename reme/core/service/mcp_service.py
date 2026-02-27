"""Model Context Protocol (MCP) service implementation."""

from contextlib import asynccontextmanager

from fastmcp import FastMCP
from fastmcp.tools import FunctionTool

from .base_service import BaseService
from ..flow import BaseFlow


class MCPService(BaseService):
    """Expose flows as Model Context Protocol (MCP) tools."""

    def __init__(self, **kwargs):
        """Initialize FastMCP instance with service settings."""
        super().__init__(**kwargs)

        @asynccontextmanager
        async def lifespan(_: FastMCP):
            await self.app.start()
            yield {}
            await self.app.close()

        self.mcp_service = FastMCP(name=self.service_config.app_name, lifespan=lifespan)

    def integrate_flow(self, flow: BaseFlow) -> str | None:
        """Register a non-streaming flow as an MCP tool."""
        if flow.stream:
            return None

        tool_call, request_model = self._prepare_route(flow)

        async def execute_tool(**kwargs):
            """Execute flow logic and return the string answer."""
            request_instance = request_model(**kwargs)
            response = await flow.call(**request_instance.model_dump(exclude_none=True))
            return response.answer

        self.mcp_service.add_tool(
            FunctionTool(
                name=tool_call.name,  # noqa
                description=tool_call.description,  # noqa
                fn=execute_tool,
                parameters=tool_call.parameters.simple_input_dump(),
            ),
        )
        return tool_call.name

    def run(self):
        """Run the MCP server with specified transport protocol."""
        super().run()
        cfg = self.service_config.mcp_service
        run_args: dict = {"transport": cfg.transport, "show_banner": False, **cfg.model_extra}
        if cfg.transport != "stdio":
            run_args.update({"host": cfg.host, "port": cfg.port})
        self.mcp_service.run(**run_args)
