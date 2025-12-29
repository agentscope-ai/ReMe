"""MCP service for exposing flows as Model Context Protocol tools.

This service registers tool-callable flows with FastMCP and runs the selected
transport (SSE, HTTP, or STDIO).
"""

import os

from fastmcp import FastMCP
from fastmcp.tools import FunctionTool

from .base_service import BaseService
from ..context import C
from ..flow import BaseToolFlow
from ..utils.pydantic_utils import create_pydantic_model


@C.register_service("mcp")
class MCPService(BaseService):
    """Service that exposes tool flows via FastMCP transports."""

    def __init__(self, **kwargs):
        """Initialize the MCP server instance."""
        super().__init__(**kwargs)
        self.mcp = FastMCP(name=os.getenv("FLOW_APP_NAME"))

    def integrate_tool_flow(self, flow: BaseToolFlow) -> bool:
        """Register a tool-callable flow as an MCP tool."""
        request_model = create_pydantic_model(flow.name, flow.tool_call.input_schema)

        async def execute_tool(**kwargs):
            response = await flow.async_call(**request_model(**kwargs).model_dump(exclude_none=True))
            return response.answer

        # add tool
        tool_call_schema = flow.tool_call.simple_input_dump()
        parameters = tool_call_schema[tool_call_schema["type"]]["parameters"]
        tool = FunctionTool(
            name=flow.name,
            description=flow.tool_call.description,
            fn=execute_tool,
            parameters=parameters,
        )

        self.mcp.add_tool(tool)
        return True

    def run(self):
        """Run the MCP server using the configured transport and options."""
        super().run()
        mcp_config = self.service_config.mcp

        if mcp_config.transport == "stdio":
            self.mcp.run(transport="stdio", show_banner=False)
        else:
            assert mcp_config.transport in ["http", "sse", "streamable-http"]
            self.mcp.run(
                transport=mcp_config.transport,
                host=mcp_config.host,
                port=mcp_config.port,
                show_banner=False,
            )
