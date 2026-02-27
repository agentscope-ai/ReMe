"""MCP (Model Context Protocol) tool integration for remote tool execution."""

from mcp.types import CallToolResult, TextContent

from .base_tool import BaseTool
from ..schema import ToolCall
from ..utils import MCPClient


class MCPTool(BaseTool):
    """Operator for calling remote MCP (Model Context Protocol) tools."""

    def __init__(
        self,
        mcp_server: str = "",
        tool_name: str = "",
        parameter_required: list[str] | None = None,
        parameter_optional: list[str] | None = None,
        parameter_deleted: list[str] | None = None,
        max_retries: int = 3,
        timeout: float | None = None,
        raise_exception: bool = False,
        **kwargs,
    ):
        super().__init__(max_retries=max_retries, raise_exception=raise_exception, **kwargs)

        self.mcp_server: str = mcp_server
        self.tool_name: str = tool_name
        self.parameter_required: list[str] | None = parameter_required
        self.parameter_optional: list[str] | None = parameter_optional
        self.parameter_deleted: list[str] | None = parameter_deleted
        self.timeout: float | None = timeout

        # Example MCP marketplace: https://bailian.console.aliyun.com/?tab=mcp#/mcp-market
        self._client: MCPClient | None = None

    @property
    def client(self) -> MCPClient:
        """Lazily initialize and return the MCP client."""
        if self._client is None:
            self._client = MCPClient(self.service_context.service_config.mcp_servers)
        return self._client

    def _build_tool_call(self) -> ToolCall:
        tool_call_dict = self.service_context.mcp_server_mapping[self.mcp_server]
        tool_call: ToolCall = tool_call_dict[self.tool_name].model_copy(deep=True)

        # Initialize required list if not exists
        if tool_call.parameters.required is None:
            tool_call.parameters.required = []

        if self.parameter_required:
            for name in self.parameter_required:
                if name not in tool_call.parameters.required:
                    tool_call.parameters.required.append(name)

        if self.parameter_optional:
            for name in self.parameter_optional:
                if name in tool_call.parameters.required:
                    tool_call.parameters.required.remove(name)

        if self.parameter_deleted:
            for name in self.parameter_deleted:
                tool_call.parameters.properties.pop(name, None)
                if tool_call.parameters.required and name in tool_call.parameters.required:
                    tool_call.parameters.required.remove(name)

        return tool_call

    async def execute(self):
        tool_result: CallToolResult = await self.client.call_tool(
            server_name=self.mcp_server,
            tool_name=self.tool_name,
            arguments=self.input_dict,
        )
        self.context.tool_result = tool_result

        text_result = []
        for block in tool_result.content:
            if isinstance(block, TextContent):
                text_result.append(block.text)
        output: str = "\n".join(text_result)
        return output
