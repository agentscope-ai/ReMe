"""MCP (Model Context Protocol) tool integration for remote tool execution."""

from typing import List

from ..context import C
from ..op import BaseOp
from ..schema import ToolCall
from ..utils import MCPClient


@C.register_op()
class MCPTool(BaseOp):
    """Operator for calling remote MCP (Model Context Protocol) tools.

    This class enables integration with external MCP servers to execute tools
    and retrieve their results. It supports parameter customization and retry logic.
    """

    def __init__(
        self,
        mcp_server: str = "",
        tool_name: str = "",
        save_response_result: bool = True,
        parameter_required: List[str] | None = None,
        parameter_optional: List[str] | None = None,
        parameter_deleted: List[str] | None = None,
        max_retries: int = 3,
        timeout: float | None = None,
        raise_exception: bool = False,
        **kwargs,
    ):

        super().__init__(
            save_response_result=save_response_result,
            max_retries=max_retries,
            raise_exception=raise_exception,
            **kwargs,
        )

        self.mcp_server: str = mcp_server
        self.tool_name: str = tool_name
        self.parameter_required: List[str] | None = parameter_required
        self.parameter_optional: List[str] | None = parameter_optional
        self.parameter_deleted: List[str] | None = parameter_deleted
        self.timeout: float | None = timeout
        # Example MCP marketplace: https://bailian.console.aliyun.com/?tab=mcp#/mcp-market

        self._client = MCPClient(C.service_config.mcp_servers)

    def _build_tool_call(self) -> ToolCall:
        tool_call_dict = C.mcp_server_tool_call_mapping[self.mcp_server]
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
        self.output = await self._client.call_tool(
            server_name=self.mcp_server,
            tool_name=self.tool_name,
            arguments=self.input_dict,
            parse_text_result=True,
        )
