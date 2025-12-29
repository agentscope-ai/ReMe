from typing import List, Optional

from . import BaseOp
from ..context import C
from ..schema import ToolCall
from ..utils import FastMcpClient


@C.register_op()
class ExternalMCP(BaseOp):

    def __init__(
            self,
            mcp_name: str = "",
            tool_name: str = "",
            save_answer: bool = True,
            input_schema_required: List[str] = None,
            input_schema_optional: List[str] = None,
            input_schema_deleted: List[str] = None,
            max_retries: int = 3,
            timeout: float | None = None,
            raise_exception: bool = False,
            **kwargs,
    ):
        self.mcp_name: str = mcp_name
        self.tool_name: str = tool_name
        self.input_schema_required: List[str] = input_schema_required
        self.input_schema_optional: List[str] = input_schema_optional
        self.input_schema_deleted: List[str] = input_schema_deleted
        self.timeout: Optional[float] = timeout
        super().__init__(save_answer=save_answer, max_retries=max_retries, raise_exception=raise_exception, **kwargs)
        # Example MCP marketplace: https://bailian.console.aliyun.com/?tab=mcp#/mcp-market

    def _build_tool_call(self) -> ToolCall:
        tool_call_dict = C.external_mcp_tool_call_dict[self.mcp_name]
        tool_call: ToolCall = tool_call_dict[self.tool_name].model_copy(deep=True)

        if self.input_schema_required:
            for name in self.input_schema_required:
                tool_call.input_schema[name].required = True

        if self.input_schema_optional:
            for name in self.input_schema_optional:
                tool_call.input_schema[name].required = False

        if self.input_schema_deleted:
            for name in self.input_schema_deleted:
                tool_call.input_schema.pop(name, None)

        return tool_call

    async def execute(self):
        mcp_server_config = C.service_config.external_mcp[self.mcp_name]
        async with FastMcpClient(
                name=self.mcp_name,
                config=mcp_server_config,
                max_retries=self.max_retries,
                timeout=self.timeout,
        ) as client:
            self.output = await client.call_tool(tool_name=self.tool_name, arguments=self.input_dict, parse_result=True)
