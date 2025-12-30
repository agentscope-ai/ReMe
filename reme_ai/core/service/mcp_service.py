import os

from fastmcp import FastMCP
from fastmcp.tools import FunctionTool

from .base_service import BaseService
from ..context import C
from ..flow import BaseFlow
from ..utils.pydantic_utils import create_pydantic_model


@C.register_service("mcp")
class MCPService(BaseService):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mcp = FastMCP(name=os.getenv("APP_NAME", "ReMe"))

    def integrate_flow(self, flow: BaseFlow) -> bool:
        request_model = create_pydantic_model(flow.name, flow.tool_call.input_schema)

        async def execute_tool(**kwargs):
            response = await flow.call(**request_model(**kwargs).model_dump(exclude_none=True))
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
        super().run()
        mcp_config = self.service_config.mcp

        if mcp_config.transport == "stdio":
            self.mcp.run(transport="stdio", show_banner=False, **mcp_config.model_extra)
        else:
            assert mcp_config.transport in ["http", "sse", "streamable-http"]
            self.mcp.run(
                transport=mcp_config.transport,
                host=mcp_config.host,
                port=mcp_config.port,
                show_banner=False,
                **mcp_config.model_extra,
            )
