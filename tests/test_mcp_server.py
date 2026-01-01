"""Dynamic MCP server implementation with JSON-schema based tool registration."""

from typing import Any

from fastmcp import FastMCP
from fastmcp.tools import FunctionTool

from reme_ai.core.schema import ToolCall
from reme_ai.core.utils import create_pydantic_model

mcp = FastMCP("DynamicSchemaServer", port=8010)

# Configuration including enum examples
MODES_CONFIG = {
    "register_user": ToolCall(
        **{
            "name": "register_user",
            "description": "Register a new user with metadata, tags, and roles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "username": {"type": "string", "description": "Unique username"},
                    "role": {
                        "type": "string",
                        "enum": ["admin", "editor", "viewer"],
                        "description": "User access level",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "User metadata",
                        "properties": {
                            "age": {"type": "integer"},
                            "location": {"type": "string"},
                        },
                        "required": ["age"],
                    },
                    "tags": {
                        "type": "array",
                        "description": "User tags",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tag_id": {"type": "string"},
                                "level": {"type": "number"},
                            },
                            "required": ["tag_id"],
                        },
                    },
                },
                "required": ["username", "metadata", "role"],
            },
        },
    ),
    "create_order": ToolCall(
        **{
            "name": "create_order",
            "description": "创建订单",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单ID"},
                    "amount": {"type": "number", "description": "订单金额"},
                    "customer": {
                        "type": "object",
                        "description": "客户信息",
                        "properties": {
                            "name": {"type": "string", "description": "客户姓名"},
                            "email": {"type": "string", "description": "客户邮箱"},
                            "phone": {"type": "string", "description": "联系电话"},
                        },
                        "required": ["name", "email"],
                    },
                },
                "required": ["order_id", "customer"],
            },
        },
    ),
}


async def core_handler(mode: str, **kwargs: Any) -> dict[str, Any]:
    """Process dynamic tool requests and return execution results."""
    print(f"Executing Mode: {mode}, Parameters: {kwargs}")
    return {
        "status": "success",
        "mode": mode,
        "received_data": kwargs,
    }


def register_dynamic_tools() -> None:
    """Iterate over tool configurations and register them to the MCP instance."""
    for mode_name, tool_call in MODES_CONFIG.items():
        # Create Pydantic model from tool parameters
        request_model = create_pydantic_model(tool_call.name, tool_call.parameters)

        # Create execution function with closure to capture current mode and model
        def create_tool_func(current_mode: str, model: type):
            async def execute_tool(**kwargs: Any) -> dict[str, Any]:
                # Validate and normalize input using Pydantic model
                validated_data = model(**kwargs).model_dump(exclude_none=True)
                return await core_handler(current_mode, **validated_data)

            return execute_tool

        tool_fn = create_tool_func(mode_name, request_model)

        # Extract parameters schema
        tool_call_schema = tool_call.simple_input_dump()
        parameters = tool_call_schema[tool_call_schema["type"]]["parameters"]

        # Create FunctionTool and register
        tool = FunctionTool(
            name=tool_call.name,
            description=tool_call.description,
            fn=tool_fn,
            parameters=parameters,
        )

        mcp.add_tool(tool)


if __name__ == "__main__":
    register_dynamic_tools()
    mcp.run(transport="sse")
