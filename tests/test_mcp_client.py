"""Test module for demonstrating MCPClient functionality."""

# pylint: disable=too-many-return-statements,too-many-statements

import asyncio
import json

from reme_ai.core.utils import MCPClient


async def main():
    """Execute demonstration of the MCPClient."""
    test_mcp = "test_mcp"
    config_data = {
        "mcpServers": {
            test_mcp: {
                "url": "http://127.0.0.1:8010/sse",
            },
        },
    }

    client = MCPClient(config_data)

    try:
        # List all available tools
        print("=" * 50)
        print("Listing available tools:")
        print("=" * 50)
        t_list = await client.list_tool_calls(test_mcp)
        for t in t_list:
            print(json.dumps(t, ensure_ascii=False, indent=2))

        # Helper function to build default values
        def build_default_value(param_info: dict) -> any:
            """Build a default value for a parameter based on its schema."""
            param_type = param_info.get("type", "string")

            # Handle enum types - use the first enum value
            if "enum" in param_info and param_info["enum"]:
                return param_info["enum"][0]

            # Handle different types
            if param_type == "string":
                return "example_string"
            elif param_type == "number":
                return 0.0
            elif param_type == "integer":
                return 0
            elif param_type == "boolean":
                return False
            elif param_type == "array":
                return []
            elif param_type == "object":
                # Recursively build nested objects
                obj = {}
                nested_properties = param_info.get("properties", {})
                nested_required = param_info.get("required", [])

                for nested_param_name in nested_required:
                    if nested_param_name in nested_properties:
                        nested_param_info = nested_properties[nested_param_name]
                        obj[nested_param_name] = build_default_value(nested_param_info)

                return obj
            else:
                return None

        # Call tools if available
        if t_list:
            # Execute the first two tools (or fewer if not enough tools available)
            tools_to_execute = min(2, len(t_list))

            for idx in range(tools_to_execute):
                print("\n" + "=" * 50)
                print(f"Calling tool #{idx + 1}:")
                print("=" * 50)

                # Get the tool's information
                current_tool = t_list[idx]
                tool_type = current_tool.get("type", "function")
                tool_body = current_tool.get(tool_type, {})

                tool_name = tool_body.get("name")
                tool_description = tool_body.get("description", "")

                # Prepare arguments based on the tool's input schema
                tool_arguments = {}
                parameters = tool_body.get("parameters", {})
                properties = parameters.get("properties", {})
                required = parameters.get("required", [])

                # Build minimal arguments for required parameters
                for param_name in required:
                    if param_name in properties:
                        param_info = properties[param_name]
                        tool_arguments[param_name] = build_default_value(param_info)

                # Validate tool name exists
                if not tool_name:
                    print("Error: Tool name not found in the tool definition")
                    print(f"Tool structure: {json.dumps(current_tool, ensure_ascii=False, indent=2)}")
                else:
                    print(f"Tool name: {tool_name}")
                    print(f"Tool description: {tool_description}")
                    print(f"Arguments: {json.dumps(tool_arguments, ensure_ascii=False, indent=2)}")

                    # Call the tool
                    result = await client.call_tool(test_mcp, tool_name, tool_arguments, parse_text_result=True)

                    print("\n" + "-" * 50)
                    print("Tool call result:")
                    print("-" * 50)
                    print(f"Content: {result}")
                    if hasattr(result, "isError"):
                        print(f"Is Error: {result.isError}")
        else:
            print("\nNo tools available to call.")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
