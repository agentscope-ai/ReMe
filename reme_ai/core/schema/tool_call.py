import json
from typing import Dict, List, Literal

from mcp.types import Tool
from pydantic import BaseModel, Field, model_validator, ConfigDict

TOOL_ATTR_TYPE = Literal["string", "array", "integer", "number", "boolean", "object"]


class ToolAttr(BaseModel):
    """Attributes for tool parameters."""

    type: TOOL_ATTR_TYPE = Field(default="string", description="tool attribute type")
    description: str = Field(default="", description="tool attribute description")
    required: bool = Field(default=True, description="tool attribute required")
    enum: List[str] | None = Field(default=None, description="tool attribute enum")
    items: dict = Field(default_factory=dict, description="tool attribute items")

    model_config = ConfigDict(extra="allow")

    def simple_input_dump(self, version: str = "default") -> dict:
        """Convert ToolAttr to input format dictionary for API requests."""
        if version == "default":
            result: dict = {
                "type": self.type,
                "description": self.description,
            }

            if self.enum:
                result["enum"] = self.enum

            if self.items:
                result["items"] = self.items

            return result

        else:
            raise NotImplementedError(f"version {version} not supported")


class ToolCall(BaseModel):
    """
    input:
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "It is very useful when you want to check the weather of a specified city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Cities or counties, such as Beijing, Hangzhou, Yuhang District, etc.",
                    }
                },
                "required": ["location"]
            }
        }
    }
    output:
    {
        "index": 0,
        "id": "call_6596dafa2a6a46f7a217da",
        "function": {
            "arguments": "{\"location\": \"Beijing\"}",
            "name": "get_current_weather"
        },
        "type": "function",
    }
    """

    index: int = Field(default=0)
    id: str = Field(default="")
    type: str = Field(default="function")
    name: str = Field(default="")

    arguments: str = Field(default="", description="tool execution arguments")

    description: str = Field(default="")
    input_schema: Dict[str, ToolAttr] = Field(default_factory=dict)
    output_schema: Dict[str, ToolAttr] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def init_tool_call(cls, data: dict):
        """Initialize ToolCall from raw data dictionary, extracting function info."""
        tool_type = data.get("type", "")
        tool_type_dict = data.get(tool_type, {})

        # Create a new dict to avoid modifying the original data
        result = data.copy()

        if "name" in tool_type_dict:
            result["name"] = tool_type_dict["name"]

        if "arguments" in tool_type_dict:
            result["arguments"] = tool_type_dict["arguments"]

        if "description" in tool_type_dict:
            result["description"] = tool_type_dict["description"]

        if "parameters" in tool_type_dict:
            parameters = tool_type_dict["parameters"]
            properties: dict = parameters.get("properties", {})
            required: list = parameters.get("required", [])
            result["input_schema"] = {}
            for name, param_attrs in properties.items():
                tool_attr = ToolAttr(**param_attrs)
                tool_attr.required = name in required
                result["input_schema"][name] = tool_attr

        return result

    @property
    def argument_dict(self) -> dict:
        """Parse and return arguments as a dictionary."""
        return json.loads(self.arguments)

    def check_argument(self) -> bool:
        """Check if arguments can be parsed as valid JSON."""
        try:
            _ = self.argument_dict
            return True
        except Exception:
            return False

    @staticmethod
    def _build_schema_dict(schema: Dict[str, ToolAttr]) -> dict:
        required_list = []
        properties = {}
        for name, tool_attr in schema.items():
            if tool_attr.required:
                required_list.append(name)
            properties[name] = tool_attr.simple_input_dump()

        return {
            "type": "object",
            "properties": properties,
            "required": required_list,
        }

    def simple_input_dump(self, version: str = "default") -> dict:
        """Convert ToolCall to input format dictionary for API requests."""
        if version == "default":
            return {
                "type": self.type,
                self.type: {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self._build_schema_dict(self.input_schema),
                },
            }

        else:
            raise NotImplementedError(f"version {version} not supported")

    def simple_output_dump(self, version: str = "default") -> dict:
        """Convert ToolCall to output format dictionary for API responses."""
        if version == "default":
            return {
                "index": self.index,
                "id": self.id,
                self.type: {
                    "arguments": self.arguments,
                    "name": self.name,
                },
                "type": self.type,
            }
        else:
            raise NotImplementedError(f"version {version} not supported")

    @classmethod
    def from_mcp_tool(cls, tool: Tool) -> "ToolCall":
        """Create a ToolCall instance from an MCP Tool object."""
        input_schema = {}
        properties = tool.inputSchema.get("properties", {})
        required = tool.inputSchema.get("required", [])
        for name, attr_dict in properties.items():
            tool_attr = ToolAttr(**attr_dict)
            if name in required:
                tool_attr.required = True
            input_schema[name] = tool_attr

        return cls(
            name=tool.name,
            description=tool.description or "",
            input_schema=input_schema,
        )

    def to_mcp_tool(self) -> Tool:
        """Convert this ToolCall to an MCP Tool object."""
        kwargs = {
            "name": self.name,
            "description": self.description,
            "inputSchema": self._build_schema_dict(self.input_schema),
        }

        # Build outputSchema from output_schema if present
        if self.output_schema:
            kwargs["outputSchema"] = self._build_schema_dict(self.output_schema)

        return Tool(**kwargs)
