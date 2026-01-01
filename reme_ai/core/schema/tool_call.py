"""
MCP Tool Schema definitions for recursive JSON Schema representation.
"""

import json
from typing import Any, Dict, List, Optional, Union

from mcp.types import Tool
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator

from ..enumeration.json_schema_enum import JsonSchemaEnum


class ToolAttr(BaseModel):
    """Recursive model representing JSON Schema attributes for tool parameters."""

    model_config = ConfigDict(extra="allow")

    type: str = Field(default=str(JsonSchemaEnum.STRING), description="The data type of the attribute")
    description: Optional[str] = Field(default=None, description="Description of the attribute")
    required: Optional[List[str]] = Field(default=None, description="Required property names for object types")
    properties: Optional[Dict[str, "ToolAttr"]] = Field(default=None, description="Child properties for objects")
    items: Optional[Union[Dict[str, Any], "ToolAttr"]] = Field(default=None, description="Schema for array items")
    enum: Optional[List[str]] = Field(default=None, description="Allowed values for the attribute")

    @field_validator("type")
    @classmethod
    def validate_type_is_valid_enum(cls, v: str) -> str:
        """Validates that the provided type string exists within JsonSchemaEnum values."""
        valid_types = [str(e) for e in JsonSchemaEnum]

        if v not in valid_types:
            raise ValueError(f"Invalid type: '{v}'. Must be one of {valid_types}")
        return v

    def simple_input_dump(self) -> dict:
        """Serializes the attribute into a standard JSON Schema dictionary."""
        res: dict = {"type": self.type}
        if self.description:
            res["description"] = self.description
        if self.enum:
            res["enum"] = self.enum

        if self.type == "object" and self.properties:
            res["properties"] = {
                k: v.simple_input_dump() if isinstance(v, ToolAttr) else v for k, v in self.properties.items()
            }
            if self.required:
                res["required"] = self.required

        if self.type == "array" and self.items:
            res["items"] = self.items.simple_input_dump() if isinstance(self.items, ToolAttr) else self.items

        return res


# Enable recursive type resolution
ToolAttr.model_rebuild()


class ToolCall(BaseModel):
    """
    Model representing a tool definition and its call structure.
    Supports parsing from standard JSON Schema formats and converting to MCP Tool objects.
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

    index: int = 0
    id: str = ""
    type: str = "function"
    name: str = ""
    description: str = ""

    arguments: str = Field(default="", description="JSON string of tool execution arguments")

    parameters: ToolAttr = Field(
        default_factory=lambda: ToolAttr(type="object", properties={}, required=[]),
        description="Specification for input parameters",
    )

    output: ToolAttr = Field(
        default_factory=lambda: ToolAttr(type="object", properties={}),
        description="Specification for the execution result (Schema)",
    )

    @model_validator(mode="before")
    @classmethod
    def init_tool_call(cls, data: dict) -> dict:
        """Initializes the model by parsing tool-specific body data."""
        data = data.copy()
        t_type = data.get("type", "function")
        body = data.get(t_type, {})

        # Extract basic metadata
        data["name"] = body.get("name", data.get("name", ""))
        data["arguments"] = body.get("arguments", data.get("arguments", ""))
        data["description"] = body.get("description", data.get("description", ""))

        # Handle parameters mapping
        if "parameters" in body:
            params = body["parameters"]
            # If parameters is already a dict, ensure it matches ToolAttr structure
            if isinstance(params, dict):
                data["parameters"] = ToolAttr(**params)

        # Handle output mapping (if provided in source)
        if "output" in body and isinstance(body["output"], dict):
            data["output"] = ToolAttr(**body["output"])

        return data

    def simple_input_dump(self) -> dict:
        """Returns a standardized tool definition dictionary."""
        return {
            "type": self.type,
            self.type: {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters.simple_input_dump(),
            },
        }

    @classmethod
    def from_mcp_tool(cls, tool: Tool) -> "ToolCall":
        """Creates a ToolCall instance from an MCP Tool object."""
        # MCP Tool inputSchema maps directly to our parameters ToolAttr
        return cls(
            name=tool.name,
            description=tool.description or "",
            parameters=ToolAttr(**tool.inputSchema),
        )

    def to_mcp_tool(self) -> Tool:
        """Converts the instance back into an MCP Tool object."""
        return Tool(
            name=self.name,
            description=self.description,
            inputSchema=self.parameters.simple_input_dump(),
        )

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

    def simple_output_dump(self) -> dict:
        """Convert ToolCall to output format dictionary for API responses."""
        return {
            "index": self.index,
            "id": self.id,
            self.type: {
                "arguments": self.arguments,
                "name": self.name,
            },
            "type": self.type,
        }
