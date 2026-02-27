"""Think tool for agent reflection and planning.

This module provides a tool that prompts the model for explicit reflection
before taking actions, helping agents reason about their next steps.
"""

from ..op import BaseTool
from ..schema import ToolCall


class ThinkTool(BaseTool):
    """Utility that prompts the model for explicit reflection text."""

    def __init__(self, add_output_reflection: bool = False, **kwargs):
        """Initialize the think tool."""
        super().__init__(**kwargs)
        self.add_output_reflection: bool = add_output_reflection

    def _build_tool_call(self) -> ToolCall:
        """Build the tool call schema for think tool."""
        return ToolCall(
            **{
                "description": self.get_prompt("tool"),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reflection": {
                            "type": "string",
                            "description": self.get_prompt("reflection"),
                        },
                    },
                    "required": ["reflection"],
                },
            },
        )

    async def execute(self):
        """Execute the think tool by processing reflection input."""
        if self.add_output_reflection:
            return self.context["reflection"]
        else:
            return self.get_prompt("reflection_output")
