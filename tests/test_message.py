"""Test cases for message schema and serialization."""

import unittest

from mcp.types import Tool

from reme.core.enumeration import Role
from reme.core.schema import ToolAttr, ToolCall, ContentBlock, Message


class TestModelDefinitions(unittest.TestCase):
    """Test suite for validating message models and their serialization methods."""

    def test_tool_attr_serialization(self):
        """Test if ToolAttr correctly dumps to JSON schema format."""
        # Test simple string attribute with enum
        attr = ToolAttr(
            type="string",
            description="The city name",
            enum=["Beijing", "London"],
        )
        dump = attr.simple_input_dump()

        print("\n=== ToolAttr.simple_input_dump() (string with enum) ===")
        print(dump)

        self.assertEqual(dump["type"], "string")
        self.assertEqual(dump["enum"], ["Beijing", "London"])
        self.assertIn("description", dump)

        # Test object attribute with required child properties
        obj_attr = ToolAttr(
            type="object",
            description="User information",
            properties={
                "name": ToolAttr(type="string", description="User name"),
                "age": ToolAttr(type="number", description="User age"),
            },
            required=["name"],  # 'name' is required, 'age' is optional
        )
        obj_dump = obj_attr.simple_input_dump()

        print("\n=== ToolAttr.simple_input_dump() (object with required) ===")
        print(obj_dump)

        self.assertEqual(obj_dump["type"], "object")
        self.assertIn("properties", obj_dump)
        self.assertEqual(obj_dump["required"], ["name"])
        self.assertIn("name", obj_dump["properties"])
        self.assertIn("age", obj_dump["properties"])

    def test_tool_call_initialization(self):
        """Test if ToolCall correctly parses raw OpenAI-style tool definitions."""
        raw_input = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Check weather info",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "description": "Temperature unit"},
                    },
                    "required": ["location"],
                },
            },
        }

        tc = ToolCall(**raw_input)

        print("\n=== ToolCall.simple_input_dump() ===")
        print(tc.simple_input_dump())
        print("\n=== ToolCall.simple_output_dump() ===")
        print(tc.simple_output_dump())

        self.assertEqual(tc.name, "get_weather")
        self.assertIn("location", tc.parameters.properties)
        self.assertIn("unit", tc.parameters.properties)
        # Check that 'location' is in the required list at ToolCall level
        self.assertIn("location", tc.parameters.required)
        self.assertNotIn("unit", tc.parameters.required)

    def test_tool_call_argument_parsing(self):
        """Test JSON argument parsing and validation."""
        tc = ToolCall(name="test", arguments='{"key": "value"}')

        self.assertTrue(tc.check_argument())
        self.assertEqual(tc.argument_dict["key"], "value")

        # Test invalid JSON
        tc.arguments = "{invalid_json}"
        self.assertFalse(tc.check_argument())

    def test_content_block_dynamic_mapping(self):
        """Test if ContentBlock correctly identifies content based on type key."""
        # Test Image Block
        img_data = {"type": "image_url", "image_url": {"url": "http://test.com/a.jpg"}}
        block = ContentBlock(**img_data)
        self.assertEqual(block.type, "image_url")
        self.assertEqual(block.content["url"], "http://test.com/a.jpg")

        # Test Text Block
        text_data = {"type": "text", "text": "Hello World"}
        block = ContentBlock(**text_data)
        self.assertEqual(block.content, "Hello World")

    def test_message_simple_dump(self):
        """Test the transformation of Message to standard API dictionary."""
        msg = Message(
            role=Role.ASSISTANT,
            content="Thinking...",
            reasoning_content="I should check the weather first.",
            tool_calls=[ToolCall(name="get_weather", arguments='{"city": "NY"}', id="call_123")],
        )

        dump = msg.simple_dump(add_reasoning=True)

        print("\n=== Message.simple_dump(add_reasoning=True) ===")
        print(dump)

        dump_no_reasoning = msg.simple_dump(add_reasoning=False)
        print("\n=== Message.simple_dump(add_reasoning=False) ===")
        print(dump_no_reasoning)

        self.assertEqual(dump["role"], "assistant")
        self.assertEqual(dump["reasoning_content"], "I should check the weather first.")
        self.assertEqual(len(dump["tool_calls"]), 1)
        self.assertEqual(dump["tool_calls"][0]["id"], "call_123")

    def test_message_format_human_readable(self):
        """Test the string representation of messages for logging/UI."""
        msg = Message(
            role=Role.USER,
            content=[
                ContentBlock(type="text", text="Look at this:"),
                ContentBlock(type="image_url", image_url={"url": "img.png"}),
            ],
        )

        formatted = msg.format_message(index=1, use_name=False)

        self.assertIn("round1", formatted)
        self.assertIn("user:", formatted)
        self.assertIn("Look at this:", formatted)
        self.assertIn("img.png", formatted)

    def test_mcp_conversion(self):
        """Test the interoperability with MCP Tool format."""
        # Create a mock MCP Tool
        mcp_tool = Tool(
            name="calculator",
            description="adds numbers",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a"],
            },
        )

        # From MCP to ToolCall
        tc = ToolCall.from_mcp_tool(mcp_tool)

        print("\n=== ToolCall from MCP - simple_input_dump() ===")
        print(tc.simple_input_dump())
        print("\n=== ToolCall from MCP - simple_output_dump() ===")
        print(tc.simple_output_dump())

        self.assertEqual(tc.name, "calculator")
        self.assertIn("a", tc.parameters.properties)
        self.assertIn("b", tc.parameters.properties)
        # Check that 'a' is in the required list
        self.assertIn("a", tc.parameters.required)
        self.assertNotIn("b", tc.parameters.required)

        # From ToolCall back to MCP structure (via to_mcp_tool)
        # Note: This checks the logic of constructing the dict for Tool(...)
        mcp_compatible = tc.to_mcp_tool()

        print("\n=== MCP Tool converted back ===")
        print(f"Name: {mcp_compatible.name}")
        print(f"Description: {mcp_compatible.description}")
        print(f"InputSchema: {mcp_compatible.inputSchema}")

        self.assertEqual(mcp_compatible.name, "calculator")
        self.assertIn("a", mcp_compatible.inputSchema["properties"])
        self.assertIn("b", mcp_compatible.inputSchema["properties"])
        self.assertEqual(mcp_compatible.inputSchema["required"], ["a"])


if __name__ == "__main__":
    unittest.main()
