"""MCP client addresses and SDK tool descriptors."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from mcp.types import Tool, ToolAnnotations
import pytest

from reme.components.client.mcp_client import MCPClient
from reme.constants import REME_DEFAULT_HOST, REME_DEFAULT_PORT


def test_default_mcp_client_uses_loopback_address():
    """The network MCP client uses a connectable loopback destination."""
    client = MCPClient()

    assert client.host == REME_DEFAULT_HOST
    assert client.port == REME_DEFAULT_PORT


def test_mcp_client_converts_wildcard_bind_address_to_loopback():
    """A wildcard service address is normalized before transport construction."""
    client = MCPClient(host="0.0.0.0", port=8123)

    assert client.host == "127.0.0.1"
    assert client.port == 8123


@pytest.mark.asyncio
async def test_mcp_catalog_uses_sdk_field_names():
    """Tool catalogs expose SDK 2 fields while preserving schemas and user metadata."""
    parameters = {
        "type": "object",
        "$defs": {"Query": {"type": "string"}},
        "properties": {"query": {"$ref": "#/$defs/Query"}},
        "required": ["query"],
    }
    output = {"type": "object", "properties": {"answer": {"type": "string"}}}
    metadata = {"input_schema": "user-owned metadata", "_meta": {"keep": True}}
    tool = Tool(
        name="search",
        input_schema=parameters,
        output_schema=output,
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
        meta=metadata,
    )
    client = MCPClient()
    client.client = SimpleNamespace(list_tools=AsyncMock(return_value=[tool]))

    [action] = await client.list_actions()

    assert action["name"] == "search"
    assert action["input_schema"] == parameters
    assert action["output_schema"] == output
    assert action["annotations"]["read_only_hint"] is True
    assert action["annotations"]["open_world_hint"] is False
    assert action["meta"] == metadata
    assert {"inputSchema", "outputSchema", "_meta"}.isdisjoint(action)
