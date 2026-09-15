"""MCP client address defaults."""

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
