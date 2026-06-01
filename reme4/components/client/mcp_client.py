"""MCP client for ReMe services."""

import json
import os
from collections.abc import AsyncGenerator
from typing import Any

from fastmcp import Client
from fastmcp.client import SSETransport, StdioTransport, StreamableHttpTransport
from fastmcp.client.client import CallToolResult

from .base_client import BaseClient
from ..component_registry import R
from ...constants import REME_SERVICE_INFO, REME_DEFAULT_HOST, REME_DEFAULT_PORT

_TRANSPORT_MAP = {
    "sse": SSETransport,
    "stdio": StdioTransport,
    "streamable-http": StreamableHttpTransport,
}


@R.register("mcp")
class MCPClient(BaseClient):
    """MCP client that communicates with ReMe MCP service via fastmcp.Client.

    Usage:
        # SSE (default)
        client = MCPClient(host="localhost", port=8000)
        async with client:
            async for text in client(action="my_tool", query="hello"):
                print(text)

        # Streamable HTTP
        client = MCPClient(transport="streamable-http", host="localhost", port=8000)

        # Stdio
        client = MCPClient(transport="stdio", command="python", args=["server.py"])

        # Custom transport object
        from fastmcp.client import SSETransport
        client = MCPClient(transport=SSETransport(url="http://host:port/sse"))
    """

    def __init__(
        self,
        transport: str | Any = "sse",
        host: str | None = None,
        port: int | None = None,
        timeout: float = 30.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(transport, str) and transport not in _TRANSPORT_MAP:
            raise ValueError(f"Unknown transport: {transport!r}, expected one of {list(_TRANSPORT_MAP)}")

        if isinstance(transport, str) and transport != "stdio":
            if not (host and port):
                if service_info := os.environ.get(REME_SERVICE_INFO):
                    try:
                        data = json.loads(service_info)
                        host = data["host"]
                        port = data["port"]
                    except Exception:
                        self.logger.warning(f"Invalid service info: {service_info}")
                        host, port = REME_DEFAULT_HOST, REME_DEFAULT_PORT
                else:
                    host, port = REME_DEFAULT_HOST, REME_DEFAULT_PORT
            self.host = host
            self.port = port

        self.transport = transport
        self.timeout = timeout

    def _build_transport(self):
        if not isinstance(self.transport, str):
            return self.transport

        cls = _TRANSPORT_MAP[self.transport]

        if self.transport == "stdio":
            command = self.kwargs.get("command", "")
            args = self.kwargs.get("args", [])
            return cls(command=command, args=args)

        path = "/sse" if self.transport == "sse" else "/mcp"
        url = f"http://{self.host}:{self.port}{path}"
        return cls(url=url)

    # pylint: disable=unnecessary-dunder-call
    async def _start(self) -> None:
        if self.client is None:
            self.client = Client(self._build_transport(), timeout=self.timeout)
            await self.client.__aenter__()

    # pylint: disable=invalid-overridden-method
    async def _execute(self, action: str, payload: dict) -> AsyncGenerator[str, None]:
        if self.client is None:
            raise RuntimeError("Client not initialized. Call _start() first.")

        result: CallToolResult = await self.client.call_tool(action, payload)
        yield self._extract_text(result)

    async def list_actions(self) -> list[dict]:
        """Return raw MCP Tool dumps; each dict gets an `action` key (the tool name)."""
        if self.client is None:
            raise RuntimeError("Client not initialized. Call _start() first.")
        tools = await self.client.list_tools()
        return [tool.model_dump() for tool in tools]

    # pylint: disable=unnecessary-dunder-call
    async def _close(self) -> None:
        if self.client is not None:
            await self.client.__aexit__(None, None, None)
            self.client = None

    @staticmethod
    def _extract_text(result: CallToolResult) -> str:
        for block in result.content:
            if hasattr(block, "text"):
                return block.text
        return str(result.content)
