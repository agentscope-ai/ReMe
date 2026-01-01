"""Module for managing Model Context Protocol (MCP) server connections."""

import os
import re
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, TextContent

from ..schema import ToolCall


class MCPClient:
    """A client manager for handling multiple MCP transport protocols."""

    def __init__(self, config: dict):
        """Initialize the client with server configuration."""
        self.config = config

    @staticmethod
    def _infer_transport_type(cfg: dict[str, Any]) -> str:
        """Infer the transport type based on configuration fields."""
        if "command" in cfg:
            return "stdio"

        if "url" in cfg:
            url = cfg["url"].lower()
            if url.endswith("/sse") or "sse" in url:
                return "sse"
            return "streamable-http"

        raise ValueError(f"Could not infer transport type for: {cfg}")

    def _replace_env_vars(self, data: str | dict | list) -> Any:
        """Replace environment variable placeholders in configuration."""
        if isinstance(data, str):
            return re.sub(r"\$\{(\w+)\}", lambda m: os.getenv(m.group(1), m.group(0)), data)
        if isinstance(data, dict):
            return {k: self._replace_env_vars(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._replace_env_vars(i) for i in data]
        return data

    @asynccontextmanager
    async def _get_transport(self, cfg: dict[str, Any]):
        """Context manager to yield the appropriate MCP transport."""
        # Pop 'type' if present, otherwise infer it
        t_type = cfg.pop("type", None) or self._infer_transport_type(cfg)

        try:
            if t_type == "stdio":
                params = StdioServerParameters(
                    command=cfg["command"],
                    args=cfg.get("args", []),
                    env=cfg.get("env", None),
                )
                async with stdio_client(params) as transport:
                    yield transport
            elif t_type == "sse":
                async with sse_client(**cfg) as transport:
                    yield transport
            elif t_type == "streamable-http":
                async with streamablehttp_client(**cfg) as transport:
                    yield transport
            else:
                raise NotImplementedError(f"Unsupported transport: {t_type}")
        finally:
            pass  # Ensure proper cleanup

    @asynccontextmanager
    async def connect_to_server(self, server_name: str):
        """Establish a session with the specified MCP server."""
        server_config = self.config.get("mcpServers", {}).get(server_name)
        if not server_config:
            raise ValueError(f"Config for '{server_name}' not found.")

        # Process environment variables and transport selection
        cfg = self._replace_env_vars(server_config)

        async with self._get_transport(cfg) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    async def list_tools(self, server_name: str) -> list[Tool]:
        """Retrieve available tools from a specific server."""
        async with self.connect_to_server(server_name) as session:
            result = await session.list_tools()
            return result.tools

    async def list_tool_calls(self, server_name: str, return_dict: bool = True) -> list[dict | ToolCall]:
        """Retrieve available tools from a specific server."""
        tools = await self.list_tools(server_name)
        tool_calls: list[ToolCall] = [ToolCall.from_mcp_tool(tool) for tool in tools]
        if return_dict:
            return [tool_call.simple_input_dump() for tool_call in tool_calls]

        return tool_calls

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
        parse_text_result: bool = False,
    ) -> CallToolResult | str:
        """Execute a tool on a specific server."""
        async with self.connect_to_server(server_name) as session:
            tool_results: CallToolResult = await session.call_tool(tool_name, arguments)
            if not parse_text_result:
                return tool_results

            text_result = []
            for block in tool_results.content:
                if isinstance(block, TextContent):
                    text_result.append(block.text)
            return "\n".join(text_result)
