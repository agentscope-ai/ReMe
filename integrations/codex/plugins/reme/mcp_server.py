#!/usr/bin/env python3
"""Codex MCP tools and native settings for the independently running ReMe service."""

from __future__ import annotations

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_context
from fastmcp.server.providers import Provider
from fastmcp.server.providers.proxy import ProxyClient, ProxyProvider
from fastmcp.tools import FunctionTool, ToolResult
from mcp.types import ToolAnnotations

from reme_config import DEFAULTS, load_config, settings_values, update_settings
from reme_mcp import call_async

SETTINGS_READ = "reme_settings_read"
SETTINGS_UPDATE = "reme_settings_update"
CHECK_CONNECTION = "reme_check_connection"

# OpenAI MCP Extensions node-v0.1.0, docs/spec.md#structured-settings.
# Use its legacy capability advertisement with ReMe's existing FastMCP runtime.
FIELDS = {
    "mcp_url": {"type": "string", "title": "ReMe MCP address", "description": "URL of your running ReMe MCP service."},
    "auto_recall": {"type": "boolean", "title": "Automatic recall"},
    "auto_memory": {
        "type": "boolean",
        "title": "Automatic memory",
        "description": "Send completed user/assistant turns to ReMe. Disabling also pauses queued retries.",
    },
    "memory_interval": {"type": "integer", "title": "Turns per memory batch", "minimum": 1},
    "recall_limit": {"type": "integer", "title": "Recall result limit", "minimum": 1},
    "recall_min_score": {"type": "number", "title": "Minimum recall score", "minimum": 0},
    "context_max_chars": {"type": "integer", "title": "Recall character limit", "minimum": 1},
    "recall_timeout": {
        "type": "number",
        "title": "Recall timeout (seconds)",
        "minimum": 0,
        "description": "Must be greater than zero.",
        "maximum": 10,
    },
    "request_timeout": {
        "type": "number",
        "title": "Memory / tool timeout (seconds)",
        "minimum": 0,
        "description": "Must be greater than zero.",
        "maximum": 600,
    },
    "shutdown_timeout": {
        "type": "number",
        "title": "Exit flush budget (seconds)",
        "minimum": 0,
        "description": "Must be greater than zero.",
        "maximum": 2,
    },
    "timezone": {"type": "string", "title": "Memory timezone", "description": "IANA timezone, such as Asia/Shanghai."},
}
SCHEMA = {"type": "object", "properties": FIELDS, "additionalProperties": False}
LAYOUT = [
    {
        "kind": "group",
        "title": "Connection",
        "items": [
            {"kind": "property", "property": "mcp_url"},
            {"kind": "tool", "tool": CHECK_CONNECTION, "title": "Check connection"},
        ],
    },
    {
        "kind": "group",
        "title": "Automatic memory",
        "items": [
            {"kind": "property", "property": key}
            for key in ("auto_recall", "auto_memory", "memory_interval", "timezone")
        ],
    },
    {
        "kind": "group",
        "title": "Recall",
        "items": [
            {"kind": "property", "property": key} for key in ("recall_limit", "recall_min_score", "context_max_chars")
        ],
    },
    {
        "kind": "group",
        "title": "Timeouts",
        "items": [
            {"kind": "property", "property": key} for key in ("recall_timeout", "request_timeout", "shutdown_timeout")
        ],
    },
]


class ReMeProvider(Provider):
    """Resolve tools against one settings snapshot, without caching across endpoints."""

    @staticmethod
    def snapshot() -> ProxyProvider:
        """Keep an in-flight tool's schema and execution on the same endpoint."""
        config = load_config()
        return ProxyProvider(
            lambda: ProxyClient(
                config["mcp_url"],
                timeout=config["request_timeout"],
                init_timeout=config["recall_timeout"],
            ),
            cache_ttl=0,
        )

    async def _list_tools(self):
        return [
            tool
            for tool in await self.snapshot().list_tools()
            if tool.name not in {SETTINGS_READ, SETTINGS_UPDATE, CHECK_CONNECTION}
        ]

    async def _get_tool(self, name, version=None):
        # Local settings are authoritative and never depend on upstream availability.
        if name in {SETTINGS_READ, SETTINGS_UPDATE, CHECK_CONNECTION}:
            return None
        return await self.snapshot().get_tool(name, version)

    async def get_tasks(self):
        """Proxy tools do not provide background MCP tasks."""
        return []


def create_server() -> FastMCP:
    """Keep settings available even when the configured upstream server is offline."""
    server = FastMCP(
        "ReMe",
        experimental_capabilities={"openai/settings": {"readTool": SETTINGS_READ, "updateTool": SETTINGS_UPDATE}},
    )
    server.add_provider(ReMeProvider())

    def read_settings() -> ToolResult:
        """Return editable values and their native settings layout."""
        return ToolResult(
            content=[],
            structured_content={
                "schema": SCHEMA,
                "values": settings_values(load_config()),
                "layout": LAYOUT,
            },
        )

    async def save_settings(**arguments) -> ToolResult:
        """Persist a partial settings update and refresh upstream discovery."""
        values = update_settings(arguments["set"])
        await get_context().session.send_tool_list_changed()
        return ToolResult(content=[], structured_content={"values": values})

    server.add_tool(
        FunctionTool(
            name=SETTINGS_READ,
            title="ReMe settings",
            description="Read ReMe plugin settings for the host settings UI.",
            fn=read_settings,
            parameters={"type": "object", "additionalProperties": False},
            output_schema={
                "type": "object",
                "properties": {
                    "schema": {"type": "object"},
                    "values": SCHEMA,
                    "layout": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["schema", "values"],
            },
            annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
        ),
    )
    server.add_tool(
        FunctionTool(
            name=SETTINGS_UPDATE,
            title="Save ReMe settings",
            description="Save changes requested in the ReMe settings UI.",
            fn=save_settings,
            parameters={
                "type": "object",
                "properties": {
                    "set": {**SCHEMA, "minProperties": 1},
                },
                "required": ["set"],
                "additionalProperties": False,
            },
            output_schema={"type": "object", "properties": {"values": SCHEMA}, "required": ["values"]},
            annotations=ToolAnnotations(
                readOnlyHint=False,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
        ),
    )

    @server.tool(name=CHECK_CONNECTION, title="Check ReMe connection", annotations={"readOnlyHint": True})
    async def check_connection() -> str:
        """Check the saved MCP endpoint without sending conversation content."""
        config = load_config()
        result = await call_async(config, "health_check", {}, config["recall_timeout"])
        return str(result["answer"])

    assert FIELDS.keys() == DEFAULTS.keys()
    return server


if __name__ == "__main__":
    create_server().run(transport="stdio", show_banner=False)
