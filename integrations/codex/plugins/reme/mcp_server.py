#!/usr/bin/env python3
"""Codex MCP tools and native settings for the independently running ReMe service."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_context
from fastmcp.server.providers import Provider
from fastmcp.server.providers.proxy import ProxyClient, ProxyProvider
from fastmcp.tools import FunctionTool, ToolResult
from mcp.types import ToolAnnotations

from reme_config import DEFAULTS, load_config, settings_values, update_settings
from reme_runtime import ReMeRuntime
from reme_tools import LOCAL_TOOLS, RUN_DREAM, STATUS, register_tools

SETTINGS_READ = "reme_settings_read"
SETTINGS_UPDATE = "reme_settings_update"
RESERVED_TOOLS = LOCAL_TOOLS | {SETTINGS_READ, SETTINGS_UPDATE}

# OpenAI MCP Extensions node-v0.1.0, docs/spec.md#structured-settings.
# Use its legacy capability advertisement with ReMe's existing FastMCP runtime.
FIELDS = {
    "mcpUrl": {
        "type": "string",
        "title": "ReMe MCP address",
        "description": "Full URL of your running ReMe MCP service.",
    },
    "requestTimeoutMs": {"type": "integer", "title": "Request timeout (ms)", "minimum": 1000, "maximum": 120000},
    "backgroundTimeoutMs": {"type": "integer", "title": "Background timeout (ms)", "minimum": 1000, "maximum": 3600000},
    "shutdownTimeoutMs": {
        "type": "integer",
        "title": "Shutdown timeout (ms)",
        "minimum": 100,
        "maximum": 60000,
        "description": "Codex limits SessionEnd to 3 seconds; the plugin uses at most 2 seconds for exit delivery.",
    },
    "autoMemoryEnabled": {"type": "boolean", "title": "Automatic memory capture"},
    "autoMemoryInterval": {"type": "integer", "title": "Capture batch size", "minimum": 1, "maximum": 1000},
    "autoDreamEnabled": {"type": "boolean", "title": "Daily memory consolidation"},
    "dreamCron": {
        "type": "string",
        "title": "Auto Dream schedule",
        "description": "Daily cron: minute hour * * *. Runs while Codex keeps this MCP connection alive.",
    },
    "dreamHint": {
        "type": "string",
        "title": "Auto Dream hint",
        "description": "Guidance for scheduled and manual consolidation.",
    },
    "rootAgentsOnly": {"type": "boolean", "title": "Root agents only"},
    "language": {"type": "string", "title": "Memory guidance language", "enum": ["en", "zh"]},
    "autoRecall": {"type": "boolean", "title": "Automatic recall"},
    "searchLimit": {"type": "integer", "title": "Search result limit", "minimum": 1, "maximum": 50},
    "recallMinScore": {"type": "number", "title": "Minimum recall score", "minimum": 0},
    "timezone": {
        "type": "string",
        "title": "Workspace timezone",
        "description": "IANA timezone, such as Asia/Shanghai.",
    },
}
SCHEMA = {"type": "object", "properties": FIELDS, "additionalProperties": False}
LAYOUT = [
    {
        "kind": "group",
        "title": "Connection",
        "items": [
            {"kind": "property", "property": "mcpUrl"},
            {
                "kind": "tool",
                "tool": STATUS,
                "title": "ReMe status",
                "description": "Check the connection and inspect memory activity and the Dream schedule.",
            },
        ],
    },
    {
        "kind": "group",
        "title": "Automatic memory",
        "items": [
            {"kind": "property", "property": key}
            for key in ("autoMemoryEnabled", "autoMemoryInterval", "rootAgentsOnly", "language")
        ],
    },
    {
        "kind": "group",
        "title": "Auto Dream",
        "items": [
            *[
                {"kind": "property", "property": key}
                for key in ("autoDreamEnabled", "dreamCron", "dreamHint", "timezone")
            ],
            {"kind": "tool", "tool": RUN_DREAM, "title": "Consolidate now (updates memory files)"},
        ],
    },
    {
        "kind": "group",
        "title": "Recall",
        "items": [{"kind": "property", "property": key} for key in ("autoRecall", "searchLimit", "recallMinScore")],
    },
    {
        "kind": "group",
        "title": "Timeouts",
        "items": [
            {"kind": "property", "property": key}
            for key in ("requestTimeoutMs", "backgroundTimeoutMs", "shutdownTimeoutMs")
        ],
    },
]


class ReMeProvider(Provider):
    """Resolve tools against one settings snapshot, without caching across endpoints."""

    @staticmethod
    def snapshot(name: str = "") -> ProxyProvider:
        """Keep an in-flight tool's schema and execution on the same endpoint."""
        config = load_config()
        return ProxyProvider(
            lambda: ProxyClient(
                config["mcpUrl"],
                timeout=config["backgroundTimeoutMs" if name in {"auto_memory", "auto_dream"} else "requestTimeoutMs"]
                / 1000,
                init_timeout=config["requestTimeoutMs"] / 1000,
            ),
            cache_ttl=0,
        )

    async def _list_tools(self):
        return [tool for tool in await self.snapshot().list_tools() if tool.name not in RESERVED_TOOLS]

    async def _get_tool(self, name, version=None):
        # Local settings are authoritative and never depend on upstream availability.
        if name in RESERVED_TOOLS:
            return None
        return await self.snapshot(name).get_tool(name, version)

    async def get_tasks(self):
        """Proxy tools do not provide background MCP tasks."""
        return []


def create_server() -> FastMCP:
    """Keep settings available even when the configured upstream server is offline."""
    runtime = ReMeRuntime()

    @asynccontextmanager
    async def lifespan(_server):
        await runtime.start()
        try:
            yield {}
        finally:
            await runtime.close()

    server = FastMCP(
        "ReMe",
        lifespan=lifespan,
        experimental_capabilities={"openai/settings": {"readTool": SETTINGS_READ, "updateTool": SETTINGS_UPDATE}},
    )
    server.add_provider(ReMeProvider())
    register_tools(server)

    def read_settings() -> ToolResult:
        """Return editable values and their native settings layout."""
        values = settings_values(load_config())
        return ToolResult(
            content=[],
            structured_content={
                "schema": SCHEMA,
                "values": values,
                "layout": LAYOUT,
            },
        )

    async def save_settings(**arguments) -> ToolResult:
        """Persist a partial settings update and refresh upstream discovery."""
        values = update_settings(arguments["set"])
        runtime.settings_changed()
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

    assert FIELDS.keys() == DEFAULTS.keys()
    return server


if __name__ == "__main__":
    create_server().run(transport="stdio", show_banner=False)
