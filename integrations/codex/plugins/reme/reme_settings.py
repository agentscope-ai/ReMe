"""Native settings tools shared by the local plugin and the ReMe HTTP service."""

from __future__ import annotations

from pathlib import Path

from fastmcp.server.dependencies import get_context
from fastmcp.tools import FunctionTool, ToolResult
from mcp.types import ToolAnnotations

SETTINGS_READ = "reme_settings_read"
SETTINGS_UPDATE = "reme_settings_update"
STATUS = "reme_status"
RUN_DREAM = "reme_run_dream"

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


def register_settings(server, config, *, path: Path | None = None, settings_changed=None) -> None:
    """Use the existing profile file; a shared server does not control its clients' connections or timers."""
    shared = path is not None
    excluded = {"mcpUrl", "autoDreamEnabled", "dreamCron", "dreamHint"} if shared else set()
    fields = {key: value for key, value in FIELDS.items() if key not in excluded}
    if shared:
        for key in ("requestTimeoutMs", "backgroundTimeoutMs", "shutdownTimeoutMs"):
            fields[key] = {**fields[key], "title": "Hook " + fields[key]["title"].lower()}
        fields["timezone"] = {**fields["timezone"], "title": "Conversation timezone"}
    schema = {**SCHEMA, "properties": fields}
    layout = []
    for group in LAYOUT:
        items = [
            item
            for item in group["items"]
            if (item["kind"] == "property" and item["property"] in fields) or (item["kind"] == "tool" and not shared)
        ]
        if items:
            layout.append({**group, "items": items})

    def read_settings() -> ToolResult:
        values = config.load_config(path, check_connection=False) if shared else config.load_config()
        return ToolResult(
            content=[],
            structured_content={"schema": schema, "values": {key: values[key] for key in fields}, "layout": layout},
        )

    async def save_settings(**arguments) -> ToolResult:
        changes = arguments["set"]
        if changes.keys() - fields.keys():
            raise ValueError("This setting is managed by the native MCP connection or the ReMe service")
        values = (
            config.update_settings(changes, path, check_connection=False) if shared else config.update_settings(changes)
        )
        if settings_changed is not None:
            settings_changed()
        if not shared:
            await get_context().session.send_tool_list_changed()
        return ToolResult(content=[], structured_content={"values": {key: values[key] for key in fields}})

    server.add_tool(
        FunctionTool(
            name=SETTINGS_READ,
            title="ReMe settings",
            description="Read ReMe plugin settings for the host settings UI.",
            fn=read_settings,
            parameters={"type": "object", "additionalProperties": False},
            output_schema={
                "type": "object",
                "properties": {"schema": {"type": "object"}, "values": schema, "layout": {"type": "array"}},
                "required": ["schema", "values"],
            },
            annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
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
                "properties": {"set": schema},
                "required": ["set"],
                "additionalProperties": False,
            },
            output_schema={"type": "object", "properties": {"values": schema}, "required": ["values"]},
            annotations=ToolAnnotations(
                read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False
            ),
        ),
    )
