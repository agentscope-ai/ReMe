#!/usr/bin/env python3
"""Codex MCP tools and native settings for the independently running ReMe service."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastmcp import FastMCP
from fastmcp.server.providers import Provider
from fastmcp.server.providers.proxy import ProxyClient, ProxyProvider

import reme_config
from reme_config import DEFAULTS, load_config
from reme_settings import FIELDS, SETTINGS_READ, SETTINGS_UPDATE, register_settings
from reme_runtime import ReMeRuntime
from reme_tools import LOCAL_TOOLS, RUN_DREAM, STATUS, register_tools

__all__ = ["create_server", "RUN_DREAM", "STATUS"]

# All consolidation must pass through reme_run_dream's profile/endpoint lock.
RESERVED_TOOLS = LOCAL_TOOLS | {SETTINGS_READ, SETTINGS_UPDATE, "auto_dream"}


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

    register_settings(server, reme_config, settings_changed=runtime.settings_changed)

    assert FIELDS.keys() == DEFAULTS.keys()
    return server


if __name__ == "__main__":
    create_server().run(transport="stdio", show_banner=False)
