"""Native MCP settings, endpoint isolation, and MCP-only hook transport."""

# pylint: disable=missing-function-docstring,redefined-outer-name,protected-access

import asyncio
import importlib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from mcp.shared.exceptions import McpError

PLUGIN = Path(__file__).resolve().parents[2] / "integrations/codex/plugins/reme"


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(PLUGIN))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "host"))
    config = importlib.import_module("reme_config")
    server = importlib.import_module("mcp_server")
    transport = importlib.import_module("reme_mcp")
    return config, server, transport


@pytest.fixture
def upstreams(plugin, monkeypatch):
    config, server, _ = plugin
    calls = []
    servers = {}
    for name in ("first", "second"):
        upstream = FastMCP(name)

        def search(query: str, _name=name) -> str:
            calls.append((_name, query))
            return f"{_name}: daily/fact.md"

        upstream.tool(search)
        servers[f"http://{name}.test/mcp"] = upstream

    def client(url, **kwargs):
        return Client(servers[url], **kwargs)

    monkeypatch.setattr(server, "ProxyClient", client)
    config.update_settings({"mcp_url": "http://first.test/mcp"})
    return calls


@pytest.mark.asyncio
async def test_settings_remain_editable_offline(plugin, monkeypatch):
    config, module, _ = plugin
    connect = AsyncMock(side_effect=ConnectionError("offline"))
    monkeypatch.setattr(module.ReMeProvider, "_list_tools", connect)
    async with Client(module.create_server()) as client:
        capability = client.initialize_result.capabilities.experimental["openai/settings"]
        assert capability == {"readTool": module.SETTINGS_READ, "updateTool": module.SETTINGS_UPDATE}
        tools = {tool.name: tool for tool in await client.list_tools()}
        assert {module.SETTINGS_READ, module.SETTINGS_UPDATE, module.CHECK_CONNECTION} <= tools.keys()
        read = await client.call_tool(capability["readTool"], {})
        settings = read.structured_content
        assert settings["values"] == config.DEFAULTS
        assert settings["schema"]["properties"].keys() == config.DEFAULTS.keys()
        assert settings["layout"][0]["items"][-1]["tool"] == module.CHECK_CONNECTION
        assert tools[module.SETTINGS_READ].annotations.readOnlyHint
        assert tools[module.SETTINGS_READ].outputSchema
        assert not (config.data_dir() / "config.json").exists()
        saved = await client.call_tool(capability["updateTool"], {"set": {"auto_memory": False}})
        assert saved.structured_content["values"] == {**config.DEFAULTS, "auto_memory": False}
        assert not config.load_config()["auto_memory"]


@pytest.mark.asyncio
async def test_settings_patch_validation_and_migration(plugin, upstreams):
    config, module, _ = plugin
    path = config.data_dir() / "config.json"
    config.write_json(path, {"api_url": "http://first.test", "mcp_url": "http://first.test/mcp", "recall_limit": 9})
    before = path.read_bytes()
    async with Client(module.create_server()) as client:
        for changes in (
            {},
            {"bogus": 1},
            {"auto_memory": "false"},
            {"memory_interval": 0},
            {"mcp_url": "http://user:secret@first.test/mcp"},
            {"timezone": "Invalid/Zone"},
            {"request_timeout": 0},
        ):
            with pytest.raises(ToolError):
                await client.call_tool(module.SETTINGS_UPDATE, {"set": changes})
            assert path.read_bytes() == before
        await client.call_tool(module.SETTINGS_UPDATE, {"set": {"auto_recall": False}})
    assert config.load_config()["recall_limit"] == 9
    assert config.load_config()["endpoint"] == "http://first.test"
    assert "api_url" not in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600
    with config.lock(config.data_dir() / "config.lock"):
        with pytest.raises(RuntimeError, match="in progress"):
            config.update_settings({"auto_recall": True})
    assert not config.load_config()["auto_recall"]
    assert not upstreams


@pytest.mark.asyncio
async def test_live_endpoint_switch_and_inflight_snapshot(plugin, upstreams):
    config, module, _ = plugin
    server = module.create_server()
    provider = module.ReMeProvider()
    previous = await provider.get_tool("search")

    @server.tool
    async def captured_search(query: str):
        return await previous.run({"query": query})

    async with Client(server) as client:
        assert (await client.call_tool("search", {"query": "one"})).data == "first: daily/fact.md"
        await client.call_tool(module.SETTINGS_UPDATE, {"set": {"mcp_url": "http://second.test/mcp"}})
        assert (await client.call_tool("search", {"query": "two"})).data == "second: daily/fact.md"
        # A resolved call retains its original endpoint even when settings change before execution.
        await client.call_tool("captured_search", {"query": "in flight"})
    assert upstreams == [("first", "one"), ("second", "two"), ("first", "in flight")]
    assert config.load_config()["mcp_url"] == "http://second.test/mcp"


def test_custom_endpoint_does_not_alias_standard_or_legacy_queue(plugin):
    config, _, _ = plugin
    standard = config.validate_config({"mcp_url": "http://service.test/mcp"})
    custom = config.validate_config({"mcp_url": "http://service.test", "api_url": "http://service.test"})
    assert standard["endpoint"] == "http://service.test"
    assert custom["endpoint"] != standard["endpoint"]


@pytest.mark.asyncio
async def test_hooks_call_mcp_and_preserve_structured_answers(plugin, monkeypatch):
    config, _, transport = plugin
    upstream = FastMCP("test")
    calls = []

    @upstream.tool
    def search(query: str) -> dict:
        calls.append(query)
        return {"path": "daily/fact.md"}

    @upstream.tool
    def auto_memory() -> str:
        raise ToolError("write failed")

    @upstream.tool
    async def delayed() -> str:
        await asyncio.sleep(1)
        return "too late"

    monkeypatch.setattr("fastmcp.Client", lambda *args, **kwargs: Client(upstream, **kwargs))
    result = await transport.call_async(config.load_config(), "search", {"query": "fact"}, 2)
    assert result == {"success": True, "answer": {"path": "daily/fact.md"}}
    assert calls == ["fact"]
    with pytest.raises(ToolError, match="write failed"):
        await transport.call_async(config.load_config(), "auto_memory", {}, 2)
    with pytest.raises((TimeoutError, McpError)):
        await transport.call_async(config.load_config(), "delayed", {}, 0.05)


@pytest.mark.asyncio
async def test_connection_action_uses_current_endpoint_and_propagates_failure(plugin, upstreams, monkeypatch):
    config, module, _ = plugin
    call = AsyncMock(return_value={"success": True, "answer": "healthy"})
    monkeypatch.setattr(module, "call_async", call)
    async with Client(module.create_server()) as client:
        await client.call_tool(
            module.SETTINGS_UPDATE,
            {"set": {"mcp_url": "http://second.test/mcp", "recall_timeout": 3}},
        )
        assert (await client.call_tool(module.CHECK_CONNECTION, {})).data == "healthy"
        assert call.call_args.args == (config.load_config(), "health_check", {}, 3)
        call.side_effect = ConnectionError("offline")
        with pytest.raises(ToolError, match="offline"):
            await client.call_tool(module.CHECK_CONNECTION, {})
    assert not upstreams
