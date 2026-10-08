"""Native MCP settings, endpoint isolation, and MCP-only hook transport."""

# pylint: disable=missing-function-docstring,redefined-outer-name,protected-access

import asyncio
import importlib
import json
import shutil
from datetime import datetime, timezone
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
    runtime = importlib.import_module("reme_runtime")
    # Protocol tests must never start a scheduled network call at the wall-clock boundary.
    monkeypatch.setattr(runtime, "next_daily_run", lambda *_: datetime(2099, 1, 1, tzinfo=timezone.utc))
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
    config.update_settings({"mcpUrl": "http://first.test/mcp"})
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
        assert {module.SETTINGS_READ, module.SETTINGS_UPDATE, module.STATUS} <= tools.keys()
        assert "reme_check_connection" not in tools
        read = await client.call_tool(capability["readTool"], {})
        settings = read.structured_content
        assert settings["values"] == config.DEFAULTS
        assert settings["schema"]["properties"].keys() == config.DEFAULTS.keys()
        assert settings["layout"][0]["items"][1]["tool"] == module.STATUS
        assert len(settings["layout"][0]["items"]) == 2
        assert tools[module.SETTINGS_READ].annotations.readOnlyHint
        assert tools[module.SETTINGS_READ].outputSchema
        assert not (config.data_dir() / "config.json").exists()
        saved = await client.call_tool(capability["updateTool"], {"set": {"autoMemoryEnabled": False}})
        assert saved.structured_content["values"] == {**config.DEFAULTS, "autoMemoryEnabled": False}
        assert not config.load_config()["autoMemoryEnabled"]


@pytest.mark.asyncio
async def test_settings_patch_validation_and_persistence(plugin, upstreams):
    config, module, _ = plugin
    path = config.data_dir() / "config.json"
    config.write_json(path, {"mcpUrl": "http://first.test/mcp", "searchLimit": 9})
    before = path.read_bytes()
    async with Client(module.create_server()) as client:
        for changes in (
            {},
            {"bogus": 1},
            {"autoMemoryEnabled": "false"},
            {"autoMemoryInterval": 0},
            {"mcpUrl": "http://user:secret@first.test/mcp"},
            {"timezone": "Invalid/Zone"},
            {"backgroundTimeoutMs": 0},
            {"language": "fr"},
            {"searchLimit": 51},
            {"autoMemoryInterval": 1001},
            {"api_url": "http://first.test"},
        ):
            with pytest.raises(ToolError):
                await client.call_tool(module.SETTINGS_UPDATE, {"set": changes})
            assert path.read_bytes() == before
        await client.call_tool(module.SETTINGS_UPDATE, {"set": {"autoRecall": False}})
    assert config.load_config()["searchLimit"] == 9
    assert config.load_config()["endpoint"] == "http://first.test"
    assert "api_url" not in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600
    with config.lock(config.data_dir() / "config.lock"):
        with pytest.raises(RuntimeError, match="in progress"):
            config.update_settings({"autoRecall": True})
    assert not config.load_config()["autoRecall"]
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
        await client.call_tool(module.SETTINGS_UPDATE, {"set": {"mcpUrl": "http://second.test/mcp"}})
        assert (await client.call_tool("search", {"query": "two"})).data == "second: daily/fact.md"
        # A resolved call retains its original endpoint even when settings change before execution.
        await client.call_tool("captured_search", {"query": "in flight"})
    assert upstreams == [("first", "one"), ("second", "two"), ("first", "in flight")]
    assert config.load_config()["mcpUrl"] == "http://second.test/mcp"


def test_custom_endpoint_does_not_alias_standard_queue(plugin):
    config, _, _ = plugin
    standard = config.validate_config({"mcpUrl": "http://service.test/mcp"})
    custom = config.validate_config({"mcpUrl": "http://service.test"})
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
async def test_status_checks_current_connection_and_reports_failure_without_hiding_local_state(
    plugin,
    upstreams,
    monkeypatch,
):
    config, module, _ = plugin
    call = AsyncMock(return_value={"success": True, "answer": "healthy"})
    monkeypatch.setattr(importlib.import_module("reme_tools"), "call_async", call)
    async with Client(module.create_server()) as client:
        await client.call_tool(
            module.SETTINGS_UPDATE,
            {"set": {"mcpUrl": "http://second.test/mcp", "requestTimeoutMs": 3000}},
        )
        result = await client.call_tool(module.STATUS, {})
        assert result.structured_content["service"]["health_check"] == {"reachable": True, "answer": "healthy"}
        assert {args.args[1] for args in call.call_args_list} == {"health_check", "status"}
        assert all(args.args[0] == config.load_config() and args.args[2:] == ({}, 3) for args in call.call_args_list)
        assert result.structured_content["checked_at"] > 0
        assert result.structured_content["language"] == "en"
        call.side_effect = ConnectionError("offline")
        offline = (await client.call_tool(module.STATUS, {})).structured_content
        assert offline["service"]["health_check"] == {"reachable": False, "error": "ConnectionError"}
        assert offline["auto_memory"]["queued_turns"] == 0
    assert not upstreams


@pytest.mark.asyncio
async def test_status_opens_a_self_contained_fullscreen_app_resource(plugin):
    _, module, _ = plugin
    async with Client(module.create_server()) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        ui = tools["reme_status"].meta["ui"]
        assert ui["visibility"] == ["app", "model"]
        content = (await client.read_resource(ui["resourceUri"]))[0]
        assert content.mimeType == "text/html;profile=mcp-app"
        assert content.meta["openai/ui"] == {
            "preferredDisplayMode": "fullscreen",
            "availableDisplayModes": ["fullscreen"],
        }
        assert content.meta["ui"]["csp"] == {"connectDomains": [], "resourceDomains": []}
        assert 'id="refresh"' in content.text
        assert "ui/initialize" in content.text
        assert "ui/notifications/tool-result" in content.text
        assert "/* STATUS_APP */" not in content.text
        assert "<script src=" not in content.text


@pytest.mark.asyncio
async def test_status_survives_removal_of_its_installation_cache(plugin, monkeypatch, tmp_path):
    config, module, _ = plugin
    features = importlib.import_module("reme_tools")
    cache = tmp_path / "plugin-cache/old-version"
    (cache / "ui").mkdir(parents=True)
    (cache / ".codex-plugin").mkdir()
    html = (PLUGIN / "ui/status.html").read_text(encoding="utf-8")
    (cache / "ui/status.html").write_text(html, encoding="utf-8")
    (cache / ".codex-plugin/plugin.json").write_text(json.dumps({"version": "old-version"}), encoding="utf-8")
    monkeypatch.setattr(features, "__file__", str(cache / "reme_tools.py"))
    monkeypatch.setattr(features, "call_async", AsyncMock(return_value={"answer": "healthy"}))
    server = module.create_server()

    async with Client(server) as client:
        shutil.rmtree(cache)  # Codex can remove the old cache while its MCP connection is still alive.
        content = (await client.read_resource(features.STATUS_URI))[0]
        assert content.text == html
        status = (await client.call_tool(module.STATUS, {})).structured_content
        assert status["plugin_version"] == "old-version"
        assert status["service"]["health_check"] == {"reachable": True, "answer": "healthy"}
        await client.call_tool(module.SETTINGS_UPDATE, {"set": {"language": "zh"}})
        assert (await client.call_tool(module.STATUS, {})).structured_content["language"] == "zh"
        assert config.load_config()["language"] == "zh"


@pytest.mark.asyncio
async def test_focused_search_uses_live_defaults_validates_input_and_reports_empty(plugin, monkeypatch):
    config, module, _ = plugin
    features = importlib.import_module("reme_tools")
    call = AsyncMock(return_value={"answer": "daily/fact.md: review on Thursday"})
    monkeypatch.setattr(features, "call_async", call)
    config.update_settings({"searchLimit": 7, "recallMinScore": 0.25, "language": "zh"})
    async with Client(module.create_server()) as client:
        result = await client.call_tool("reme_search", {"query": " decision "})
        assert result.data == "daily/fact.md: review on Thursday"
        assert call.call_args.args[2] == {"query": "decision", "limit": 7, "min_score": 0.25}
        await client.call_tool("reme_search", {"query": "decision", "limit": 2, "min_score": 0})
        assert call.call_args.args[2] == {"query": "decision", "limit": 2, "min_score": 0}
        for arguments in ({"query": " "}, {"query": "a", "limit": 51}, {"query": "a", "min_score": -1}):
            with pytest.raises(ToolError):
                await client.call_tool("reme_search", arguments)
        call.return_value = {"answer": ""}
        assert (await client.call_tool("reme_search", {"query": "missing"})).data == "没有找到相关记忆。"
        call.side_effect = ConnectionError("private service details")
        with pytest.raises(ToolError, match="ReMe search failed"):
            await client.call_tool("reme_search", {"query": "private query"})
    log = (config.data_dir() / "hooks.log").read_text()
    assert "search_failed" in log and "ConnectionError" in log
    assert all(secret not in log for secret in ("private", "Thursday", "decision"))


@pytest.mark.asyncio
async def test_status_is_useful_offline_and_excludes_queue_contents(plugin, monkeypatch):
    config, module, _ = plugin
    state = importlib.import_module("reme_state")
    features = importlib.import_module("reme_tools")
    current = config.load_config()
    queued = state.queue_root(current) / "session-one"
    config.write_json(queued / "one.json", {"private": "user conversation"})
    config.write_json(queued / "two.json", {"private": "acknowledged"})
    config.write_json(queued / "two.done", {})
    other = config.validate_config({"mcpUrl": "http://another.test/mcp"})
    config.write_json(state.queue_root(other) / "other/three.json", {})
    state.log_status("recall_found", config=current)
    state.log_status("memory_failed", RuntimeError("secret error text"), config=current)
    state.log_status("dream_completed", config=other)
    monkeypatch.setattr(features, "call_async", AsyncMock(side_effect=ConnectionError("private endpoint error")))
    async with Client(module.create_server()) as client:
        result = await client.call_tool("reme_status", {})
        snapshot = result.structured_content
        assert snapshot["auto_memory"]["queued_turns"] == snapshot["auto_memory"]["queued_sessions"] == 1
        assert snapshot["auto_memory"]["other_endpoint_queued_turns"] == 1
        assert snapshot["service"]["health_check"] == {"reachable": False, "error": "ConnectionError"}
        assert snapshot["auto_dream"]["next_run_at"]
        assert snapshot["auto_dream"]["cron"] == "0 23 * * *"
        assert snapshot["auto_dream"]["scheduler"] == "codex_mcp"
        assert [row["event"] for row in snapshot["recent_activity"]] == ["recall_found", "memory_failed"]
        assert "queued: 1 turns" in result.content[0].text
        assert "private" not in json.dumps(snapshot)
    assert (queued / "one.json").exists()  # Reading status never acknowledges or removes pending work.


@pytest.mark.asyncio
async def test_manual_dream_uses_saved_settings_serializes_processes_and_tracks_result(plugin, monkeypatch):
    config, module, _ = plugin
    state = importlib.import_module("reme_state")
    features = importlib.import_module("reme_runtime")
    config.update_settings({"dreamHint": "merge project decisions"})
    current = config.load_config()
    started, release = asyncio.Event(), asyncio.Event()

    async def dream(settings, action, payload, timeout):
        assert settings == current and action == "auto_dream" and timeout == 3600
        assert payload == {"date": "2026-09-30", "hint": "merge project decisions"}
        started.set()
        await release.wait()
        return {"answer": "Consolidated into digest/project.md"}

    monkeypatch.setattr(features, "call_async", dream)
    async with Client(module.create_server()) as first, Client(module.create_server()) as second:
        tools = {tool.name: tool for tool in await first.list_tools()}
        assert tools["reme_run_dream"].annotations.destructiveHint
        task = asyncio.create_task(first.call_tool("reme_run_dream", {"date": "2026-09-30"}))
        await asyncio.wait_for(started.wait(), 2)
        assert state.local_status(current)["auto_dream"]["last_run"]["status"] == "running"
        with pytest.raises(ToolError, match="already running"):
            await second.call_tool("reme_run_dream", {})
        release.set()
        assert "digest/project.md" in (await task).data
    run = state.local_status(current)["auto_dream"]["last_run"]
    assert run["status"] == "completed" and run["completed_at"] >= run["started_at"]
    assert "project decisions" not in (state.endpoint_dir(current) / "dream.json").read_text()


@pytest.mark.asyncio
async def test_failed_dream_keeps_failure_state_and_releases_lock(plugin, monkeypatch):
    config, module, _ = plugin
    state = importlib.import_module("reme_state")
    features = importlib.import_module("reme_runtime")
    call = AsyncMock(side_effect=TimeoutError("sensitive details"))
    monkeypatch.setattr(features, "call_async", call)
    async with Client(module.create_server()) as client:
        for value in ("2026-02-30", "20260930"):
            with pytest.raises(ToolError):
                await client.call_tool("reme_run_dream", {"date": value})
        call.assert_not_called()
        with pytest.raises(ToolError):
            await client.call_tool("reme_run_dream", {})
        run = state.local_status(config.load_config())["auto_dream"]["last_run"]
        assert run["status"] == "failed" and run["error"] == "TimeoutError"
        call.side_effect = None
        call.return_value = {"answer": "done"}
        assert (await client.call_tool("reme_run_dream", {"hint": "one-off"})).data == "done"
        assert call.call_args.args[2]["hint"] == "one-off"


def test_diagnostics_rotate_and_detect_interrupted_dream(plugin, monkeypatch):
    config, _, _ = plugin
    state = importlib.import_module("reme_state")
    current = config.load_config()
    monkeypatch.setattr(state, "LOG_BYTES", 800)
    for _ in range(30):
        state.log_status("memory_saved", config=current, turns=5)
    assert (config.data_dir() / "hooks.log.1").exists()
    rows = state.recent_activity(current, limit=3)
    assert len(rows) == 3 and all(row["turns"] == 5 for row in rows)
    assert (config.data_dir() / "hooks.log").stat().st_mode & 0o777 == 0o600
    path = state.endpoint_dir(current) / "dream.json"
    config.write_json(path, {"status": "running", "started_at": 1})
    assert state.local_status(current)["auto_dream"]["last_run"]["status"] == "interrupted"
    assert json.loads(path.read_text())["status"] == "running"  # Status does not rewrite history.


@pytest.mark.asyncio
async def test_reme_job_failures_reach_plugin_when_service_error_flag_is_enabled(plugin, monkeypatch):
    from reme.components.service.mcp_tools import add_mcp_job
    from reme.schema import Response

    class FailingJob:
        """Exercise the real service's MCP error contract without a model call."""

        name = "auto_memory"
        description = "test memory job"
        parameters = {"type": "object", "properties": {}}

        async def __call__(self, **_kwargs):
            return Response(success=False, answer="memory extraction failed")

    config, _, transport = plugin
    upstream = FastMCP("real-error-contract")
    add_mcp_job(upstream, FailingJob(), injected_job_kwargs={}, tool_error_on_failure=True)
    monkeypatch.setattr("fastmcp.Client", lambda *args, **kwargs: Client(upstream, **kwargs))
    with pytest.raises(ToolError, match="memory extraction failed"):
        await transport.call_async(config.load_config(), "auto_memory", {}, 2)
