"""Native settings on the shared HTTP server, without a per-session proxy."""

# pylint: disable=missing-function-docstring,redefined-outer-name

from types import SimpleNamespace
import asyncio
import socket

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from integrations.codex.plugins.reme import reme_config
from reme.components.service import HttpService
from reme.components.job import BaseJob


@pytest.fixture
def shared(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    path = reme_config.data_dir() / "config.json"

    async def lifecycle():
        pass

    service = HttpService(codex_settings_file=str(path))
    service.build_service(SimpleNamespace(config=SimpleNamespace(app_name="test"), start=lifecycle, close=lifecycle))
    return service, path


@pytest.mark.asyncio
async def test_gui_updates_the_file_read_by_hooks_and_other_connections(shared):
    service, path = shared
    async with Client(service.mcp_server) as gui, Client(service.mcp_server) as other:
        assert gui.server_capabilities.experimental["openai/settings"] == {
            "readTool": "reme_settings_read",
            "updateTool": "reme_settings_update",
        }
        before = (await gui.call_tool("reme_settings_read", {})).structured_content
        assert before["values"]["searchLimit"] == 5
        assert not path.exists()
        await gui.call_tool("reme_settings_update", {"set": {"searchLimit": 9, "autoRecall": False}})
        after = (await other.call_tool("reme_settings_read", {})).structured_content
        assert after["values"]["searchLimit"] == 9
        assert not reme_config.load_config()["autoRecall"]
        assert path.stat().st_mode & 0o777 == 0o600
        assert "mcpUrl" not in before["schema"]["properties"]
        assert "autoDreamEnabled" not in before["schema"]["properties"]
        assert "autoDreamEnabled" in reme_config.load_config()  # Existing profile fields survive a patch.


@pytest.mark.asyncio
@pytest.mark.parametrize("patch", [{"searchLimit": 0}, {"timezone": "Invalid/Zone"}, {"mcpUrl": "http://other/mcp"}])
async def test_invalid_or_connection_settings_do_not_replace_the_profile(shared, patch):
    service, path = shared
    reme_config.write_json(path, {"autoRecall": True})
    before = path.read_bytes()
    async with Client(service.mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("reme_settings_update", {"set": patch})
    assert path.read_bytes() == before


@pytest.mark.parametrize("options", [{"host": "0.0.0.0"}, {"host": "192.0.2.1"}, {"mcp_enabled": False}])
def test_local_profile_tools_require_an_explicit_loopback_mcp_listener(tmp_path, options):
    with pytest.raises(ValueError, match="loopback"):
        HttpService(codex_settings_file=str(tmp_path / "config.json"), **options)


def test_default_http_service_does_not_advertise_local_settings():
    service = HttpService()
    assert service.codex_settings_file is None


def test_jobs_cannot_replace_the_native_settings_tools(shared):
    service, _ = shared
    job = BaseJob(name="reme_settings_update")
    with pytest.raises(ValueError, match="native settings"):
        service.add_jobs(SimpleNamespace(context=SimpleNamespace(jobs={job.name: job})))


@pytest.mark.asyncio
async def test_two_http_clients_share_settings_on_the_existing_mcp_endpoint(shared):
    """Exercise HTTP initialization, native settings discovery, persistence and concurrent sessions."""
    import uvicorn

    service, path = shared
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(service.service, log_level="error"))
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(5):
                while not server.started:
                    if serving.done():
                        await serving
                    await asyncio.sleep(0.01)
            async with (
                Client(f"http://127.0.0.1:{port}/mcp") as first,
                Client(
                    f"http://127.0.0.1:{port}/mcp",
                ) as second,
            ):
                capability = first.server_capabilities.experimental["openai/settings"]
                await first.call_tool(capability["updateTool"], {"set": {"searchLimit": 11}})
                result = await second.call_tool(capability["readTool"], {})
                assert result.structured_content["values"]["searchLimit"] == 11
                assert reme_config.load_config(path)["searchLimit"] == 11
        finally:
            server.should_exit = True
            await asyncio.wait_for(serving, 5)
