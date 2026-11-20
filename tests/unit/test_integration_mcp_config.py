"""User-owned plugin installation sources and transport selection, without host mutations."""

# pylint: disable=missing-function-docstring,redefined-outer-name

import importlib.util
import json
from pathlib import Path
import shlex
import shutil
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "integrations/configure_mcp.py"


@pytest.fixture(params=["codex", "claude-code"])
def setup(request, tmp_path, monkeypatch):
    host = request.param
    home = tmp_path / "host with spaces"
    monkeypatch.setenv("CODEX_HOME" if host == "codex" else "CLAUDE_CONFIG_DIR", str(home))
    spec = importlib.util.spec_from_file_location("reme_configure_mcp", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = home / "reme/config.json"
    settings.parent.mkdir(parents=True)
    key = "mcpUrl" if host == "codex" else "mcp_url"
    return module, host, settings, key


def connection(module, host, installation):
    return json.loads((installation / module.HOSTS[host][1] / ".mcp.json").read_text())["mcpServers"]["reme"]


def test_http_uses_saved_endpoint_and_preserves_local_data(setup):
    module, host, settings, key = setup
    values = (
        {key: "https://example.com/memory/mcp", "language": "zh"}
        if host == "codex"
        else {key: "https://example.com/memory/mcp", "memory_interval": 1}
    )
    settings.write_text(json.dumps(values))
    before = settings.read_bytes()
    queue = settings.parent / "queue/session/pending.json"
    queue.parent.mkdir(parents=True)
    queue.write_text("private conversation")
    directory, plugin_path, market_path, *_ = module.HOSTS[host]
    source = module.ROOT / directory / plugin_path / ".mcp.json"
    source_before = source.read_bytes()

    installation = module.configure(host)

    mcp = connection(module, host, installation)
    assert mcp["type"] == "http" and mcp["url"] == values[key]
    assert "command" not in mcp and "args" not in mcp
    assert settings.read_bytes() == before and queue.read_text() == "private conversation"
    assert not list(installation.rglob("pending.json"))
    assert source.read_bytes() == source_before
    assert (installation / market_path).is_file()
    assert (installation / plugin_path / "hooks/hooks.json").is_file()
    assert (installation / plugin_path / "skills/reme-memory/SKILL.md").is_file()
    assert not list(installation.rglob("__pycache__"))
    if host == "codex":
        assert mcp["tool_timeout_sec"] == 3610


def test_switching_transports_rebuilds_only_the_managed_installation(setup):
    module, host, settings, key = setup
    settings.write_text(json.dumps({"autoRecall" if host == "codex" else "auto_recall": False}))
    installation = module.configure(host, "stdio", "http://127.0.0.1:2444/mcp/")
    mcp = connection(module, host, installation)
    assert mcp["command"] == "python3" and "url" not in mcp
    target = installation / module.HOSTS[host][1]
    assert (target / ("mcp_server.py" if host == "codex" else "hooks/mcp_bridge.py")).is_file()
    if host == "codex":
        assert (target / "ui/status.html").is_file()
    values = json.loads(settings.read_text())
    assert values[key] == "http://127.0.0.1:2444/mcp"
    assert values["autoRecall" if host == "codex" else "auto_recall"] is False
    (installation / "obsolete").write_text("generated artifact")

    assert module.configure(host, "http") == installation
    assert connection(module, host, installation)["url"] == values[key]
    assert not (installation / "obsolete").exists()


@pytest.mark.parametrize("url", ["file:///tmp/mcp", "http://user:secret@example.com/mcp", "http://bad/mcp?query=x"])
def test_invalid_address_does_not_change_configuration_or_installation(setup, url):
    module, host, settings, _ = setup
    installation = module.configure(host)
    previous = (installation / module.HOSTS[host][1] / ".mcp.json").read_bytes()
    with pytest.raises(ValueError):
        module.configure(host, "http", url)
    assert not settings.exists()
    assert (installation / module.HOSTS[host][1] / ".mcp.json").read_bytes() == previous


def test_unmanaged_directory_and_symlink_are_not_replaced(setup, tmp_path):
    module, host, settings, _ = setup
    destination = settings.parent / "plugin-marketplace"
    destination.mkdir()
    protected = destination / "user-file"
    protected.write_text("owned by the user")
    with pytest.raises(ValueError, match="unmanaged"):
        module.configure(host)
    assert protected.read_text() == "owned by the user"
    protected.unlink()
    destination.rmdir()
    destination.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module.configure(host)
    assert destination.is_symlink()


@pytest.mark.parametrize("failure", ["installation", "settings"])
def test_failed_replacement_restores_the_previous_installation(setup, failure):
    module, host, settings, _ = setup
    installation = module.configure(host, "stdio")
    previous = connection(module, host, installation)
    replace = module.os.replace

    def fail(source, destination):
        if (failure == "installation" and Path(source).name == "marketplace") or (
            failure == "settings" and Path(destination) == settings
        ):
            raise OSError("simulated replacement failure")
        return replace(source, destination)

    with patch.object(module.os, "replace", side_effect=fail):
        with pytest.raises(OSError, match="simulated"):
            module.configure(host, "http", "http://127.0.0.1:2444/mcp")
    assert connection(module, host, installation) == previous
    assert not settings.exists()


def test_cli_defaults_to_http_and_reports_native_install_commands(setup, monkeypatch, capsys):
    module, host, settings, _ = setup
    monkeypatch.setattr("sys.argv", [str(SCRIPT), host])
    module.main()
    output = capsys.readouterr().out
    assert "Prepared http MCP" in output
    assert "plugin marketplace add" in output
    command = "plugin add reme@reme-codex" if host == "codex" else "claude plugin install reme@reme-marketplace"
    assert command in output
    registration = next(line for line in output.splitlines() if "plugin marketplace add" in line)
    assert shlex.split(registration)[-1] == str(settings.parent / "plugin-marketplace")


def test_invalid_transport_fails_before_writing(setup):
    module, host, settings, _ = setup
    with pytest.raises(ValueError):
        module.configure(host, "sse")
    assert not settings.exists()
    assert not (settings.parent / "plugin-marketplace").exists()


@pytest.mark.parametrize("values", [[], None, {"unsupported": True}])
def test_invalid_saved_settings_fail_without_changing_local_data(setup, values):
    module, host, settings, _ = setup
    settings.write_text(json.dumps(values))
    previous = settings.read_bytes()
    with pytest.raises(ValueError):
        module.configure(host)
    assert settings.read_bytes() == previous
    assert not (settings.parent / "plugin-marketplace").exists()


def test_claude_custom_mcp_and_http_addresses_update_together(setup):
    module, host, settings, key = setup
    if host != "claude-code":
        with pytest.raises(ValueError, match="only used by Claude Code"):
            module.configure(host, api_url="https://example.com/jobs")
        return
    settings.write_text(json.dumps({"api_url": "https://old.example.com/jobs"}))
    before = settings.read_bytes()
    with pytest.raises(ValueError, match="Also pass --api-url"):
        module.configure(host, mcp_url="https://example.com/tools")
    assert settings.read_bytes() == before

    installation = module.configure(host, mcp_url="https://example.com/tools", api_url="https://example.com/jobs/")
    assert connection(module, host, installation)["url"] == "https://example.com/tools"
    assert json.loads(settings.read_text()) == {key: "https://example.com/tools", "api_url": "https://example.com/jobs"}


def test_http_hooks_reject_a_stale_native_address_until_reinstallation(setup):
    module, host, settings, key = setup
    installation = module.configure(host)
    _, plugin_path, _, config_path, _ = module.HOSTS[host]
    spec = importlib.util.spec_from_file_location("reme_installed_settings", installation / plugin_path / config_path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    assert adapter.load_config()[key] == "http://127.0.0.1:2333/mcp"
    settings.write_text(json.dumps({key: "http://127.0.0.1:2444/mcp"}))
    with pytest.raises(ValueError, match="Regenerate and reinstall"):
        adapter.load_config()

    module.configure(host)
    spec.loader.exec_module(adapter)  # A reinstalled host launches a new Hook process.
    assert adapter.load_config()[key] == "http://127.0.0.1:2444/mcp"


def test_loaded_connection_snapshot_survives_cache_removal(setup):
    module, host, _, key = setup
    installation = module.configure(host)
    _, plugin_path, _, config_path, _ = module.HOSTS[host]
    spec = importlib.util.spec_from_file_location("reme_removed_installation", installation / plugin_path / config_path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    shutil.rmtree(installation)
    assert adapter.load_config()[key] == "http://127.0.0.1:2333/mcp"
