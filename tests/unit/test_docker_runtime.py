"""Container configuration and health checks without Docker or model calls."""

import json
from io import BytesIO
from unittest.mock import Mock

import pytest

from deploy.docker import reme_container
from reme.config import parse_kwargs, resolve_app_config


def test_container_defaults_and_cli_override_precedence(tmp_path):
    """CLI overrides win over container settings, including structured service values."""
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"service": {"backend": "http", "port": 8001}, "language": "zh"}))
    command = reme_container.start_command(
        ['service={"port":8003,"web_enabled":false}', "log_to_file=true"],
        {
            "REME_CONFIG": str(config_file),
            "REME_WORKSPACE_DIR": str(tmp_path / "data"),
            "REME_HOST": "0.0.0.0",
            "REME_PORT": "8002",
            "REME_TIMEZONE": "UTC",
        },
    )
    config = resolve_app_config(log_config=False, **parse_kwargs(*command[2:]))
    assert config["service"] == {"backend": "http", "host": "0.0.0.0", "port": 8003, "web_enabled": False}
    assert config["workspace_dir"] == str(tmp_path / "data")
    assert config["timezone"] == "UTC"
    assert config["language"] == "zh"
    assert config["log_to_file"] is True


def test_container_preserves_file_config_port_and_quoted_paths(tmp_path):
    """Unset environment options preserve the file; paths survive CLI serialization."""
    config_file = tmp_path / 'configuration "with spaces".json'
    config_file.write_text(json.dumps({"service": {"backend": "http", "host": "0.0.0.0", "port": 8234}}))
    command = reme_container.start_command([], {"REME_CONFIG": str(config_file)})
    state_path = tmp_path / "health.json"
    reme_container.write_health_state(command, state_path)
    assert json.loads(state_path.read_text()) == {"url": "http://127.0.0.1:8234/health_check"}


def test_health_state_contains_no_config_credentials(tmp_path):
    """The ephemeral probe address follows effective CLI settings and stores no secrets."""
    command = reme_container.start_command(
        ["service.port=8901", "components.as_llm.default.credential.api_key=not-a-real-key"],
        {"REME_PORT": "8900", "REME_HOST": "0.0.0.0"},
    )
    state_path = tmp_path / "health.json"
    reme_container.write_health_state(command, state_path)
    assert json.loads(state_path.read_text()) == {"url": "http://127.0.0.1:8901/health_check"}
    assert state_path.stat().st_mode & 0o777 == 0o600


def test_one_shot_job_does_not_register_http_health_state(tmp_path):
    """One-shot jobs use the ordinary CLI lifecycle without pretending to serve HTTP."""
    state_path = tmp_path / "health.json"
    command = reme_container.start_command(["job=version"], {})
    reme_container.write_health_state(command, state_path)
    assert not state_path.exists()


@pytest.mark.parametrize("port", ["0", "65536", "invalid"])
def test_container_rejects_invalid_port(port, tmp_path):
    """Invalid addresses fail before the server starts."""
    with pytest.raises(ValueError):
        command = reme_container.start_command([], {"REME_PORT": port})
        reme_container.write_health_state(command, tmp_path / "health.json")


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"success": True, "metadata": {"health": {"healthy": True}}}, 0),
        ({"success": True, "metadata": {"health": {"healthy": False}}}, 1),
        ({"success": False, "metadata": {"health": {"healthy": True}}}, 1),
        ({"success": True, "metadata": {}}, 1),
        ({"success": True, "metadata": None}, 1),
        (["not a Job response"], 1),
    ],
)
def test_healthcheck_validates_response_body(payload, expected, tmp_path, monkeypatch):
    """HTTP 200 alone cannot turn an unhealthy or malformed response into success."""
    state_path = tmp_path / "health.json"
    state_path.write_text(json.dumps({"url": "http://127.0.0.1:2333/health_check"}))
    opener = Mock()
    opener.open.return_value = BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr(reme_container, "build_opener", Mock(return_value=opener))
    assert reme_container.healthcheck(state_path) == expected
    request = opener.open.call_args.args[0]
    assert request.get_method() == "POST"
    assert request.data == b"{}"


def test_healthcheck_handles_missing_state_and_unreachable_server(tmp_path, monkeypatch):
    """Missing initialization and connection errors fail the probe cleanly."""
    state_path = tmp_path / "health.json"
    assert reme_container.healthcheck(state_path) == 1
    state_path.write_text(json.dumps({"url": "http://127.0.0.1:2333/health_check"}))
    opener = Mock()
    opener.open.side_effect = OSError("connection refused")
    monkeypatch.setattr(reme_container, "build_opener", Mock(return_value=opener))
    assert reme_container.healthcheck(state_path) == 1


@pytest.mark.parametrize("action", [["start"], ["--start"], ["reme", "-start"], ["reme", "start"]])
def test_entrypoint_executes_start_and_preserves_cli_action_syntax(action, tmp_path, monkeypatch):
    """All supported start spellings apply defaults and replace the supervisor child."""
    monkeypatch.setattr(reme_container.sys, "argv", ["entrypoint", *action, "service.port=8901"])
    monkeypatch.setattr(reme_container, "HEALTH_STATE_PATH", tmp_path / "health.json")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("REME_WORKSPACE_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("reme.utils.load_env", Mock())
    state_writer = Mock()
    monkeypatch.setattr(reme_container, "write_health_state", state_writer)
    exec_command = Mock(side_effect=RuntimeError("exec called"))
    monkeypatch.setattr(reme_container.os, "execvp", exec_command)

    with pytest.raises(RuntimeError, match="exec called"):
        reme_container.main()

    executable, arguments = exec_command.call_args.args
    assert executable == "reme"
    assert arguments[:2] == ["reme", "start"]
    assert parse_kwargs(*arguments[2:])["service"]["port"] == 8901
    state_writer.assert_called_once_with(arguments)


def test_entrypoint_passes_other_commands_without_shell_interpolation(tmp_path, monkeypatch):
    """Diagnostic commands retain literal arguments and discard stale HTTP probe state."""
    command = ["python", "-c", "print('$HOME and `literal text`')"]
    monkeypatch.setattr(reme_container.sys, "argv", ["entrypoint", *command])
    state_path = tmp_path / "health.json"
    state_path.write_text("stale")
    monkeypatch.setattr(reme_container, "HEALTH_STATE_PATH", state_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    exec_command = Mock(side_effect=RuntimeError("exec called"))
    monkeypatch.setattr(reme_container.os, "execvp", exec_command)

    with pytest.raises(RuntimeError, match="exec called"):
        reme_container.main()

    exec_command.assert_called_once_with("python", command)
    assert not state_path.exists()
