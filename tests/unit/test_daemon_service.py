"""Exercise setup, encrypted persistence and the public address across child replacements."""

# The test replaces its subprocess and reaps the current owner in finally.
# pylint: disable=consider-using-with

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import yaml
from cryptography.fernet import Fernet
from fastmcp import Client
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from reme.daemon.config import Profile, Settings, resolve_worker_config
from reme.daemon.storage import SettingsStore
from reme.daemon.service import build_gateway, run_supervised
from reme.daemon.supervisor import Supervisor


def test_import_once_and_keep_saved_values(monkeypatch, tmp_path):
    """System values win during import; later environment edits cannot replace saved values."""
    launch_dir = tmp_path / "subproject"
    launch_dir.mkdir()
    monkeypatch.chdir(launch_dir)
    monkeypatch.setattr("reme.utils.env_utils._LOADED", False)
    monkeypatch.setattr("reme.utils.env_utils._LOADED_VALUES", {})
    monkeypatch.setenv("LLM_API_KEY", "system-key")
    monkeypatch.setenv("REME_MODEL_NAME", "fixture-import-model")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_API_KEY=file-key\nLLM_BASE_URL=https://model.example.com/v1\nDASHSCOPE_API_KEY=speech-key\n",
    )
    arguments = {"config": "proactive", "workspace_dir": str(tmp_path / "workspace")}
    _, saved, view = resolve_worker_config(arguments, None)
    assert saved.api_key.get_secret_value() == "system-key"
    assert saved.base_url == "https://model.example.com/v1"
    assert saved.asr_enabled and saved.tts_enabled
    assert not view["missing_fields"]
    assert "system-key" not in json.dumps(view)
    env_file.write_text("LLM_API_KEY=changed-key\nLLM_BASE_URL=https://changed.example.com\n")
    monkeypatch.setenv("LLM_API_KEY", "changed-system-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "changed-speech-key")
    saved.asr_enabled = saved.tts_enabled = False
    config, _, view = resolve_worker_config(arguments, saved)
    server = config["components"]["codex_app_server"]["default"]
    assert server["api_key"] == "system-key" and server["base_url"] == saved.base_url
    assert 'cli_auth_credentials_store="ephemeral"' in server["config_overrides"]
    assert config["components"]["tts"]["default"]["api_key"] == "speech-key"
    assert not config["jobs"]["asr"]["enabled"] and not config["jobs"]["tts"]["enabled"]
    assert not view["asr_enabled"] and not view["tts_enabled"]


def test_default_instance_directory_identifies_the_listener(monkeypatch, tmp_path):
    """Listen on independent addresses while keeping the same endpoint's settings on restart."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in ("REME_DAEMON_DIR", "REME_ADMIN_TOKEN", "REME_ADMIN_PUBLISHED_HOST"):
        monkeypatch.delenv(name, raising=False)
    supervisor = Mock()
    monkeypatch.setattr("reme.daemon.service.Supervisor", supervisor)
    monkeypatch.setattr("reme.daemon.service.build_gateway", Mock())
    monkeypatch.setattr("reme.daemon.service.uvicorn.run", Mock())

    arguments = {"config": "proactive", "workspace_dir": "workspace-a"}
    other = {**arguments, "workspace_dir": "workspace-b"}
    run_supervised(arguments, {"host": "127.0.0.1", "port": 2333}, {})
    run_supervised(other, {"host": "127.0.0.1", "port": 2444}, {})
    run_supervised(other, {"host": "127.0.0.2", "port": 2333}, {})
    run_supervised(other, {}, {})

    first, second, alternative_host, restarted = [call.args[0] for call in supervisor.call_args_list]
    assert len({first, second, alternative_host}) == 3
    assert restarted == first


@pytest.mark.parametrize("field", ["base_url", "speech_base_url"])
def test_remote_provider_keys_require_encrypted_transport(field):
    """Remote API URLs require HTTPS while local protocol fixtures may use loopback."""
    with pytest.raises(ValueError, match="HTTPS"):
        Settings(**{field: "http://provider.example.com/v1"})
    assert getattr(Settings(**{field: "http://127.0.0.1:8999/v1"}), field).startswith("http://127.")
    assert getattr(Settings(**{field: "https://provider.example.com/v1"}), field).startswith("https:")


def test_settings_store_encrypts_keys_and_rejects_wrong_key(monkeypatch, tmp_path):
    """Keys stay encrypted on disk and decryption failure preserves the original file."""
    monkeypatch.setenv("REME_ENCRYPTION_KEY", Fernet.generate_key().decode())
    store = SettingsStore(tmp_path)
    profile = Profile(revision=3, settings=Settings(api_key="provider-secret", dashscope_api_key="speech-secret"))
    store.commit(store.prepare(profile))
    content = store.path.read_text()
    assert "provider-secret" not in content and "speech-secret" not in content
    assert SettingsStore(tmp_path).load() == profile
    if os.name != "nt":
        assert store.path.stat().st_mode & 0o777 == 0o600
    monkeypatch.setenv("REME_ENCRYPTION_KEY", Fernet.generate_key().decode())
    with pytest.raises(ValueError, match="encryption key"):
        SettingsStore(tmp_path).load()
    assert store.path.read_text() == content


@pytest.mark.parametrize(
    "settings",
    [[], {"api_key": 123}, {"dashscope_api_key": {}}],
)
def test_invalid_saved_settings_keep_the_gateway_available(tmp_path, settings):
    """Malformed settings require repair and are preserved for recovery."""
    path = tmp_path / "settings.json"
    content = json.dumps({"revision": 0, "key_id": None, "settings": settings})
    path.write_text(content)
    supervisor = Supervisor(tmp_path, {}, {})
    assert supervisor.port is None
    assert "Restore settings.json" in supervisor.settings()["settings_error"]
    assert path.read_text() == content


@pytest.mark.asyncio
async def test_restart_records_failed_cleanup_and_keeps_the_owned_worker(tmp_path):
    """A cleanup error must settle the operation without losing a live child."""
    supervisor = Supervisor(tmp_path, {}, {})
    supervisor.active = Profile(revision=1)
    supervisor.view = {"model_editable": True, "missing_fields": []}
    worker = Mock(port=2333, process=Mock(returncode=None), stop=AsyncMock(side_effect=OSError("stop failed")))
    supervisor.worker = worker
    candidate = Profile(revision=2)
    operation = {"id": "restart", "phase": "accepted", "error": None}
    supervisor.operations[operation["id"]] = operation

    prepared = supervisor.store.prepare(candidate)
    await supervisor._replace(candidate, prepared, operation)  # pylint: disable=protected-access

    state = supervisor.status(operation["id"])
    assert state["ready"] and state["revision"] == 1
    assert state["operation"]["phase"] == "failed"
    assert "stop failed" in state["operation"]["error"]
    assert supervisor.worker is worker
    assert supervisor.settings()["model_editable"]
    assert json.loads(supervisor.state_file.read_text())["operations"]["restart"] == operation


@pytest.mark.asyncio
async def test_remote_administration_requires_explicit_token_and_tls(tmp_path):
    """Verify the public authorization boundary and the authenticated Compose route."""
    supervisor = Supervisor(tmp_path, {}, {})
    fixture_token = "fixture-admin"
    headers = {"X-ReMe-Admin-Token": fixture_token}
    app = build_gateway(supervisor, None, local_listener=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("192.0.2.1", 1)),
        base_url="https://reme.example",
    ) as client:
        assert (await client.get("/admin/config", headers=headers)).status_code == 403
    app = build_gateway(
        supervisor,
        None,
        local_listener=False,
        admin_token=fixture_token,
        remote_management=True,
        published_loopback=True,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("192.0.2.1", 1)),
        base_url="https://reme.example",
    ) as client:
        for method, endpoint in (
            ("GET", "config"),
            ("GET", "status"),
            ("POST", "apply"),
            ("POST", "restart"),
            ("POST", "test_model"),
        ):
            assert (await client.request(method, f"/admin/{endpoint}")).status_code == 401
        assert (await client.get("/admin/config", headers=headers)).is_success
        assert (await client.get("http://reme.example/admin/config", headers=headers)).status_code == 403
        assert (await client.get("http://127.0.0.1/admin/config")).status_code == 401
        assert (await client.get("http://127.0.0.1/admin/config", headers=headers)).is_success


@pytest.mark.parametrize(
    ("token", "status"),
    [("fixture-admin", 200), ("wrong-admin", 401), ("Bearer fixture-admin", 401), ("", 401), (None, 401)],
)
@pytest.mark.asyncio
async def test_only_admin_header_authenticates_management_requests(tmp_path, token, status):
    """Platform Authorization headers never substitute for the dedicated management credential."""
    app = build_gateway(
        Supervisor(tmp_path, {}, {}),
        None,
        local_listener=False,
        admin_token="fixture-admin",
        remote_management=True,
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://reme.example") as client:
        for authorization in ("Bearer platform-token", "Bearer fixture-admin"):
            headers = {"Authorization": authorization}
            if token is not None:
                headers["X-ReMe-Admin-Token"] = token
            reply = await client.get("/admin/config", headers=headers)
            assert reply.status_code == status
            assert "www-authenticate" not in reply.headers


@pytest.mark.parametrize(
    ("peer", "scheme", "origin", "status"),
    [
        ("192.0.2.10", "https", "https://reme.example", 200),
        ("192.0.2.11", "https", "https://reme.example", 403),
        ("192.0.2.10", "http", "http://reme.example", 403),
        ("192.0.2.10", "https", "https://unrelated.example", 403),
    ],
)
@pytest.mark.asyncio
async def test_management_behind_proxy_checks_peer_https_and_origin(tmp_path, peer, scheme, origin, status):
    """An HTTP listener accepts external HTTPS only through the configured trusted proxy."""
    app = build_gateway(
        Supervisor(tmp_path, {}, {}),
        None,
        local_listener=False,
        admin_token="fixture-admin",
        remote_management=True,
    )
    proxy = ProxyHeadersMiddleware(app, trusted_hosts="192.0.2.10")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy, client=(peer, 12345)),
        base_url="http://reme.example",
    ) as client:
        reply = await client.get(
            "/admin/config",
            headers={
                "X-ReMe-Admin-Token": "fixture-admin",
                "X-Forwarded-Proto": scheme,
                "Origin": origin,
            },
        )
        assert reply.status_code == status


def test_supervised_listener_receives_proxy_configuration(monkeypatch, tmp_path):
    """Keep proxy trust options on the public listener that checks management requests."""
    monkeypatch.setenv("REME_DAEMON_DIR", str(tmp_path))
    monkeypatch.setattr("reme.daemon.service.Supervisor", Mock())
    monkeypatch.setattr("reme.daemon.service.build_gateway", Mock())
    run = Mock()
    monkeypatch.setattr("reme.daemon.service.uvicorn.run", run)

    run_supervised(
        {},
        {"host": "0.0.0.0", "port": 7860, "proxy_headers": True, "forwarded_allow_ips": "192.0.2.10"},
        {},
    )

    assert run.call_args.kwargs["proxy_headers"] is True
    assert run.call_args.kwargs["forwarded_allow_ips"] == "192.0.2.10"
    assert run.call_args.kwargs["host"] == "0.0.0.0"
    assert run.call_args.kwargs["port"] == 7860


@pytest.mark.asyncio
async def test_operation_history_failures_leave_useful_status(monkeypatch, tmp_path):
    """Corrupt history cannot block startup and a write failure cannot strand an accepted operation."""
    path = tmp_path / "state.json"
    path.write_text("broken state")
    supervisor = Supervisor(tmp_path, {}, {})
    assert supervisor.status(None)["error"] == "Cannot read the ReMe operation history"
    assert path.read_text() == "broken state"
    profile = Profile(settings=Settings(workspace_dir=str(tmp_path / "workspace")))
    supervisor.active = profile
    launch = AsyncMock(return_value=(None, {"view": {"missing_fields": []}}))
    start = AsyncMock()
    monkeypatch.setattr(supervisor, "_launch", launch)
    monkeypatch.setattr(supervisor, "_start", start)
    writer = Mock(side_effect=OSError("fixture disk error"))
    monkeypatch.setattr("reme.daemon.supervisor.write_private", writer)
    with pytest.raises(RuntimeError, match="operation history"):
        await supervisor.request(None)
    assert supervisor.status(None)["operation"] is None
    start.assert_not_awaited()
    writer.side_effect = [None, OSError("fixture disk error")]
    operation = await supervisor.request(None)
    await supervisor.task
    start.assert_awaited_once_with(profile)
    result = supervisor.status(operation["id"])
    assert result["operation"]["phase"] == "succeeded"
    assert result["error"] == "Cannot save the ReMe operation history"


def _wait_for(check):
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        try:
            value = check()
            if value:
                return value
        except httpx.TransportError:
            pass
        time.sleep(0.05)
    raise AssertionError("ReMe did not reach the expected state")


def test_setup_apply_recover_and_restart_with_saved_settings(tmp_path):
    """A new Codex instance waits for setup and retains the encrypted settings after a host restart."""
    workspace = tmp_path / "workspace"
    target = tmp_path / "target"
    launch_dir = tmp_path / "subproject"
    launch_dir.mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_API_KEY=ignored-dotenv-key\nLLM_BASE_URL=https://model.example.com/v1\n",
        encoding="utf-8",
    )
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "daily").write_text("blocks directory creation", encoding="utf-8")
    static = tmp_path / "web"
    static.mkdir()
    (static / "index.html").write_text("<title>ReMe Studio</title>", encoding="utf-8")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = tmp_path / "app.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "workspace_dir": str(workspace),
                "enable_logo": False,
                "log_to_file": False,
                "service": {
                    "backend": "http",
                    "port": port,
                    "mcp_enabled": True,
                    "web_enabled": True,
                    "web_static_dir": str(static),
                },
                "components": {
                    "codex_app_server": {
                        "default": {
                            "backend": "sdk",
                            "auth_mode": "api_key",
                            "api_key": "${LLM_API_KEY:-}",
                            "base_url": "${LLM_BASE_URL}",
                            "codex_home": ".codex",
                        },
                    },
                    "agent_wrapper": {"default": {"backend": "codex", "model": "fixture-model", "effort": "high"}},
                    "asr": {"default": {"backend": "dashscope"}},
                    "tts": {"default": {"backend": "dashscope"}},
                },
                "jobs": {
                    "asr": {"backend": "base", "steps": [{"backend": "asr_step", "asr": "default"}]},
                    "tts": {"backend": "base", "steps": [{"backend": "tts_step", "tts": "default"}]},
                    "app_config": {"backend": "base", "steps": [{"backend": "app_config_step"}]},
                    "get_client_info": {"backend": "base", "steps": [{"backend": "client_info_step"}]},
                    "stream_demo": {
                        "backend": "stream",
                        "steps": [{"backend": "stream_demo_step1"}, {"backend": "stream_demo_step2"}],
                    },
                },
            },
        ),
        encoding="utf-8",
    )
    daemon = tmp_path / "daemon"
    environment = {
        **os.environ,
        "REME_DAEMON_DIR": str(daemon),
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        "REME_ENCRYPTION_KEY": Fernet.generate_key().decode(),
        "LLM_API_KEY": "",
        "DASHSCOPE_API_KEY": "",
    }
    environment.pop("LLM_BASE_URL", None)
    environment.pop("REME_ADMIN_TOKEN", None)
    environment.pop("REME_ADMIN_PUBLISHED_HOST", None)
    launch = [sys.executable, "-m", "reme.reme", "start", f"config={config}"]
    origin = f"http://127.0.0.1:{port}"
    with (tmp_path / "service.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            launch,
            cwd=launch_dir,
            env=environment,
            stdout=log,
            stderr=log,
        )  # pylint: disable=consider-using-with
        try:
            with httpx.Client(base_url=origin, trust_env=False, timeout=15) as client:

                def waiting_for_setup():
                    reply = client.get("/admin/status")
                    return reply.json() if reply.is_success and reply.json()["needs_setup"] else None

                initial = _wait_for(waiting_for_setup)
                assert initial["missing_fields"] == ["api_key"]
                assert client.get("/").is_success and client.post("/get_client_info", json={}).status_code == 503
                assert not (daemon / "settings.json").exists()
                view = client.get("/admin/config").json()
                assert view["available_harnesses"] == ["codex"] and not view["api_key_set"]
                assert not view["asr_enabled"] and not view["tts_enabled"]
                assert client.post("/admin/apply", json={"revision": 0}).status_code == 400
                assert client.post("/admin/restart", headers={"Origin": "https://unrelated.example"}).status_code == 403
                assert (
                    client.get(
                        "/admin/config",
                        headers={"Host": "unrelated.example", "Origin": "http://unrelated.example"},
                    ).status_code
                    == 403
                )
                assert client.get("/admin/config", headers={"Origin": origin}).is_success
                assert client.get(
                    "/admin/config",
                    headers={"Origin": "https://appassets.androidplatform.net"},
                ).is_success

                def settled(identity):
                    status = client.get("/admin/status", params={"operation_id": identity}).json()
                    return status if status["operation"]["phase"] in {"succeeded", "failed"} else None

                accepted = client.post(
                    "/admin/apply",
                    json={"revision": 0, "api_key": "test-provider-key", "workspace_dir": str(target)},
                )
                assert accepted.status_code == 202, accepted.text
                result = _wait_for(lambda: settled(accepted.json()["id"]))
                assert result["ready"] and not result["needs_setup"] and result["revision"] == 1
                assert client.post("/get_client_info", json={}).json()["answer"]["workspace_dir"] == str(target)
                assert not client.post("/get_client_info", json={}).json()["answer"]["asr"]
                assert "test-provider-key" not in (daemon / "settings.json").read_text()
                assert "test-provider-key" not in client.get("/admin/config").text
                assert "test-provider-key" not in (daemon / "state.json").read_text()
                assert client.post("/admin/apply", json={"revision": 0}).status_code == 409
                assert client.post("/admin/apply", json={"revision": 1, "tts_enabled": True}).status_code == 400

                speech = {
                    "asr_enabled": True,
                    "tts_enabled": True,
                    "speech_base_url": "https://speech.example.com/api/v1",
                    "asr_model": "fixture-asr",
                    "tts_model": "fixture-tts",
                    "tts_voice": "fixture-voice",
                }
                accepted = client.post(
                    "/admin/apply",
                    json={"revision": 1, "dashscope_api_key": "test-speech-key", **speech},
                )
                assert accepted.status_code == 202, accepted.text
                assert _wait_for(lambda: settled(accepted.json()["id"]))["operation"]["phase"] == "succeeded"
                view = client.get("/admin/config").json()
                assert all(view[key] == value for key, value in speech.items())
                public = client.post("/app_config", json={}).json()["answer"]
                assert public["components"]["tts"]["default"]["api_key"] == "***"
                assert public["components"]["tts"]["default"]["voice"] == "fixture-voice"
                assert client.post("/get_client_info", json={}).json()["answer"]["tts"]
                saved = (daemon / "settings.json").read_text()
                assert "test-speech-key" not in saved
                accepted = client.post(
                    "/admin/apply",
                    json={
                        "revision": 2,
                        "workspace_dir": str(broken),
                        "api_key": "rejected-key",
                        "tts_voice": "rejected-voice",
                    },
                )
                assert accepted.status_code == 202, accepted.text
                result = _wait_for(lambda: settled(accepted.json()["id"]))
                assert result["operation"]["phase"] == "failed" and result["ready"] and result["revision"] == 2
                assert (daemon / "settings.json").read_text() == saved
                assert client.get("/admin/config").json()["tts_voice"] == "fixture-voice"
                with client.stream("POST", "/stream_demo", json={"query": "ab", "repeat": 1, "interval": 0}) as reply:
                    assert [line for line in reply.iter_lines() if line][-1] == "data:[DONE]"

                async def mcp_tools():
                    async with Client(f"{origin}/mcp") as mcp:
                        return {tool.name for tool in await mcp.list_tools()}

                assert "get_client_info" in asyncio.run(mcp_tools())
                process.terminate()
                process.wait(timeout=15)
                config.write_text(config.read_text().replace("${LLM_API_KEY:-}", "${LLM_API_KEY}"))
                environment.pop("LLM_API_KEY")
                env_file.write_text(
                    "LLM_BASE_URL=https://changed.example.com\nDASHSCOPE_API_KEY=changed-speech-key\n",
                    encoding="utf-8",
                )
                process = subprocess.Popen(
                    launch,
                    cwd=launch_dir,
                    env=environment,
                    stdout=log,
                    stderr=log,
                )  # pylint: disable=consider-using-with
                _wait_for(lambda: client.get("/admin/status").json()["ready"])
                view = client.get("/admin/config").json()
                assert view["base_url"] == "https://model.example.com/v1" and view["revision"] == 2
                assert all(view[key] == value for key, value in speech.items())
                assert (daemon / "settings.json").read_text() == saved
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
