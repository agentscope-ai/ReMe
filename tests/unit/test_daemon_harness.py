"""Managed model selection, persistence and isolated connection tests."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from reme.daemon.config import Profile, Settings, SettingsRequest, resolve_worker_config
from reme.daemon.supervisor import Supervisor
from reme.daemon.service import build_gateway
from reme.daemon.worker import _test_model


@pytest.fixture
def arguments(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("reme.daemon.config.load_env", lambda **_kwargs: {})
    monkeypatch.setenv("LLM_API_KEY", "shared-model-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://model.example.test/api")
    monkeypatch.setenv("REME_MODEL_NAME", "shared-model")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    return {"config": "proactive", "workspace_dir": str(tmp_path / "workspace"), "log_config": False}


@pytest.mark.parametrize("harness,preset", [("codex", "proactive"), ("claude_code", "proactive_cc")])
def test_presets_import_shared_model_environment(arguments, harness, preset):
    config, settings, view = resolve_worker_config({**arguments, "config": preset}, None)
    assert settings.harness == harness
    assert settings.api_key.get_secret_value() == "shared-model-key"
    assert settings.base_url == "https://model.example.test/api"
    assert settings.model == "shared-model" and settings.effort == "xhigh"
    assert config["components"]["agent_wrapper"]["default"]["backend"] == harness
    assert view["model_editable"] and view["available_harnesses"] == ["codex", "claude_code"]
    assert view["api_key_set"] and "api_key" not in view and not view["missing_fields"]


def test_claude_requires_credentials_and_uses_its_default_endpoint(arguments, monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL")
    monkeypatch.setenv("LLM_API_KEY", "")
    _, settings, view = resolve_worker_config({**arguments, "config": "proactive_cc"}, None)
    assert settings.base_url == "https://dashscope.aliyuncs.com/apps/anthropic"
    assert view["missing_fields"] == ["api_key"]


def test_saved_managed_settings_require_an_explicit_harness(arguments):
    _, saved, _ = resolve_worker_config({**arguments, "config": "proactive_cc"}, None)
    saved = Settings(**{**saved.private_values(), "harness": None, "api_key": None})
    with pytest.raises(ValueError, match="selected Harness"):
        resolve_worker_config({**arguments, "config": "proactive_cc"}, saved)


def test_saved_selection_switches_all_runtimes_and_survives_environment_changes(arguments, monkeypatch):
    _, saved, _ = resolve_worker_config(arguments, None)
    saved = Settings(**{**saved.private_values(), "harness": "claude_code", "effort": "high"})
    monkeypatch.setenv("LLM_API_KEY", "changed-environment-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://changed.example.test")
    monkeypatch.setenv("REME_MODEL_NAME", "changed-model")
    config, _, view = resolve_worker_config(arguments, saved)
    components = config["components"]
    wrapper = components["agent_wrapper"]["default"]
    assert wrapper["backend"] == "claude_code"
    assert wrapper["api_key"] == "shared-model-key" and wrapper["model"] == "shared-model"
    assert wrapper["effort"] == "high"
    assert wrapper["system_prompt"] == {"type": "preset", "preset": "claude_code", "snapshot": False}
    assert components["anthropic_relay"]["default"]["base_url"] == saved.base_url
    assert all(runtime["agent_wrapper"] == "default" for runtime in components["agent_runtime"].values())
    assert components["agent_manager"]["default"]["agent_wrapper"] == "default"
    assert config["environment"]["LLM_API_KEY"] == "shared-model-key" and view["harness"] == "claude_code"

    saved = Settings(**{**saved.private_values(), "harness": "codex", "effort": "minimal"})
    config, _, view = resolve_worker_config({**arguments, "config": "proactive_cc"}, saved)
    wrapper = config["components"]["agent_wrapper"]["default"]
    server = config["components"]["codex_app_server"]["default"]
    assert wrapper["backend"] == "codex" and wrapper["effort"] == "minimal"
    assert "anthropic_relay" not in wrapper and "system_prompt" not in wrapper
    assert server["api_key"] == "shared-model-key" and server["base_url"] == saved.base_url
    assert view["harness"] == "codex"


@pytest.mark.parametrize("effort", ["none", "minimal"])
def test_claude_rejects_unsupported_effort(effort):
    with pytest.raises(ValueError, match="Claude Code effort"):
        Settings(harness="claude_code", effort=effort)


def test_custom_configuration_only_offers_its_own_harness(arguments, tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text("extends: proactive\n", encoding="utf-8")
    config_arguments = {**arguments, "config": str(path)}
    _, saved, view = resolve_worker_config(config_arguments, None)
    assert view["available_harnesses"] == ["codex"]
    saved = Settings(**{**saved.private_values(), "harness": "claude_code"})
    with pytest.raises(ValueError, match="selected Harness"):
        resolve_worker_config(config_arguments, saved)


def test_claude_patch_keeps_omitted_and_blank_credentials(tmp_path):
    supervisor = Supervisor(tmp_path, {}, {})
    supervisor.active = Profile(
        revision=2,
        settings=Settings(api_key="saved-key", dashscope_api_key="speech-key", harness="codex", effort="high"),
    )
    candidate = supervisor._candidate(SettingsRequest(revision=2, harness="claude_code", api_key=""))
    assert candidate.revision == 3 and candidate.settings.harness == "claude_code"
    assert candidate.settings.api_key.get_secret_value() == "saved-key"
    assert candidate.settings.dashscope_api_key.get_secret_value() == "speech-key"
    assert supervisor.active.settings.harness == "codex"


@pytest.mark.asyncio
async def test_admin_accepts_claude_model_testing_and_saving(tmp_path):
    supervisor = Supervisor(tmp_path, {}, {})
    supervisor.test_model = AsyncMock()
    supervisor.request = AsyncMock(return_value={"id": "claude-change", "phase": "accepted", "error": None})
    app = build_gateway(supervisor, None)
    payload = {
        "revision": 0,
        "harness": "claude_code",
        "base_url": "https://model.example.test/anthropic",
        "model": "shared-model",
        "effort": "high",
        "api_key": "test-provider-key",
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        assert (await client.post("/admin/test_model", json=payload)).is_success
        assert (await client.post("/admin/apply", json=payload)).status_code == 202
        assert (await client.post("/admin/apply", json={**payload, "effort": "minimal"})).status_code == 422
    supervisor.test_model.assert_awaited_once()
    supervisor.request.assert_awaited_once()
    candidate = supervisor.request.call_args.args[0]
    assert candidate.harness == "claude_code" and candidate.base_url == payload["base_url"]


@pytest.mark.asyncio
@pytest.mark.parametrize("preset", ["proactive", "proactive_cc"])
@pytest.mark.parametrize("success", [True, False])
async def test_connection_test_uses_selected_dependencies_in_an_isolated_workspace(
    arguments,
    monkeypatch,
    preset,
    success,
):
    config, _, _ = resolve_worker_config({**arguments, "config": preset}, None)
    config["environment"]["CLAUDE_CONFIG_DIR"] = "/original/claude-home"
    captured = {}
    wrapper = SimpleNamespace(
        reply=AsyncMock(return_value={"status": "completed" if success else "failed", "result": "OK"}),
    )

    class TestApplication:
        """Capture connection-test wiring without making provider requests."""

        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.context = SimpleNamespace(components={"agent_wrapper": {"default": wrapper}})

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    monkeypatch.setattr("reme.daemon.worker.Application", TestApplication)
    if success:
        await _test_model(config)
    else:
        with pytest.raises(RuntimeError, match="did not complete"):
            await _test_model(config)
    assert captured["jobs"] == {} and captured["workspace_dir"] != arguments["workspace_dir"]
    assert not Path(captured["workspace_dir"]).exists()
    components = captured["components"]
    if preset == "proactive_cc":
        assert set(components) == {"agent_wrapper", "anthropic_relay"}
        assert components["agent_wrapper"]["default"]["tools"] == []
        assert captured["environment"]["CLAUDE_CONFIG_DIR"].startswith(captured["workspace_dir"])
    else:
        assert set(components) == {"agent_wrapper", "codex_app_server"}
        assert components["codex_app_server"]["default"]["codex_home"] == ".codex"
    wrapper.reply.assert_awaited_once_with("Reply with OK.", job_tools=[])
