"""Decision model registration, optional config, and secret redaction."""

# pylint: disable=redefined-outer-name,unused-argument

import asyncio
from dataclasses import dataclass, field
from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest
import httpx

from reme.application import Application
from reme.components import R
from reme.components.as_decision_model import AgentScopeAsDecisionModel
from reme.components.as_decision_model import agentscope as provider_module
from reme.components.client import HttpClient
from reme.config import parse_kwargs, resolve_app_config
from reme.enumeration import ComponentEnum
from reme.steps.common.app_config import AppConfigStep
from reme.steps.common.decision_demo import DecisionDemoStep


@pytest.fixture
def decision_env(monkeypatch):
    """Keep all examples independent of the user's local provider settings."""
    for name in ("DECISION_MODEL_BACKEND", "DECISION_MODEL_NAME", "DECISION_MODEL_API_KEY", "DECISION_MODEL_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


def test_registration_and_default_opt_in(decision_env):
    """Imports register providers, but default jobs never enable classification."""
    assert R.get(ComponentEnum.AS_DECISION_MODEL, "agentscope") is AgentScopeAsDecisionModel
    assert R.get(ComponentEnum.STEP, "decision_demo_step") is DecisionDemoStep
    default = resolve_app_config(log_config=False)
    assert "as_decision_model" not in default["components"]
    assert "decide" not in default["jobs"]


def test_demo_config_env_and_cli_override(decision_env, monkeypatch):
    """Provider env expands and CLI transport settings deep-merge over the file."""
    monkeypatch.setenv("DECISION_MODEL_API_KEY", "test-key")
    monkeypatch.setenv("DECISION_MODEL_NAME", "decision-test")
    config = resolve_app_config(
        log_config=False,
        **parse_kwargs(
            "config=decision_demo",
            "components.as_decision_model.default.timeout=12",
        ),
    )
    component = config["components"]["as_decision_model"]["default"]
    assert component == {
        "backend": "agentscope",
        "model": "decision-test",
        "credential": {"api_key": "test-key", "base_url": None},
        "timeout": 12,
        "max_retries": 2,
        "retry_delay": 0.5,
    }
    default = resolve_app_config(log_config=False)
    assert config["service"] == default["service"]
    assert config["service"]["backend"] == "http"
    assert config["jobs"]["decide"]["enable_serve"] is True
    assert "decision_demo" not in config["jobs"]
    assert {name: job for name, job in config["jobs"].items() if name != "decide"} == default["jobs"]
    assert {name: group for name, group in config["components"].items() if name != "as_decision_model"} == default[
        "components"
    ]


def test_application_starts_without_optional_imports_and_redacts_key(decision_env, monkeypatch, tmp_path):
    """The inherited config wires components without creating a decision model client."""
    monkeypatch.setenv("DECISION_MODEL_API_KEY", "test-key")
    monkeypatch.setattr(provider_module, "import_module", lambda _: pytest.fail("Unexpected optional import"))
    config = resolve_app_config(config="decision_demo", log_config=False)

    async def run():
        app = Application(
            **config,
            workspace_dir=str(tmp_path),
            enable_logo=False,
            log_to_console=False,
            log_to_file=False,
        )
        async with app:
            response = await AppConfigStep(app_context=app.context)()
            assert response.answer["components"]["as_decision_model"]["default"]["credential"]["api_key"] == "***"

    asyncio.run(run())


def test_decision_client_calls_inherited_http_service(decision_env, monkeypatch, tmp_path, capsys):
    """The two-stage workflow exposes /decide and sends client payloads to its Step."""

    @dataclass
    class Result:
        """Mock only the decision model boundary; exercise the actual service and client."""

        model: str = "decision-test"
        content: dict = field(default_factory=lambda: {"urgent": {"type": "binary_answer", "probability": 0.8}})

    calls = []

    async def classify(_self, state, questions):
        calls.append((state, questions))
        return Result()

    monkeypatch.setattr(AgentScopeAsDecisionModel, "__call__", classify)
    config = resolve_app_config(config="decision_demo", log_config=False)
    reme_module = import_module("reme.reme")
    monkeypatch.setattr(reme_module, "running_app_config", lambda: config)

    async def run():
        app = Application(
            **config,
            workspace_dir=str(tmp_path),
            enable_logo=False,
            log_to_console=False,
            log_to_file=False,
        )
        service = app.context.service
        service.build_service(app)
        service.add_jobs(app)
        assert "/decide" in {route.path for route in service.service.routes}
        assert service.mcp_server is not None
        assert "decide" in {tool.name for tool in await service.mcp_server.list_tools()}

        async def start_client(client):
            client.client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=service.service),
                base_url=client.base_url,
            )

        monkeypatch.setattr(HttpClient, "_start", start_client)
        async with app:
            await reme_module.call_server("decide", state="I was charged twice.")
            displayed = json.loads(capsys.readouterr().out)
            assert displayed["success"] is True
            assert displayed["answer"]["content"]["urgent"]["probability"] == 0.8
            assert calls[0][0] == "I was charged twice."
            assert {q["type"] for q in calls[0][1].values()} == {"binary", "choice", "score"}
            custom = {"only": {"type": "binary", "instructions": "Is this urgent?"}}
            await reme_module.call_server("decide", state={"request": "hello"}, questions=custom)
            assert calls[1] == ({"request": "hello"}, custom)
            assert json.loads(capsys.readouterr().out)["success"] is True

    asyncio.run(run())


def test_import_with_optional_modules_blocked():
    """A fresh interpreter catches accidental optional imports anywhere in bootstrap."""
    script = """
import sys
from importlib.abc import MetaPathFinder
class BlockOptional(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'agentscope.classifier' or fullname.startswith('typesafe_sdk'):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, BlockOptional())
import reme
from reme.components.as_decision_model import AgentScopeAsDecisionModel
from reme.steps.common.decision_demo import DecisionDemoStep
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_extra_reuses_agentscope_pin():
    """Installing core and decision models together must not request different AgentScope versions."""
    config = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
    extras = config["project"]["optional-dependencies"]
    assert extras["decision"] == ["agentscope[classifier-jev]==2.0.11"]
    assert extras["as"] == ["agentscope[model-ollama]==2.0.11"]
    assert "reme-ai[decision]" in extras["full"]
    assert "reme-ai[decision]" not in extras["core"]


@pytest.mark.parametrize(
    "config_name,component_name,job_name",
    [("decision_demo", "default", "decide"), ("decision_compaction", "compaction", "compact_context")],
)
def test_decision_configs_use_adapter_defaults_without_optional_imports(
    decision_env,
    monkeypatch,
    config_name,
    component_name,
    job_name,
    tmp_path,
):
    """Both entry points wire the generic component without loading its SDK."""
    monkeypatch.setattr(provider_module, "import_module", lambda _: pytest.fail("Unexpected optional import"))
    config = resolve_app_config(config=config_name, log_config=False)
    component = config["components"]["as_decision_model"][component_name]
    assert component["backend"] == "agentscope"
    assert component["model"] is None
    assert config["jobs"][job_name]["steps"][0]["as_decision_model"] == component_name
    app = Application(**config, workspace_dir=str(tmp_path), enable_logo=False, log_to_file=False)
    model = app.context.components[ComponentEnum.AS_DECISION_MODEL][component_name]
    assert isinstance(model, AgentScopeAsDecisionModel)
    assert model.options.model is None
