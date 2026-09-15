"""Validate the built-in cross-plugin cookbook application."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from reme.components import ApplicationContext
from reme.components.agent_wrapper import CcAgentWrapper
from reme.config.config_parser import _load_config, deep_merge_config, expand_env_vars
from reme.schema import ApplicationConfig

REPOSITORY = Path(__file__).resolve().parents[2]


def _cookbook(monkeypatch) -> dict:
    credentials = {
        "DINGTALK_APP_KEY": "app-key",
        "DINGTALK_APP_SECRET": "app-secret",
        "DINGTALK_ROBOT_CODE": "robot-code",
        "DINGTALK_CONVERSATION_IDS": "group-one,group-two",
        "LLM_API_KEY": "llm-api-key",
        "EMBEDDING_API_KEY": "embedding-api-key",
    }
    for name in (
        "LLM_BACKEND",
        "LLM_MODEL_NAME",
        "LLM_BASE_URL",
        "CLAUDE_CODE_BASE_URL",
        "EMBEDDING_BACKEND",
        "EMBEDDING_MODEL_NAME",
        "EMBEDDING_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in credentials.items():
        monkeypatch.setenv(name, value)
    return _load_config("cookbook")


def test_cookbook_extends_default_and_enables_composed_plugins(monkeypatch):
    """The named variant inherits the normal service and declares plugin load order."""
    config = _cookbook(monkeypatch)

    assert config["service"]["backend"] == "http"
    assert config["plugins"] == ["auto-fin", "daily-paper", "dingtalk"]


def test_cookbook_requires_dingtalk_application_credentials(monkeypatch):
    """A configured background bridge fails fast instead of supervising empty credentials."""
    for name in ("DINGTALK_APP_KEY", "DINGTALK_APP_SECRET", "DINGTALK_ROBOT_CODE"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="undefined env var: DINGTALK_APP_KEY"):
        _load_config("cookbook")


def test_cookbook_enables_embedding_and_separate_agent_backends(monkeypatch):
    """The composed application enables vector search and isolates the DingTalk Claude Code bridge."""
    components = _cookbook(monkeypatch)["components"]

    assert components["as_embedding"]["default"] == {
        "backend": "openai",
        "model": "text-embedding-v4",
        "dimensions": 1024,
        "credential": {
            "api_key": "embedding-api-key",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
        "parameters": {},
    }
    assert components["embedding_store"]["default"] == {
        "backend": "local",
        "as_embedding": "default",
    }
    assert components["file_store"]["default"]["embedding_store"] == "default"

    assert components["as_llm"]["default"]["backend"] == "openai"
    assert components["as_llm"]["default"]["model"] == "qwen3.8-max"
    assert components["as_llm"]["default"]["credential"] == {
        "api_key": "llm-api-key",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }
    assert components["agent_wrapper"]["default"]["backend"] == "agentscope"
    assert components["agent_wrapper"]["claude_code"] == {
        "backend": "claude_code",
        "model": "qwen3.8-max",
        "api_key": "llm-api-key",
        "base_url": "https://dashscope.aliyuncs.com/apps/anthropic",
        "permission_mode": "bypassPermissions",
    }


def test_cookbook_dingtalk_keeps_claude_tools_and_adds_only_memory_search(monkeypatch, tmp_path):
    """The final SDK options retain native tools, disable WebSearch, and add only ReMe search."""
    config = _cookbook(monkeypatch)
    wrapper_config = dict(config["components"]["agent_wrapper"]["claude_code"])
    wrapper_config.pop("backend")
    wrapper = CcAgentWrapper(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    search = SimpleNamespace(name="search", description="Search memory", parameters={})
    monkeypatch.setattr(wrapper, "_resolve_job_tools", lambda _names: [search])

    opts = wrapper._build_options(  # pylint: disable=protected-access
        "hello",
        **wrapper_config,
        job_tools=config["jobs"]["dingtalk_wait"]["steps"][0]["job_tools"],
    )

    assert opts.tools is None
    assert opts.disallowed_tools == ["WebSearch"]
    assert opts.permission_mode == "bypassPermissions"
    assert opts.allowed_tools == ["search"]
    assert set(opts.mcp_servers) == {wrapper.MCP_SERVER_NAME}


def test_cookbook_appends_dingtalk_to_business_pipelines(monkeypatch):
    """Both manual and cron pipelines deliver their final tagged report."""
    jobs = _cookbook(monkeypatch)["jobs"]
    auto_fin_steps = jobs["auto_fin"]["steps"]
    daily_paper_steps = jobs["daily_paper"]["steps"]

    assert [step["backend"] for step in auto_fin_steps] == [
        "auto_fin_data_step",
        "auto_fin_topic_step",
        "auto_fin_merge_step",
        "auto_tag_step",
        "dingtalk_markdown_send_step",
    ]
    assert jobs["auto_fin_cron"]["steps"] == auto_fin_steps
    assert auto_fin_steps[-1]["title"] == "ReMe Auto Fin"

    assert [step["backend"] for step in daily_paper_steps] == [
        "daily_paper_collect_step",
        "daily_paper_rank_step",
        "daily_paper_select_step",
        "daily_paper_analyze_step",
        "daily_paper_digest_step",
        "auto_tag_step",
        "dingtalk_markdown_send_step",
    ]
    assert jobs["daily_paper_cron"]["steps"] == daily_paper_steps
    assert daily_paper_steps[-1]["input_mapping"] == {
        "daily_paper_digest_path": "markdown_path",
    }
    assert daily_paper_steps[-1]["title"] == "ReMe Daily Paper"
    for step in (auto_fin_steps[-1], daily_paper_steps[-1]):
        assert step["app_key"] == "app-key"
        assert step["app_secret"] == "app-secret"
        assert step["robot_code"] == "robot-code"
        assert step["conversation_ids"] == "group-one,group-two"


def test_cookbook_owns_safe_send_and_background_bridge_jobs(monkeypatch):
    """Cross-plugin orchestration owns the private sender and long-running bridge."""
    jobs = _cookbook(monkeypatch)["jobs"]

    send = jobs["dingtalk_send"]
    assert send["backend"] == "base"
    assert send["enable_serve"] is False
    assert send["parameters"]["required"] == ["markdown_path"]

    wait = jobs["dingtalk_wait"]
    assert wait["backend"] == "background"
    assert wait["supervisor"] is True
    assert wait["close_timeout"] == 10
    assert wait["steps"] == [
        {
            "backend": "dingtalk_wait_step",
            "agent_wrapper": "claude_code",
            "app_key": "app-key",
            "app_secret": "app-secret",
            "robot_code": "robot-code",
            "worker_count": 4,
            "job_tools": ["search"],
        },
    ]


def test_cookbook_overrides_merge_with_pure_plugin_defaults(monkeypatch):
    """Step overrides retain the business plugins' parameters and schedules."""
    application = {}
    manifests = (
        REPOSITORY / "plugins" / "auto-fin" / "src" / "reme_auto_fin" / "plugin.yaml",
        REPOSITORY / "plugins" / "daily_paper" / "src" / "reme_daily_paper" / "plugin.yaml",
        REPOSITORY / "plugins" / "dingtalk" / "src" / "reme_dingtalk" / "plugin.yaml",
    )
    for path in manifests:
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        application = deep_merge_config(application, expand_env_vars(manifest.get("application_defaults") or {}))
    application = deep_merge_config(application, _cookbook(monkeypatch))

    config = ApplicationConfig(**application)

    assert config.jobs["auto_fin"].backend == "base"
    assert config.jobs["auto_fin"].parameters["properties"]["topics"]["default"] == "黄金,机器人,半导体"
    assert config.jobs["auto_fin_cron"].backend == "cron"
    assert config.jobs["auto_fin_cron"].model_extra["cron"] == "0 18 * * *"
    assert config.jobs["daily_paper"].backend == "base"
    assert config.jobs["daily_paper_cron"].backend == "cron"
    assert config.jobs["daily_paper_cron"].model_extra["cron"] == "0 8 * * *"
    assert config.jobs["dingtalk_wait"].backend == "background"
