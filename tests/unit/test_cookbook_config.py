"""Validate the built-in cross-plugin cookbook application."""

from pathlib import Path

import pytest
import yaml

from reme.config.config_parser import _load_config, deep_merge_config, expand_env_vars
from reme.schema import ApplicationConfig

REPOSITORY = Path(__file__).resolve().parents[2]


def _cookbook(monkeypatch) -> dict:
    credentials = {
        "DINGTALK_APP_KEY": "app-key",
        "DINGTALK_APP_SECRET": "app-secret",
        "DINGTALK_ROBOT_CODE": "robot-code",
        "DINGTALK_CONVERSATION_IDS": "group-one,group-two",
    }
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
            "agent_wrapper": "default",
            "app_key": "app-key",
            "app_secret": "app-secret",
            "robot_code": "robot-code",
            "worker_count": 4,
            "builtin_tools": False,
            "job_tools": ["search", "read"],
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
