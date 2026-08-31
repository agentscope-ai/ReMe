"""Plugin registration, packaged presets and offline evaluation contracts."""

# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access

from importlib.metadata import EntryPoint, EntryPoints
from pathlib import Path
import tomllib

import pytest

from reme_beam.config import config_path
from reme_beam.steps.auto_memory import _interpolate_timestamps

from reme import Application
from reme.components.agent_wrapper.base_agent_wrapper import BaseAgentWrapper
from reme.components.component_registry import R
from reme.config import resolve_app_config
from reme.enumeration import ComponentEnum
from reme.plugin import resolve_plugin_runtime
from reme.plugin_manifest import load_package_manifest


@pytest.fixture
def installed_plugin(monkeypatch):
    """Use real entry-point declarations without requiring an editable install."""
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    entries = EntryPoints(
        EntryPoint(name=name, value=value, group=group)
        for group, values in project["project"]["entry-points"].items()
        for name, value in values.items()
    )
    monkeypatch.setattr("reme.entry_point.metadata.entry_points", lambda: entries)


@pytest.mark.usefixtures("installed_plugin")
def test_backends_are_opt_in_and_do_not_change_default():
    manifest = load_package_manifest("reme_beam", plugin_name="beam")
    assert not manifest.application_defaults
    default = resolve_app_config(log_config=False)
    disabled = resolve_plugin_runtime(default)
    config = resolve_app_config(log_config=False, plugins=["beam"])
    enabled = resolve_plugin_runtime(config)
    assert enabled.config == config
    assert enabled.config["jobs"] == default["jobs"]
    assert "dream_cron" in enabled.config["jobs"]
    for backend in manifest.backends:
        assert R.get(ComponentEnum.STEP, backend) is None
        assert disabled.registry.get(ComponentEnum.STEP, backend) is None
        assert enabled.registry.get(ComponentEnum.STEP, backend) is not None


@pytest.mark.parametrize("name", ["beam", "beam.yaml"])
@pytest.mark.usefixtures("installed_plugin")
def test_preset_keeps_manual_evaluation_and_user_overrides(name):
    config = resolve_app_config(
        config=name,
        log_config=False,
        components={"as_llm": {"bench": {"model": "test-model"}}},
    )
    runtime = resolve_plugin_runtime(config)
    assert runtime.config == config
    assert config["plugins"] == ["beam"]
    assert config["components"]["as_llm"]["bench"]["model"] == "test-model"
    jobs = config["jobs"]
    assert all(job["backend"] == "base" for job in jobs.values())
    assert {"auto_dream", "dream_cron", "index_update_loop", "resource_watch_loop"}.isdisjoint(jobs)
    assert {"auto_memory", "agentic_answer", "answer_judge", "index_update", "digest_update"} <= jobs.keys()
    for job_name in ("read", "edit", "write", "frontmatter_update", "daily_write"):
        assert job_name in jobs
    assert {"read_daily", "edit_daily", "write_daily"}.isdisjoint(jobs)


def test_preset_requires_installed_plugin(monkeypatch):
    monkeypatch.setattr("reme.entry_point.metadata.entry_points", EntryPoints)
    config = resolve_app_config(config=str(config_path()), log_config=False)
    with pytest.raises(ValueError, match="Plugin 'beam' is not installed"):
        resolve_plugin_runtime(config)


def test_auto_memory_prompt_retains_date_contract():
    prompt = (config_path().parents[1] / "steps" / "auto_memory.yaml").read_text(encoding="utf-8")
    assert "date={today}" in prompt or "`date`: {today}" in prompt or "`date`：{today}" in prompt


def test_timestamp_interpolation_preserves_historical_time():
    messages = [
        {"role": "user", "content": "first", "created_at": "2024-01-01T10:00:00"},
        {"role": "assistant", "content": "middle"},
        {"role": "user", "content": "last", "created_at": "2024-01-01T12:00:00"},
    ]
    interpolated = _interpolate_timestamps(messages)
    assert [msg["created_at"] for msg in interpolated] == [
        "2024-01-01T10:00:00",
        "2024-01-01T11:00:00",
        "2024-01-01T12:00:00",
    ]
    assert "created_at" not in messages[1]


class _Agent(BaseAgentWrapper):
    def __init__(self, result):
        super().__init__()
        self.result = result
        self.calls = []

    async def reply(self, inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        return {"result": self.result}


@pytest.mark.asyncio
@pytest.mark.usefixtures("installed_plugin")
async def test_existing_jobs_start_and_answer_with_mock_models(tmp_path):
    config = resolve_app_config(config="beam.yaml", log_config=False)
    # Exercise the real Job/Step lifecycle, without starting model or index clients.
    config.update(
        components={},
        service={"backend": "cli"},
        workspace_dir=str(tmp_path),
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
    )
    app = Application(**config)
    async with app:
        answer_agent = _Agent("Paris")
        answer = await app.run_job(
            "agentic_answer",
            query="Where?",
            query_time="2024-01-01T12:00:00",
            compress_session=True,
            agent_wrapper=answer_agent,
        )
        assert answer.success, answer.answer
        assert answer.answer == "Paris"
        assert answer.metadata["query"] == "Where?"
        assert "2024-01-01T12:00:00" in answer.metadata["sys_prompt"]
        call = answer_agent.calls[0][1]
        assert call["job_tools"] == ["search", "add_draft", "read_all_draft", "read"]
        assert call["injected_job_kwargs"]["_search"]["_compress"] == {"session": "true"}
        judge_agent = _Agent('{"score": 0.5, "reason": "partial"}')
        result = await app.run_job(
            "answer_judge",
            llm_response=answer.answer,
            rubric=["Names Paris"],
            probing_question="Where?",
            question_type="information_extraction",
            agent_wrapper=judge_agent,
        )
        assert result.success, result.answer
        assert result.answer == "0.5"
        assert result.metadata["llm_judge_score"] == 0.5
        assert result.metadata["llm_judge_responses"] == [{"score": 0.5, "reason": "partial"}]
