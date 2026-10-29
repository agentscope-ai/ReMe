"""Benchmark Home defaults and lifecycle wiring, without models or dataset writes."""

import hashlib
from importlib import import_module
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml

from reme.schema import Response

from benchmark import home_runner

ROOT = Path(__file__).resolve().parents[3]


def test_preserved_lme_split_and_evidence():
    """Retain dev_meta's exact reviewed partition and annotations."""
    directory = ROOT / "benchmark/longmemeval"
    assert hashlib.sha256((directory / "home_evidence_sessions.jsonl").read_bytes()).hexdigest() == (
        "35655c51210a96b595910923576fa97b93c81abd82e3cfcda801180007c23a7d"
    )
    workspace, spec = home_runner.load_config(directory / "config_home.yaml")
    selected = spec["benchmark"]["case_ids"]
    assert len(selected) == len(set(selected)) == 250
    # Hash the original first-five-group sequence to preserve both IDs and order.
    assert hashlib.sha256("\n".join(selected).encode()).hexdigest() == (
        "f4bee54f5afe9e0b6c6be8f95c7e1a803a3c2ba8183a77c5f8188e1875818e67"
    )
    assert spec["runtime"]["plugins"] == ["lme"]
    assert spec["candidate"]["paths"] == ["plugins/lme/src"]
    assert workspace == directory / "workspaces/home-groups-1-5"


def test_beam_defaults_use_all_100k_cases_from_any_cwd(tmp_path, monkeypatch):
    """Paths and selection do not depend on where Python was launched."""
    monkeypatch.chdir(tmp_path)
    workspace, spec = home_runner.load_config(ROOT / "benchmark/beam/config_home.yaml")
    assert spec["benchmark"]["variant"] == "100K"
    assert spec["benchmark"]["all_cases"] is True
    assert spec["benchmark"]["case_ids"] is None
    assert spec["benchmark"]["source"] == str(ROOT / "benchmark/beam/dataset/BEAM")
    assert spec["runtime"]["source_repo"] == str(ROOT)
    assert spec["runtime"]["plugins"] == ["beam"]
    assert spec["candidate"]["paths"] == ["plugins/beam/src"]
    assert workspace == ROOT / "benchmark/beam/workspaces/home-100K"


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", ["beam", "longmemeval"])
async def test_python_runner_starts_prepares_and_closes(tmp_path, monkeypatch, benchmark):
    """Both Python entry points invoke only home_prepare through a local application."""
    module = import_module(f"benchmark.{benchmark}.run_home")
    application = AsyncMock()
    application.run_job.return_value = Response(answer={"dataset": {"case_count": 1}})
    observed = {}

    def factory(**kwargs):
        observed.update(kwargs)
        return application

    monkeypatch.setattr(home_runner, "Application", factory)
    result = await module.run(workspace_dir=str(tmp_path / "experiment"))
    assert observed["plugins"] == ["home", "beam-judge" if benchmark == "beam" else "lme-judge"]
    assert observed["service"] == {"backend": "cli"}
    assert observed["workspace_dir"] == str(tmp_path / "experiment")
    assert observed["components"]["home.evaluation"]["default"]["sampling_policy"] == {
        "max_cases": 5,
        "max_queries": None,
        "temperature": 1.0,
        "ewma_alpha": 0.3,
        "uniform_exploration": 0.2,
    }
    application.start.assert_awaited_once()
    application.close.assert_awaited_once()
    assert application.run_job.await_count == 1
    assert application.run_job.call_args.args == ("home_prepare",)
    assert result["workspace_dir"] == str(tmp_path / "experiment")


@pytest.mark.asyncio
async def test_runner_validates_and_passes_custom_sampling_policy(tmp_path, monkeypatch):
    """Partial policies merge typed defaults and never leak into preparation inputs."""
    config = yaml.safe_load((ROOT / "benchmark/beam/config_home.yaml").read_text())
    config["sampling_policy"] = {"max_cases": 2, "max_queries": 7}
    path = tmp_path / "home.yaml"
    path.write_text(yaml.safe_dump(config))
    application = AsyncMock()
    application.run_job.return_value = Response(answer={})
    observed = {}

    def factory(**kwargs):
        observed.update(kwargs)
        return application

    monkeypatch.setattr(home_runner, "Application", factory)
    await home_runner.prepare(path)
    policy = observed["components"]["home.evaluation"]["default"]["sampling_policy"]
    assert policy["max_cases"] == 2 and policy["max_queries"] == 7 and policy["ewma_alpha"] == 0.3
    assert "sampling_policy" not in application.run_job.call_args.kwargs["spec"]
    for invalid in ({"max_cases": 0}, {"unknown": 1}, {"max_queries": "7"}):
        config["sampling_policy"] = invalid
        path.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError):
            await home_runner.prepare(path)
    assert application.start.await_count == 1
    assert application.run_job.await_count == 1
    assert application.run_job.call_args.args == ("home_prepare",)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["startup", "preparation"])
async def test_runner_propagates_failure_and_closes(monkeypatch, failure):
    """A failed preparation is never printed or returned as success."""
    application = AsyncMock()
    if failure == "startup":
        application.start.side_effect = RuntimeError("startup failed")
    else:
        application.run_job.return_value = Response(success=False, answer="already prepared")
    monkeypatch.setattr(home_runner, "Application", lambda **kwargs: application)
    with pytest.raises(RuntimeError):
        await home_runner.prepare(ROOT / "benchmark/beam/config_home.yaml")
    application.close.assert_awaited_once()
