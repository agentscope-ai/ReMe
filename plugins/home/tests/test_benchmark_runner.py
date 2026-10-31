"""Benchmark Home defaults and lifecycle wiring, without models or dataset writes."""

import hashlib
import json
import sys
from importlib import import_module
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml

from reme.schema import Response

from benchmark.beam import run_home as beam_runner
from benchmark.longmemeval import run_home as lme_runner

ROOT = Path(__file__).resolve().parents[3]


def test_preserved_lme_split_and_evidence():
    """Retain dev_meta's exact reviewed partition and annotations."""
    directory = ROOT / "benchmark/longmemeval"
    assert hashlib.sha256((directory / "home_evidence_sessions.jsonl").read_bytes()).hexdigest() == (
        "35655c51210a96b595910923576fa97b93c81abd82e3cfcda801180007c23a7d"
    )
    workspace, spec = lme_runner.load_config(directory / "config_home.yaml")
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
    workspace, spec = beam_runner.load_config(ROOT / "benchmark/beam/config_home.yaml")
    assert spec["benchmark"]["variant"] == "100K"
    assert spec["benchmark"]["all_cases"] is True
    assert spec["benchmark"]["case_ids"] is None
    assert spec["benchmark"]["source"] == str(ROOT / "benchmark/beam/dataset/BEAM")
    assert spec["runtime"]["source_repo"] == str(ROOT)
    assert spec["runtime"]["plugins"] == ["beam"]
    assert spec["candidate"]["paths"] == ["plugins/beam/src"]
    assert workspace == ROOT / "benchmark/beam/workspaces/home-100K"


@pytest.mark.parametrize("benchmark", ["beam", "longmemeval"])
def test_runner_preserves_explicit_candidate_paths(tmp_path, benchmark):
    """Runner defaults preserve custom scopes from existing configurations."""
    module = import_module(f"benchmark.{benchmark}.run_home")
    config = yaml.safe_load(module.DEFAULT_CONFIG.read_text())
    config["preparation"]["candidate"] = {"paths": ["reme/steps"]}
    path = tmp_path / "home.yaml"
    path.write_text(yaml.safe_dump(config))
    _, spec = module.load_config(path)
    assert spec["candidate"]["paths"] == ["reme/steps"]


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

    monkeypatch.setattr(module, "Application", factory)
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
    assert observed["components"]["agent_wrapper"]["home_optimizer"] == {
        "validation_cycle_time_budget_minutes": 150,
        "max_validations": 10,
    }
    application.start.assert_awaited_once()
    application.close.assert_awaited_once()
    assert application.run_job.await_count == 1
    assert application.run_job.call_args.args == ("home_prepare",)
    assert result["workspace_dir"] == str(tmp_path / "experiment")


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", ["beam", "longmemeval"])
async def test_runner_validates_and_passes_custom_sampling_policy(tmp_path, monkeypatch, benchmark):
    """Partial policies merge typed defaults and never leak into preparation inputs."""
    module = import_module(f"benchmark.{benchmark}.run_home")
    config = yaml.safe_load(module.DEFAULT_CONFIG.read_text())
    config["sampling_policy"] = {"max_cases": 2, "max_queries": 7}
    path = tmp_path / "home.yaml"
    path.write_text(yaml.safe_dump(config))
    application = AsyncMock()
    application.run_job.return_value = Response(answer={})
    observed = {}

    def factory(**kwargs):
        observed.update(kwargs)
        return application

    monkeypatch.setattr(module, "Application", factory)
    await module.prepare(path)
    policy = observed["components"]["home.evaluation"]["default"]["sampling_policy"]
    assert policy["max_cases"] == 2 and policy["max_queries"] == 7 and policy["ewma_alpha"] == 0.3
    assert "sampling_policy" not in application.run_job.call_args.kwargs["spec"]
    for invalid in ({"max_cases": 0}, {"unknown": 1}, {"max_queries": "7"}):
        config["sampling_policy"] = invalid
        path.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError):
            await module.prepare(path)
    assert application.start.await_count == 1
    assert application.run_job.await_count == 1
    assert application.run_job.call_args.args == ("home_prepare",)


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", ["beam", "longmemeval"])
@pytest.mark.parametrize("budget", ["omitted", 45.5, None])
async def test_runner_passes_cycle_budget_to_optimizer(tmp_path, monkeypatch, benchmark, budget):
    """Both local runners apply the default, custom duration and explicit disable value."""
    module = import_module(f"benchmark.{benchmark}.run_home")
    config = yaml.safe_load(module.DEFAULT_CONFIG.read_text())
    if budget == "omitted":
        config.pop("validation_cycle_time_budget_minutes", None)
    else:
        config["validation_cycle_time_budget_minutes"] = budget
    path = tmp_path / "home.yaml"
    path.write_text(yaml.safe_dump(config))
    application = AsyncMock()
    application.run_job.return_value = Response(answer={})
    observed = {}

    def factory(**kwargs):
        observed.update(kwargs)
        return application

    monkeypatch.setattr(module, "Application", factory)
    await module.run(path)
    expected = 150 if budget == "omitted" else budget
    assert observed["components"]["agent_wrapper"]["home_optimizer"] == {
        "validation_cycle_time_budget_minutes": expected,
        "max_validations": 10,
    }
    assert "validation_cycle_time_budget_minutes" not in application.run_job.call_args.kwargs["spec"]
    application.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", ["beam", "longmemeval"])
@pytest.mark.parametrize("budget", [0, -1, True, "150", float("nan"), float("inf")])
async def test_runner_rejects_invalid_cycle_budget_before_start(tmp_path, monkeypatch, benchmark, budget):
    """Reject nonpositive, nonnumeric and nonfinite durations without creating an application."""
    module = import_module(f"benchmark.{benchmark}.run_home")
    config = yaml.safe_load(module.DEFAULT_CONFIG.read_text())
    config["validation_cycle_time_budget_minutes"] = budget
    path = tmp_path / "home.yaml"
    path.write_text(yaml.safe_dump(config))
    created = []
    monkeypatch.setattr(module, "Application", lambda **kwargs: created.append(kwargs))
    with pytest.raises(ValueError):
        await module.run(path)
    assert not created


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", ["beam", "longmemeval"])
@pytest.mark.parametrize("failure", ["startup", "preparation"])
async def test_runner_propagates_failure_and_closes(monkeypatch, failure, benchmark):
    """A failed preparation is never printed or returned as success."""
    application = AsyncMock()
    if failure == "startup":
        application.start.side_effect = RuntimeError("startup failed")
    else:
        application.run_job.return_value = Response(success=False, answer="already prepared")
    module = import_module(f"benchmark.{benchmark}.run_home")
    monkeypatch.setattr(module, "Application", lambda **kwargs: application)
    with pytest.raises(RuntimeError):
        await module.prepare(module.DEFAULT_CONFIG)
    application.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", ["beam", "longmemeval"])
@pytest.mark.parametrize("limit", ["omitted", 3, 0, -1, True, "10", 1.5, None])
async def test_runner_formal_budget_configuration(tmp_path, monkeypatch, benchmark, limit):
    """Validate host count budgets and pass them only to the optimizer wrapper."""
    module = import_module(f"benchmark.{benchmark}.run_home")
    config = yaml.safe_load(module.DEFAULT_CONFIG.read_text())
    if limit == "omitted":
        config.pop("max_validations", None)
    else:
        config["max_validations"] = limit
    path = tmp_path / "home.yaml"
    path.write_text(yaml.safe_dump(config))
    application = AsyncMock()
    application.run_job.return_value = Response(answer={})
    observed = {}

    def factory(**kwargs):
        observed.update(kwargs)
        return application

    monkeypatch.setattr(module, "Application", factory)
    if limit in ("omitted", 3):
        await module.prepare(path)
        assert observed["components"]["agent_wrapper"]["home_optimizer"]["max_validations"] == (
            10 if limit == "omitted" else limit
        )
        assert "max_validations" not in application.run_job.call_args.kwargs["spec"]
    else:
        with pytest.raises(ValueError):
            await module.prepare(path)
        assert not observed


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", ["beam", "longmemeval"])
@pytest.mark.parametrize("minutes", [30, None])
async def test_optimizer_lifecycle_uses_same_config_and_does_not_prepare(tmp_path, monkeypatch, benchmark, minutes):
    """A separate optimization launch consumes the same count, timer and sampling settings."""
    module = import_module(f"benchmark.{benchmark}.run_home")
    config = yaml.safe_load(module.DEFAULT_CONFIG.read_text())
    config.update(max_validations=3, validation_cycle_time_budget_minutes=minutes)
    path = tmp_path / "home.yaml"
    path.write_text(yaml.safe_dump(config))
    application = AsyncMock()
    application.run_job.return_value = Response(answer={"status": "completed"})
    observed = {}

    def factory(**kwargs):
        observed.update(kwargs)
        return application

    monkeypatch.setattr(module, "Application", factory)
    result = await module.optimize(path, workspace_dir=str(tmp_path / "existing"), instruction="Improve retrieval")
    assert result["status"] == "completed"
    assert observed["components"]["agent_wrapper"]["home_optimizer"] == {
        "max_validations": 3,
        "validation_cycle_time_budget_minutes": minutes,
    }
    application.run_job.assert_awaited_once_with("home_optimize", instruction="Improve retrieval")
    application.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", ["beam", "longmemeval"])
async def test_optimizer_failure_closes_application(monkeypatch, benchmark):
    """Failed optimization retains the normal cleanup and error contract."""
    module = import_module(f"benchmark.{benchmark}.run_home")
    application = AsyncMock()
    application.run_job.return_value = Response(success=False, answer="optimization failed")
    monkeypatch.setattr(module, "Application", lambda **kwargs: application)
    with pytest.raises(RuntimeError, match="optimization failed"):
        await module.optimize()
    application.run_job.assert_awaited_once_with("home_optimize")
    application.close.assert_awaited_once()


@pytest.mark.parametrize("benchmark", ["beam", "longmemeval"])
def test_optimizer_cli_forwards_configuration_and_focus(tmp_path, monkeypatch, capsys, benchmark):
    """The explicit CLI mode reaches optimization and preserves the preparation default."""
    module = import_module(f"benchmark.{benchmark}.run_home")
    config = tmp_path / "home.yaml"
    optimize = AsyncMock(return_value={"status": "completed"})
    prepare = AsyncMock()
    monkeypatch.setattr(module, "optimize", optimize)
    monkeypatch.setattr(module, "prepare", prepare)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_home.py",
            "--optimize",
            "--config",
            str(config),
            "--workspace-dir",
            str(tmp_path / "existing"),
            "--instruction",
            "Improve retrieval",
        ],
    )
    module.main()
    optimize.assert_awaited_once_with(
        config,
        workspace_dir=str(tmp_path / "existing"),
        instruction="Improve retrieval",
    )
    prepare.assert_not_called()
    assert json.loads(capsys.readouterr().out) == {"status": "completed"}


@pytest.mark.parametrize("benchmark", ["beam", "longmemeval"])
def test_cli_rejects_optimization_focus_in_preparation_mode(monkeypatch, benchmark):
    """Do not silently discard an instruction when the caller forgot --optimize."""
    module = import_module(f"benchmark.{benchmark}.run_home")
    monkeypatch.setattr(sys, "argv", ["run_home.py", "--instruction", "Improve retrieval"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
