"""Standalone test workspaces use pinned sources and isolated evaluation lifecycles."""

# pylint: disable=missing-function-docstring,unused-argument,protected-access,redefined-outer-name

import copy
from importlib import import_module
import json
from pathlib import Path
from subprocess import CalledProcessError
from unittest.mock import AsyncMock

import pytest
import yaml

from reme_home.evaluation.artifacts import read_json
from reme_home.evaluation.inputs import EvaluationSpec, load_inputs
from reme_home.preparation.code import tree_fingerprint
from reme_home.preparation.evaluation_source import EvaluationSource
from reme_home.evaluation import component as evaluation

from test_evaluation import FakeFactory

from test_preparation import (  # noqa: F401  # pylint: disable=unused-import
    command,
    home_application,
    preparation_inputs,
    snapshot,
)

from benchmark import home_eval
from reme.schema import Response
from reme.components.agent_wrapper import AsAgentWrapper


def write_config(tmp_path, app, spec, commit):
    config = {
        "source_workspace": str(app.config.workspace_dir),
        "test_workspace": str(tmp_path / "test-workspace"),
        "code_commit": commit,
        "benchmark": copy.deepcopy(spec["benchmark"]),
        "evaluation": {"concurrency": 2, "timeout_seconds": 60},
    }
    path = tmp_path / "test.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, config


async def prepare_source(app, inputs):
    spec, _ = inputs
    response = await app.run_job("home_prepare", spec=spec)
    assert response.success, response.answer
    home = Path(app.config.workspace_dir) / "home"
    return home, response.answer["code"]["initial_commit"]


@pytest.mark.asyncio
async def test_import_historical_commit_ignores_dirty_checkout_and_missing_original_repo(app, inputs, tmp_path):
    home, _ = await prepare_source(app, inputs)
    repo = home / "code/repo"
    command(repo, "switch", "-c", "candidate")
    file = repo / "plugins/beam/src/reme_beam/__init__.py"
    file.write_text("value = 2\n")
    file.chmod(0o755)
    command(repo, "add", ".")
    command(repo, "-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "-m", "candidate")
    selected = command(repo, "rev-parse", "HEAD")
    file.write_text("value = 3\n")
    (repo / "untracked.txt").write_text("must not export")
    spec, _ = inputs
    original = Path(spec["runtime"]["source_repo"])
    original.rename(tmp_path / "original-unavailable")
    path, _ = write_config(tmp_path, app, spec, selected[:10])
    before = snapshot(home)

    result = await home_eval.run(path, prepare_only=True)

    assert result["source_commit_sha"] == selected
    target = Path(result["test_workspace"]) / "home"
    imported = target / "code/repo/plugins/beam/src/reme_beam/__init__.py"
    assert imported.read_text() == "value = 2\n"
    assert imported.stat().st_mode & 0o111
    assert not (target / "code/repo/untracked.txt").exists()
    assert command(target / "code/repo", "branch", "--show-current") == "init"
    assert tree_fingerprint(target / "code/runtime") == tree_fingerprint(home / "code/runtime")
    assert read_json(target / "metadata/preparation.json")["test_evaluation"]["source_commit_sha"] == selected
    assert not (target / "datasets/debug-cases/manifest.json").exists()
    assert snapshot(home) == before


@pytest.mark.asyncio
async def test_reopen_scores_all_queries_without_optimizer_history_and_closes(app, inputs, tmp_path, monkeypatch):
    home, commit = await prepare_source(app, inputs)
    path, _ = write_config(tmp_path, app, inputs[0], commit)
    await home_eval.run(path, prepare_only=True)
    applications = []
    factory = home_eval.Application

    def tracked_application(**kwargs):
        application = factory(**kwargs)
        application.close = AsyncMock(wraps=application.close)

        async def evaluate(name, **options):
            assert name == "home_evaluate"
            assert options["record_history"] is False
            assert options["concurrency"] == 2
            assert not options.get("sampling") and not options.get("debug")
            record, _, cases = load_inputs(Path(kwargs["workspace_dir"]) / "home", EvaluationSpec(**options))
            assert len(cases) == record["dataset"]["case_count"] == 2
            assert sum(len(case.queries) for case in cases) == 2
            return Response(answer={"result_dir": "test-results"})

        application.run_job = AsyncMock(side_effect=evaluate)
        applications.append(application)
        return application

    monkeypatch.setattr(home_eval, "Application", tracked_application)
    result = await home_eval.run(path)
    assert result["evaluation"] == {"result_dir": "test-results"}
    applications[-1].close.assert_awaited_once()
    assert not list((home / "evaluations").rglob("manifest.json"))
    assert not list((home / "sampling").rglob("*.json"))


@pytest.mark.asyncio
@pytest.mark.parametrize("app", ["beam-judge", "lme-judge"], indirect=True)
async def test_full_evaluation_records_source_provenance_with_fake_docker(app, inputs, tmp_path, monkeypatch):
    spec, lme = inputs
    if "lme-judge" in app.config.plugins:
        spec["benchmark"] = {"name": "longmemeval", "source": str(lme), "all_cases": True}
    else:
        for file in Path(spec["benchmark"]["source"]).glob("chats/*/*/probing_questions/probing_questions.json"):
            data = read_json(file)
            data["fact"][0]["rubric"] = ["says hello"]
            file.write_text(json.dumps(data))
    home, commit = await prepare_source(app, inputs)
    path, _ = write_config(tmp_path, app, spec, commit)
    monkeypatch.setattr(evaluation, "DockerReMeSandboxFactory", FakeFactory)
    answer = "yes" if spec["benchmark"]["name"] == "longmemeval" else '{"score": 0.5}'
    monkeypatch.setattr(AsAgentWrapper, "reply", AsyncMock(return_value={"result": answer}))
    before = snapshot(home)
    result = await home_eval.run(path)
    run = Path(result["evaluation"]["result_dir"])
    manifest = read_json(run / "manifest.json")
    assert manifest["test_evaluation"]["source_commit_sha"] == commit
    assert manifest["commit_sha"] == result["evaluation_commit_sha"]
    assert manifest["execution_mode"] == "full_validation"
    assert manifest["options"]["record_history"] is False
    assert result["evaluation"]["scored_query_count"] == result["query_count"]
    assert result["evaluation"]["mean_query_score"] == (1.0 if answer == "yes" else 0.5)
    assert snapshot(home) == before


@pytest.mark.asyncio
async def test_evaluation_failure_closes_application_and_preserves_preparation(app, inputs, tmp_path, monkeypatch):
    _, commit = await prepare_source(app, inputs)
    path, config = write_config(tmp_path, app, inputs[0], commit)
    factory = home_eval.Application
    applications = []

    def failing_application(**kwargs):
        application = factory(**kwargs)
        application.run_job = AsyncMock(return_value=Response(success=False, answer="Infrastructure failure"))
        application.close = AsyncMock(wraps=application.close)
        applications.append(application)
        return application

    monkeypatch.setattr(home_eval, "Application", failing_application)
    with pytest.raises(RuntimeError, match="Infrastructure failure"):
        await home_eval.run(path)
    applications[-1].close.assert_awaited_once()
    assert (Path(config["test_workspace"]) / "home/metadata/preparation.json").is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize("app", ["lme-judge"], indirect=True)
async def test_lme_default_selection_excludes_actual_training_cases(app, inputs, tmp_path):
    spec, lme = inputs
    original = json.loads(lme.read_text())
    held_out = {**original[0], "question_id": "q2"}
    lme.write_text(json.dumps([*original, held_out]))
    spec["benchmark"] = {"name": "longmemeval", "source": str(lme), "case_ids": ["q1"]}
    _, commit = await prepare_source(app, inputs)
    path, config = write_config(tmp_path, app, spec, commit)
    config["benchmark"] = {"name": "longmemeval", "source": str(lme), "all_cases": True}
    config["exclude_training_cases"] = True
    path.write_text(yaml.safe_dump(config))
    result = await home_eval.run(path, prepare_only=True, expected_benchmark="longmemeval")
    target = Path(result["test_workspace"]) / "home"
    _, dataset, cases = load_inputs(target, EvaluationSpec(record_history=False))
    assert dataset.case_ids == ["q2"]
    assert [case.case_id for case in cases] == ["q2"]
    assert result["case_count"] == result["query_count"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["candidate", "dataset", "runtime", "test_code", "test_data"])
async def test_reopen_rejects_changed_identity_or_artifacts(app, inputs, tmp_path, change):
    _, commit = await prepare_source(app, inputs)
    path, config = write_config(tmp_path, app, inputs[0], commit)
    result = await home_eval.run(path, prepare_only=True)
    home = Path(result["test_workspace"]) / "home"
    if change == "candidate":
        config["code_commit"] = "1234567890"
        path.write_text(yaml.safe_dump(config))
    elif change == "dataset":
        config["benchmark"]["case_ids"] = ["1"]
        path.write_text(yaml.safe_dump(config))
    elif change == "runtime":
        (home / "code/runtime/reme/__init__.py").write_text("tampered")
    elif change == "test_code":
        (home / "code/repo/plugins/beam/src/reme_beam/__init__.py").write_text("tampered")
    else:
        file = next((home / "datasets/training-set/cases").glob("*.json"))
        case = read_json(file)
        case["queries"][0]["question"] = "tampered"
        file.write_text(json.dumps(case))
    before = snapshot(home)
    with pytest.raises((ValueError, CalledProcessError)):
        await home_eval.run(path, prepare_only=True)
    assert snapshot(home) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("location", ["same", "child", "parent"])
async def test_rejects_overlapping_workspaces_before_start(app, inputs, tmp_path, monkeypatch, location):
    _, commit = await prepare_source(app, inputs)
    path, config = write_config(tmp_path, app, inputs[0], commit)
    source = Path(app.config.workspace_dir)
    target = {"same": source, "child": source / "test", "parent": source.parent}[location]
    config["test_workspace"] = str(target)
    path.write_text(yaml.safe_dump(config))
    factory = AsyncMock()
    monkeypatch.setattr(home_eval, "Application", factory)
    with pytest.raises(ValueError, match="non-nested"):
        await home_eval.run(path)
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_import_runtime_drift_rolls_back_and_allows_retry(app, inputs, tmp_path, monkeypatch):
    source_home, commit = await prepare_source(app, inputs)
    path, config = write_config(tmp_path, app, inputs[0], commit)
    install = EvaluationSource.install

    def fail_install(self, runtime, candidate):
        runtime.mkdir()
        (runtime / "partial").write_text("partial import")
        raise ValueError("Source fixed runtime changed during import")

    monkeypatch.setattr(EvaluationSource, "install", fail_install)
    with pytest.raises(ValueError, match="during import"):
        await home_eval.run(path, prepare_only=True)
    target = Path(config["test_workspace"]) / "home"
    assert not (target / "metadata/preparation.json").exists()
    assert not list((target / "code/runtime").iterdir())
    assert not list((target / "code/repo").iterdir())
    monkeypatch.setattr(EvaluationSource, "install", install)
    await home_eval.run(path, prepare_only=True)
    assert tree_fingerprint(target / "code/runtime") == tree_fingerprint(source_home / "code/runtime")


@pytest.mark.parametrize("benchmark", ["beam", "longmemeval"])
@pytest.mark.asyncio
async def test_benchmark_entry_points_select_correct_defaults(benchmark, monkeypatch):
    module = import_module(f"benchmark.{benchmark}.run_home_eval")
    runner = AsyncMock(return_value={"status": "completed"})
    monkeypatch.setattr(module, "run_evaluation", runner)
    assert await module.run(commit="abcdef0") == {"status": "completed"}
    runner.assert_awaited_once_with(module.DEFAULT_CONFIG, expected_benchmark=benchmark, commit="abcdef0")


@pytest.mark.parametrize("options", [{"sampling": True}, {"record_history": True}, {"max_evaluations": 1}])
def test_test_runner_rejects_optimizer_modes(options):
    with pytest.raises(ValueError):
        home_eval.TestRunConfig.model_validate(
            {
                "source_workspace": "/source",
                "test_workspace": "/test",
                "code_commit": "abcdef0",
                "benchmark": {"name": "beam", "source": "/data", "variant": "100K", "all_cases": True},
                "evaluation": {"record_history": False, **options},
            },
        )
