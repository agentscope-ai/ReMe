"""Successful-history ownership and evaluation parameter precedence through real Jobs."""

# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument,protected-access

import json
import asyncio
import os
from pathlib import Path
import subprocess
import sys
from threading import Event
from unittest.mock import AsyncMock

import pytest
from agentscope.message import ToolResultState
from reme_home.evaluation.history import (
    HomeEvaluationHistory,
    EvaluationRecord,
    read_state,
    rebuild_sampling_state,
    start_validation,
)
from reme_home.evaluation.budget import validation_budget_status
from reme_home.evaluation import history as history_module
from reme_home.sampling import SamplingPolicy, create_sampling_plan
from test_evaluation import (  # noqa: F401  # pylint: disable=unused-import
    FakeFactory,
    commit_candidate,
    home_application,
    preparation_inputs,
    prepared_home,
    rebuild_review,
)

from reme.components.agent_wrapper.as_agent_wrapper import AsAgentWrapper

pytestmark = pytest.mark.usefixtures("rebuild_review")


def history(app):
    return app.context.components["home.evaluation_history"]["default"]


def manifest(response):
    return json.loads((Path(response.answer["result_dir"]) / "manifest.json").read_text())


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_completion_waits_for_concurrent_code_browsing(app, prepared, monkeypatch, cancelled):
    from reme_home.code.component import OPERATIONS

    manager = history(app)
    publishing, browsing, release = Event(), Event(), Event()
    original_publish, original_status = manager.publish_completed, OPERATIONS["status"]

    def publish(*args):
        publishing.set()
        assert browsing.wait(5)
        return original_publish(*args)

    def status(*args, **kwargs):
        browsing.set()  # HomeCode owns workspace.lock before dispatching here.
        assert release.wait(5)
        return original_status(*args, **kwargs)

    monkeypatch.setattr(manager, "publish_completed", publish)
    monkeypatch.setitem(OPERATIONS, "status", status)
    evaluation = asyncio.create_task(app.run_job("home_evaluate"))
    code_status = None
    try:
        assert await asyncio.to_thread(publishing.wait, 5)
        code_status = asyncio.create_task(app.run_job("home_code_status"))
        assert await asyncio.to_thread(browsing.wait, 5)
        await asyncio.sleep(0.1)
        assert not evaluation.done()
        if cancelled:
            evaluation.cancel("original cancellation")
            await asyncio.sleep(0)
            evaluation.cancel("repeated cancellation")
            await asyncio.sleep(0)
            assert not evaluation.done()
        release.set()
        if cancelled:
            with pytest.raises(asyncio.CancelledError, match="original cancellation"):
                await evaluation
        else:
            response = await evaluation
            assert response.success, response.answer
            assert manifest(response)["status"] == "completed"
        assert (await code_status).success
        assert len(manager.list_results()) == 1
        state = read_state(prepared)
        manager.rebuild()
        assert read_state(prepared) == state and state["successful_round"] == 1
    finally:
        release.set()
        await asyncio.gather(evaluation, *([code_status] if code_status is not None else []), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["record_started", "record", "invalidate"])
async def test_history_writes_wait_for_code_read_without_losing_result(app, prepared, monkeypatch, stage):
    from reme_home.code.component import OPERATIONS

    commit_candidate(prepared)
    manager = history(app)
    reached, browsing, acquiring, release = Event(), Event(), Event(), Event()
    original, original_status = getattr(manager, stage), OPERATIONS["status"]

    def write(*args, **kwargs):
        reached.set()
        assert browsing.wait(5)
        acquiring.set()
        return original(*args, **kwargs)

    def status(*args, **kwargs):
        browsing.set()  # The real code Job already owns workspace.lock.
        assert release.wait(5)
        return original_status(*args, **kwargs)

    if stage == "invalidate":

        def fail_record(*args):
            raise OSError("injected registration failure after completion publication")

        monkeypatch.setattr(manager, "record", fail_record)
    monkeypatch.setattr(manager, stage, write)
    monkeypatch.setitem(OPERATIONS, "status", status)
    evaluation = asyncio.create_task(app.run_job("home_evaluate", max_evaluations=1))
    code_status = None
    try:
        assert await asyncio.to_thread(reached.wait, 5)
        code_status = asyncio.create_task(app.run_job("home_code_status"))
        assert await asyncio.to_thread(acquiring.wait, 5)
        await asyncio.sleep(0.1)
        assert not evaluation.done()
        # This coroutine must remain runnable while the history worker waits.
        release.set()
        response = await evaluation
        assert response.success, response.answer
        assert (await code_status).success
        payload = manifest(response)
        assert payload["status"] == "completed" and payload["success"] is True
        plan = json.loads((prepared / payload["sampling"]["plan_path"]).read_text())
        assert payload["sandbox_started_at"] == plan["sandbox_started_at"]
        budget = validation_budget_status(prepared, 1)
        assert budget["used_validations"] == 1 and budget["remaining_validations"] == 0
        assert response.answer["validation_budget"] == budget
        state = read_state(prepared)
        assert state["round"] == state["successful_round"] == 1
        assert state["started_validations"] == {plan["validation_key"]: 1}
        assert len(manager.list_results()) == 1
        manager.rebuild()
        assert read_state(prepared) == state
        retry = await app.run_job("home_evaluate")
        assert retry.answer["error_code"] == "UNCHANGED_CANDIDATE"
    finally:
        release.set()
        await asyncio.gather(evaluation, *([code_status] if code_status is not None else []), return_exceptions=True)


@pytest.mark.asyncio
async def test_start_cache_failure_keeps_start_evidence_and_exhausts_budget(app, prepared, monkeypatch):
    commit_candidate(prepared)
    manager = history(app)
    manager.rebuild()  # Start with a valid but empty cache, not just a missing cache.
    original = history_module.write_json

    def fail_cache(path, value, **kwargs):
        if path == prepared / "sampling/state.json":
            raise OSError("injected start cache publication failure")
        return original(path, value, **kwargs)

    monkeypatch.setattr(history_module, "write_json", fail_cache)
    response = await app.run_job("home_evaluate", max_evaluations=1, sampling=True)
    assert not response.success
    paths = list((prepared / "evaluations").glob("*/*/*/manifest.json"))
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text())
    assert payload["status"] == "failed" and payload["sandbox_started_at"]
    plan = json.loads((prepared / payload["sampling"]["plan_path"]).read_text())
    assert payload["sandbox_started_at"] == plan["sandbox_started_at"]
    assert payload["sampling"]["round"] == plan["round"] == 1
    assert not (prepared / "sampling/state.json").exists()
    expected = {plan["validation_key"]: 1}
    assert read_state(prepared)["started_validations"] == expected
    assert rebuild_sampling_state(prepared, persist=False)["started_validations"] == expected
    budget = validation_budget_status(prepared, 1)
    assert budget["used_validations"] == 1 and budget["remaining_validations"] == 0
    rejected = await app.run_job("home_evaluate", max_evaluations=1)
    assert rejected.answer["error_code"] == "VALIDATION_BUDGET_EXHAUSTED"
    assert len(FakeFactory.instances) == 1
    assert manager.list_results() == []


@pytest.mark.parametrize("interrupted_write", ["manifest.json", "state.json"])
def test_interrupted_start_rebuilds_from_plan_and_retry_is_idempotent(tmp_path, monkeypatch, interrupted_write):
    class Interrupted(BaseException):
        """Model process interruption outside ordinary exception handling."""

    home = tmp_path / "home"
    run = home / "evaluations/candidate/abcdef0/1"
    run.mkdir(parents=True)
    original = history_module.write_json
    original(
        run / "manifest.json",
        {
            "mode": "scored",
            "dataset_kind": "training-set",
            "status": "running",
            "validation_id": "1",
            "case_ids": ["1"],
            "query_ids": {"1": ["q1"]},
        },
    )
    rebuild_sampling_state(home)  # An old cache must not hide newly published evidence.

    def interrupt(path, value, **kwargs):
        if path.name == interrupted_write:
            raise Interrupted()
        return original(path, value, **kwargs)

    monkeypatch.setattr(history_module, "write_json", interrupt)
    with pytest.raises(Interrupted):
        start_validation(home, run, SamplingPolicy())
    plan_path = home / "sampling/plans/candidate/abcdef0/1.json"
    evidence = plan_path.read_bytes()
    plan = json.loads(evidence)
    assert not (home / "sampling/state.json").exists()
    if interrupted_write == "manifest.json":
        assert "sandbox_started_at" not in json.loads((run / "manifest.json").read_text())
    assert read_state(home)["started_validations"] == {"candidate/abcdef0/1": 1}
    assert validation_budget_status(home, 1)["remaining_validations"] == 0
    monkeypatch.setattr(history_module, "write_json", original)
    assert start_validation(home, run, SamplingPolicy()) == plan
    assert plan_path.read_bytes() == evidence
    assert json.loads((run / "manifest.json").read_text())["sandbox_started_at"] == plan["sandbox_started_at"]
    assert read_state(home)["round"] == 1
    assert rebuild_sampling_state(home)["started_validations"] == {"candidate/abcdef0/1": 1}


@pytest.mark.asyncio
async def test_history_catalog_keeps_successful_low_scores_and_deduplicates(app, prepared):
    manager = history(app)
    assert isinstance(manager, HomeEvaluationHistory) and manager.is_started
    assert manager.list_results() == []
    app.context.components["agent_wrapper"]["judge"].reply.return_value = {"result": '{"score": 0}'}
    response = await app.run_job("home_evaluate")
    assert response.success, response.answer
    results = manager.list_results()
    assert len(results) == 1 and isinstance(results[0], EvaluationRecord)
    assert results[0].mean_query_score == 0 and results[0].scored_query_count == 2
    assert manager.get_result(results[0].run_key) == results[0]
    run = Path(response.answer["result_dir"])
    before = manager.sampling_state()
    assert manager.record(run) == results[0]
    assert len(manager.list_results()) == 1 and manager.sampling_state() == before
    assert not (prepared / "evaluations/history.json").exists()
    # A stale or corrupt legacy index cannot hide authoritative results.
    legacy = prepared / "evaluations/history.json"
    legacy.write_text("obsolete index")
    assert manager.list_results() == results
    assert manager.rebuild() == results
    assert legacy.read_text() == "obsolete index"
    debug = await app.run_job("home_evaluate", sampling=True, debug=True)
    assert debug.success and manager.record(Path(debug.answer["result_dir"])) is None
    assert manager.list_results() == results
    with pytest.raises(KeyError):
        manager.get_result("unknown")


@pytest.mark.asyncio
async def test_failed_run_does_not_change_next_probabilities_or_successful_history(app, prepared, monkeypatch):
    baseline = await app.run_job("home_evaluate")
    assert baseline.success
    manager = history(app)
    records, observations = manager.list_results(), manager.sampling_state()
    policy = SamplingPolicy()
    from reme_home.preparation.data.models import CaseSpec

    cases = [
        CaseSpec.model_validate_json(path.read_text())
        for path in sorted(
            (prepared / "datasets/training-set/cases").glob("*.json"),
        )
    ]
    monkeypatch.setattr("reme_home.sampling.time.time_ns", lambda: 5678)
    before = create_sampling_plan(cases, observations, policy)
    original = FakeFactory.create_cases

    async def create(factory, *args, **kwargs):
        workers = await original(factory, *args, **kwargs)
        for worker in workers:
            build = worker.run_build

            async def fail_one(jobs, worker=worker, build=build):
                if worker.case_id == "2":
                    raise ConnectionError("only one case failed")
                return await build(jobs)

            worker.run_build = fail_one
        return workers

    monkeypatch.setattr(FakeFactory, "create_cases", create)
    commit_candidate(prepared)
    failed = await app.run_job("home_evaluate", concurrency=1)
    assert not failed.success
    assert failed.answer["scored_query_count"] == 1  # Its healthy case is also excluded from feedback.
    assert manager.record(Path(failed.answer["result_dir"])) is None
    assert manager.list_results() == records and manager.sampling_state() == observations
    assert create_sampling_plan(cases, manager.sampling_state(), policy) == before
    assert read_state(prepared)["round"] == 2
    assert read_state(prepared)["successful_round"] == 1
    assert manager.rebuild() == records
    assert manager.sampling_state() == observations


@pytest.mark.asyncio
async def test_history_and_observations_rebuild_without_touching_artifacts(app, prepared):
    baseline = await app.run_job("home_evaluate")
    commit_candidate(prepared)
    sampled = await app.run_job("home_evaluate", sampling=True, sampling_policy={"max_cases": 1})
    assert baseline.success and sampled.success
    manager = history(app)
    expected, observations = manager.list_results(), manager.sampling_state()
    source = Path(sampled.answer["result_dir"]) / "summary.json"
    original = source.read_bytes()
    (prepared / "sampling/state.json").unlink()
    assert manager.list_results() == expected
    assert manager.sampling_state() == observations
    assert manager.rebuild() == expected
    assert source.read_bytes() == original
    assert not (prepared / "evaluations/history.json").exists()
    assert (prepared / "sampling/state.json").is_file()


@pytest.mark.asyncio
async def test_call_policy_merges_component_defaults_without_mutating_them(app, prepared):
    evaluator = app.context.components["home.evaluation"]["default"]
    evaluator.sampling_policy = SamplingPolicy(max_cases=2, max_queries=2, temperature=0.7, ewma_alpha=0.6)
    response = await app.run_job("home_evaluate", sampling=True, sampling_policy={"max_cases": 1})
    assert response.success, response.answer
    options = manifest(response)["options"]
    assert options["sampling"] is True
    assert "sampling_policy" not in options
    plan = json.loads((prepared / manifest(response)["sampling"]["plan_path"]).read_text())
    assert plan["policy"] == {
        "max_cases": 1,
        "max_queries": 2,
        "temperature": 0.7,
        "ewma_alpha": 0.6,
        "uniform_exploration": 0.2,
    }
    assert response.answer["case_count"] == 1
    commit_candidate(prepared)
    following = await app.run_job("home_evaluate", sampling=True)
    assert following.success and following.answer["case_count"] == 2
    assert evaluator.sampling_policy.max_cases == 2
    plan = json.loads((prepared / manifest(following)["sampling"]["plan_path"]).read_text())
    assert plan["policy"]["max_cases"] == 2
    for kwargs in (
        {"sampling": "true"},
        {"sampling": {"max_cases": 1}},
        {"record_history": "false"},
        {"sampling_policy": {"max_cases": 0}},
        {"sampling_policy": {"unknown": 1}},
    ):
        assert not (await app.run_job("home_evaluate", **kwargs)).success
    assert len(history(app).list_results()) == 2


@pytest.mark.asyncio
async def test_job_defaults_and_hidden_agent_injection_use_one_job(app, prepared):
    job = app.context.jobs["home_evaluate"]
    assert "home_validate" not in app.context.jobs
    job.kwargs.update(sampling=True, sampling_policy={"max_cases": 1})
    default = await app.run_job("home_evaluate")
    assert default.success and default.answer["case_count"] == 1
    commit_candidate(prepared)
    explicit = await app.run_job("home_evaluate", sampling=False)
    assert explicit.success and explicit.answer["case_count"] == 2
    tool = AsAgentWrapper._make_tool(
        job,
        injected_job_kwargs={
            "sampling": True,
            "sampling_policy": {"max_cases": 1, "max_queries": 1},
        },
    )
    assert "sampling" not in tool.input_schema["properties"]
    commit_candidate(prepared)
    result = await tool.call()
    assert result.state == ToolResultState.SUCCESS, result
    records = history(app).list_results()
    assert records[-1].case_count == records[-1].scored_query_count == 1
    assert len(records) == 3
    with pytest.raises(ValueError, match="injected"):
        await tool.call(sampling=False)
    with pytest.raises(ValueError, match="injected"):
        await tool.call(sampling_policy={"max_cases": 100})
    assert len(history(app).list_results()) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("baseline", [False, True])
async def test_standalone_full_evaluation_preserves_history_and_rebuild_excludes_it(app, prepared, baseline):
    manager = history(app)
    if baseline:
        assert (await app.run_job("home_evaluate")).success
    records, observations = manager.list_results(), manager.sampling_state()
    cache_paths = [prepared / name for name in ("sampling/state.json",)]
    caches = [path.read_bytes() if path.exists() else None for path in cache_paths]
    if baseline:
        commit_candidate(prepared)
    response = await app.run_job("home_evaluate", sampling=False, record_history=False)
    assert response.success, response.answer
    assert response.answer["case_count"] == response.answer["scored_query_count"] == 2
    assert response.answer["mean_query_score"] == 0.5 and response.answer["record_history"] is False
    run = Path(response.answer["result_dir"])
    payload = manifest(response)
    assert payload["status"] == "completed" and payload["success"] is True
    assert payload["options"]["record_history"] is False and payload["sandbox_started_at"]
    assert "sampling" not in payload and "selection_plan" not in payload
    assert (run / "summary.json").is_file() and (run / "cases/1/queries/result.json").is_file()
    assert manager.record_started(run, SamplingPolicy()) == {}
    assert manager.record(run) is None
    assert [path.read_bytes() if path.exists() else None for path in cache_paths] == caches
    assert manager.list_results() == records and manager.sampling_state() == observations
    assert manager.rebuild() == records and manager.sampling_state() == observations
    commit_candidate(prepared)
    following = await app.run_job("home_evaluate")
    assert following.success and following.answer["record_history"] is True
    state = read_state(prepared)
    assert state["round"] == state["successful_round"] == 1 + int(baseline)
    # Legacy manifests without the new flag keep the previous opt-in behavior.
    tracked_run = Path(following.answer["result_dir"])
    tracked_manifest = manifest(following)
    tracked_manifest["options"].pop("record_history")
    (tracked_run / "manifest.json").write_text(json.dumps(tracked_manifest))
    assert len(manager.rebuild()) == 1 + int(baseline)


@pytest.mark.asyncio
async def test_untracked_full_evaluation_does_not_consult_history(app, prepared, monkeypatch):
    manager = history(app)

    def fail(*args, **kwargs):
        raise AssertionError("standalone full evaluation must not access history")

    for name in ("sampling_state", "record_started", "record", "invalidate"):
        monkeypatch.setattr(manager, name, fail)
    response = await app.run_job("home_evaluate", record_history=False)
    assert response.success, response.answer
    assert response.answer["case_count"] == response.answer["scored_query_count"] == 2


@pytest.mark.asyncio
async def test_sampling_and_history_are_independent_and_selection_evidence_is_local(app, prepared):
    assert (await app.run_job("home_evaluate")).success
    manager = history(app)
    records, observations = manager.list_results(), manager.sampling_state()
    commit_candidate(prepared)
    response = await app.run_job(
        "home_evaluate",
        sampling=True,
        record_history=False,
        sampling_policy={"max_cases": 1},
    )
    assert response.success and response.answer["case_count"] == 1
    payload = manifest(response)
    assert "sampling" not in payload
    assert payload["selection_plan"]["case_ids"] == payload["case_ids"]
    assert payload["selection_plan"]["seed"] and payload["selection_plan"]["policy"]["max_cases"] == 1
    assert manager.rebuild() == records and manager.sampling_state() == observations
    assert read_state(prepared)["round"] == read_state(prepared)["successful_round"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["infra", "exception"])
async def test_untracked_failures_preserve_existing_history(app, prepared, monkeypatch, failure):
    assert (await app.run_job("home_evaluate")).success
    manager = history(app)
    records, observations = manager.list_results(), manager.sampling_state()
    paths = [prepared / name for name in ("sampling/state.json",)]
    caches = [path.read_bytes() for path in paths]
    if failure == "infra":
        monkeypatch.setattr(FakeFactory, "bad_export", True)
    else:
        original = FakeFactory.create_cases

        async def fail(factory, *args, **kwargs):
            await original(factory, *args, **kwargs)
            raise ConnectionError("pool failed after first initialized sandbox")

        monkeypatch.setattr(FakeFactory, "create_cases", fail)
    commit_candidate(prepared)
    response = await app.run_job("home_evaluate", record_history=False)
    assert not response.success
    assert [path.read_bytes() for path in paths] == caches
    assert manager.rebuild() == records and manager.sampling_state() == observations
    assert read_state(prepared)["round"] == read_state(prepared)["successful_round"] == 1


@pytest.mark.asyncio
async def test_history_job_defaults_and_agent_injection_are_enforced(app, prepared):
    job = app.context.jobs["home_evaluate"]
    assert job.parameters["properties"]["record_history"]["default"] is True
    job.kwargs["record_history"] = False
    standalone = await app.run_job("home_evaluate")
    assert standalone.success and not history(app).list_results()
    commit_candidate(prepared)
    baseline = await app.run_job("home_evaluate", sampling=False, record_history=True)
    assert baseline.success and baseline.answer["case_count"] == 2
    tool = AsAgentWrapper._make_tool(job, injected_job_kwargs={"record_history": True})
    assert "record_history" not in tool.input_schema["properties"]
    commit_candidate(prepared)
    await tool.call()
    assert len(history(app).list_results()) == 2
    with pytest.raises(ValueError, match="injected"):
        await tool.call(record_history=False)


@pytest.mark.asyncio
async def test_legacy_cache_rebuild_and_explicit_null_query_budget(app, prepared):
    evaluator = app.context.components["home.evaluation"]["default"]
    evaluator.sampling_policy = SamplingPolicy(max_cases=2, max_queries=1)
    result = await app.run_job("home_evaluate", sampling=True, sampling_policy={"max_queries": None})
    assert result.success and result.answer["scored_query_count"] == 2
    plan = json.loads((prepared / manifest(result)["sampling"]["plan_path"]).read_text())
    assert plan["policy"]["max_queries"] is None
    assert evaluator.sampling_policy.max_queries == 1
    manager = history(app)
    observations = manager.sampling_state()
    legacy = read_state(prepared)
    legacy["schema_version"] = 1
    legacy.pop("successful_round")
    legacy["cases"]["1"]["ewma_loss"] = 999
    (prepared / "sampling/state.json").write_text(json.dumps(legacy))
    assert manager.sampling_state() == observations
    manager.rebuild()
    assert read_state(prepared)["schema_version"] == 2
    assert read_state(prepared)["successful_round"] == 1


@pytest.mark.asyncio
async def test_cancellation_after_durable_completion_keeps_history_consistent(app, prepared, monkeypatch):
    manager = history(app)
    original = manager.record
    entered, release = Event(), Event()

    def blocked(run):
        entered.set()
        assert release.wait(5)
        return original(run)

    monkeypatch.setattr(manager, "record", blocked)
    task = asyncio.create_task(app.run_job("home_evaluate"))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    records = manager.list_results()
    assert len(records) == 1
    run = prepared / "evaluations" / records[0].run_key
    completed = json.loads((run / "manifest.json").read_text())
    assert completed["status"] == "completed" and completed["success"] is True
    assert manager.rebuild() == records


@pytest.mark.asyncio
@pytest.mark.parametrize("after_record", [False, True])
@pytest.mark.parametrize("invalidation_fails", [False, True])
async def test_registration_failure_invalidates_only_caches(
    app,
    prepared,
    monkeypatch,
    after_record,
    invalidation_fails,
):
    manager = history(app)
    original = manager.record

    def fail(*args):
        if after_record:
            original(*args)
        raise OSError("sampling observations could not be saved")

    monkeypatch.setattr(manager, "record", fail)
    if invalidation_fails:

        def fail_invalidate():
            raise OSError("injected cache invalidation failure")

        monkeypatch.setattr(manager, "invalidate", fail_invalidate)
    response = await app.run_job("home_evaluate")
    assert response.success, response.answer
    runs = list((prepared / "evaluations").glob("*/*/*/manifest.json"))
    assert len(runs) == 1
    run = runs[0].parent
    completed = json.loads(runs[0].read_text())
    assert completed["status"] == "completed" and completed["success"] is True
    assert (run / "summary.json").is_file() and (run / "cases/1/queries/result.json").is_file()
    assert (prepared / "sampling/state.json").exists() is (invalidation_fails and after_record)
    assert read_state(prepared)["round"] == 1 and read_state(prepared)["successful_round"] == 1
    monkeypatch.setattr(manager, "record", original)
    assert len(manager.list_results()) == 1
    observations = manager.sampling_state()
    repeat = await app.run_job("home_evaluate")
    assert not repeat.success and repeat.answer["error_code"] == "UNCHANGED_CANDIDATE"
    assert manager.rebuild() == manager.list_results()
    assert manager.sampling_state() == observations
    commit_candidate(prepared)
    assert (await app.run_job("home_evaluate")).success
    state = read_state(prepared)
    manager.rebuild()
    assert read_state(prepared) == state
    assert state["successful_round"] == 2


@pytest.mark.asyncio
async def test_process_exit_after_completion_recovers_before_next_evaluation(app, prepared):
    from reme.application import Application

    config = app.config.model_dump(mode="json")
    await app.close()
    script = """
import asyncio
from importlib.metadata import EntryPoint
import json
import os
import sys
from unittest.mock import AsyncMock
sys.path[:] = json.loads(sys.argv[1])
from reme.application import Application
from reme_home.evaluation import component
from test_evaluation import FakeFactory
import reme.plugin
entries = {name: EntryPoint(name=name, value=value, group="reme.plugins") for name, value in
           [("home", "reme_home"), ("beam-judge", "judge_beam")]}
reme.plugin.find_entry_points = lambda group, name: [entries[name]]
component.DockerReMeSandboxFactory = FakeFactory
async def run():
    app = Application(**json.loads(sys.argv[2]))
    await app.start()
    app.context.components["agent_wrapper"]["judge"].reply = AsyncMock(
        return_value={"result": '{"score": 0.5}'})
    app.context.components["home.evaluation_history"]["default"].record = lambda run: os._exit(23)
    await app.run_job("home_evaluate")
asyncio.run(run())
"""
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", script, json.dumps(sys.path), json.dumps(config)],
        capture_output=True,
        text=True,
        timeout=30,
        env=os.environ.copy(),
    )
    assert result.returncode == 23, result.stderr
    assert not (prepared / "sampling/state.json").exists()
    restarted = Application(**config)
    await restarted.start()
    try:
        judge = restarted.context.components["agent_wrapper"]["judge"]
        judge.reply = AsyncMock(return_value={"result": '{"score": 0.5}'})
        manager = history(restarted)
        assert len(manager.list_results()) == 1
        assert read_state(prepared)["successful_round"] == 1
        commit_candidate(prepared)
        following = await restarted.run_job("home_evaluate")
        assert following.success, following.answer
        assert len(manager.list_results()) == 2
        state = read_state(prepared)
        assert state["successful_round"] == 2
        manager.rebuild()
        assert read_state(prepared) == state
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_repeated_cancellation_drains_preparation_before_application_close(app, prepared, monkeypatch):
    evaluator = app.context.components["home.evaluation"]["default"]
    entered, release, finished = Event(), Event(), Event()
    output = prepared / "thread-finished.txt"

    def prepare(*args):
        entered.set()
        assert release.wait(5)
        output.write_text("finished")
        finished.set()

    monkeypatch.setattr(evaluator, "_prepare", prepare)
    task = asyncio.create_task(app.run_job("home_evaluate"))
    close = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel("first cancellation")
        await asyncio.sleep(0)
        task.cancel("second cancellation")
        await asyncio.sleep(0)
        close = asyncio.create_task(app.close())
        await asyncio.sleep(0.05)
        assert evaluator._pending and not close.done() and not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError, match="first cancellation"):
            await task
        await close
        assert finished.is_set() and output.read_text() == "finished"
        assert not evaluator._pending
    finally:
        release.set()
        await asyncio.gather(task, *([close] if close is not None else []), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_blocking_preserves_kwargs_and_thread_errors(fail):
    from reme_home.evaluation.component import blocking

    error = OSError("thread operation failed")

    def operation(value, *, suffix):
        if fail:
            raise error
        return value + suffix

    if fail:
        with pytest.raises(OSError) as raised:
            await blocking(operation, "value", suffix="-suffix")
        assert raised.value is error
    else:
        assert await blocking(operation, "value", suffix="-suffix") == "value-suffix"


@pytest.mark.asyncio
async def test_blocking_drains_thread_through_repeated_cancellation(tmp_path):
    from reme_home.evaluation.component import blocking

    entered, release, finished = Event(), Event(), Event()
    output = tmp_path / "thread-finished.txt"

    def write(*, content):
        entered.set()
        assert release.wait(5)
        output.write_text(content)
        finished.set()

    task = asyncio.create_task(blocking(write, content="finished"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel("first cancellation")
        await asyncio.sleep(0)
        task.cancel("second cancellation")
        await asyncio.sleep(0)
        assert not task.done() and not finished.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError, match="first cancellation"):
            await task
        assert finished.is_set() and output.read_text() == "finished"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
