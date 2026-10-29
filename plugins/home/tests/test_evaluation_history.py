"""Successful-history ownership and evaluation parameter precedence through real Jobs."""

# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument,protected-access

import json
from pathlib import Path

import pytest
from reme_home.evaluation.history import HomeEvaluationHistory, EvaluationRecord, read_state
from reme_home.sampling import SamplingPolicy, create_sampling_plan
from test_evaluation import (  # noqa: F401  # pylint: disable=unused-import
    FakeFactory,
    home_application,
    preparation_inputs,
    prepared_home,
)

from reme.components.agent_wrapper.as_agent_wrapper import AsAgentWrapper


def history(app):
    return app.context.components["home.evaluation_history"]["default"]


def manifest(response):
    return json.loads((Path(response.answer["result_dir"]) / "manifest.json").read_text())


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
    payload = json.loads((prepared / "evaluations/history.json").read_text())
    assert payload["results"][0]["run_key"] == results[0].run_key
    assert "golden_answer" not in json.dumps(payload)
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
    sampled = await app.run_job("home_evaluate", sampling=True, sampling_policy={"max_cases": 1})
    assert baseline.success and sampled.success
    manager = history(app)
    expected, observations = manager.list_results(), manager.sampling_state()
    source = Path(sampled.answer["result_dir"]) / "summary.json"
    original = source.read_bytes()
    (prepared / "evaluations/history.json").unlink()
    (prepared / "sampling/state.json").unlink()
    assert manager.list_results() == expected
    assert manager.sampling_state() == observations
    assert manager.rebuild() == expected
    assert source.read_bytes() == original
    assert (prepared / "evaluations/history.json").is_file()
    assert (prepared / "sampling/state.json").is_file()


@pytest.mark.asyncio
async def test_call_policy_merges_component_defaults_without_mutating_them(app, prepared):
    evaluator = app.context.components["home.evaluation"]["default"]
    evaluator.sampling_policy = SamplingPolicy(max_cases=2, max_queries=2, temperature=0.7, ewma_alpha=0.6)
    response = await app.run_job("home_evaluate", sampling=True, sampling_policy={"max_cases": 1})
    assert response.success, response.answer
    options = manifest(response)["options"]
    assert options["sampling"] is True
    assert options["sampling_policy"] == {
        "max_cases": 1,
        "max_queries": 2,
        "temperature": 0.7,
        "ewma_alpha": 0.6,
        "uniform_exploration": 0.2,
    }
    assert response.answer["case_count"] == 1
    following = await app.run_job("home_evaluate", sampling=True)
    assert following.success and following.answer["case_count"] == 2
    assert evaluator.sampling_policy.max_cases == 2
    assert manifest(following)["options"]["sampling_policy"]["max_cases"] == 2
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
    await tool.call()
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
    cache_paths = [prepared / name for name in ("evaluations/history.json", "sampling/state.json")]
    caches = [path.read_bytes() if path.exists() else None for path in cache_paths]
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
    paths = [prepared / name for name in ("evaluations/history.json", "sampling/state.json")]
    caches = [path.read_bytes() for path in paths]
    if failure == "infra":
        monkeypatch.setattr(FakeFactory, "bad_export", True)
    else:
        original = FakeFactory.create_cases

        async def fail(factory, *args, **kwargs):
            await original(factory, *args, **kwargs)
            raise ConnectionError("pool failed after first initialized sandbox")

        monkeypatch.setattr(FakeFactory, "create_cases", fail)
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
    baseline = await app.run_job("home_evaluate", sampling=False, record_history=True)
    assert baseline.success and baseline.answer["case_count"] == 2
    tool = AsAgentWrapper._make_tool(job, injected_job_kwargs={"record_history": True})
    assert "record_history" not in tool.input_schema["properties"]
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
    assert manifest(result)["options"]["sampling_policy"]["max_queries"] is None
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
    import asyncio
    from threading import Event

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
async def test_registration_failure_invalidates_only_caches(app, prepared, monkeypatch):
    manager = history(app)
    original = manager._index

    def fail():
        raise OSError("history index could not be loaded")

    monkeypatch.setattr(manager, "_index", fail)
    response = await app.run_job("home_evaluate")
    assert not response.success
    runs = list((prepared / "evaluations").glob("*/*/*/manifest.json"))
    assert len(runs) == 1
    run = runs[0].parent
    failed = json.loads(runs[0].read_text())
    assert failed["status"] == "failed" and failed["success"] is False
    assert (run / "summary.json").is_file() and (run / "cases/1/queries/result.json").is_file()
    assert not (prepared / "sampling/state.json").exists()
    assert read_state(prepared)["round"] == 1 and read_state(prepared)["successful_round"] == 0
    monkeypatch.setattr(manager, "_index", original)
    assert manager.list_results() == [] and not manager.sampling_state()["cases"]
