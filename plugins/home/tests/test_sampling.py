"""Adaptive sampling parity and actual-start accounting through real Home Jobs."""

# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument,protected-access

import asyncio
import copy
import json
from pathlib import Path
from threading import Event

import pytest
from reme_home import sampling
from reme_home.evaluation import history as history_module
from reme_home.preparation.data.models import CaseSpec, GroundTruth, QuerySpec
from reme_home.workspace import _initialization_lock
from test_evaluation import (  # noqa: F401  # pylint: disable=unused-import
    FakeFactory,
    archive,
    command,
    commit_candidate,
    home_application,
    preparation_inputs,
    prepared_home,
    rebuild_review,
)

pytestmark = pytest.mark.usefixtures("rebuild_review")


def cases():
    return [
        CaseSpec(
            case_id=key,
            sessions=[],
            queries=[
                QuerySpec(query_id=f"query-{key}-{index}", question="?", groundtruth=GroundTruth())
                for index in range(4)
            ],
        )
        for key in "abc"
    ]


def test_plan_matches_dev_meta_golden_draw(monkeypatch):
    # Golden output computed with dev_meta b53048e4 sampling.py, seed=123456.
    monkeypatch.setattr(sampling.time, "time_ns", lambda: 123456)
    state = {
        "round": 2,
        "cases": {
            "a": {
                "observed_count": 2,
                "ewma_loss": 0.9,
                "last_selected_round": 2,
                "last_error_round": 1,
            },
        },
        "queries": {},
    }
    before = copy.deepcopy(state)
    policy = sampling.SamplingPolicy(max_cases=2, max_queries=3, temperature=0.8)
    plan = sampling.create_sampling_plan(cases(), state, policy)
    assert plan["case_ids"] == ["a", "b"]
    assert plan["query_ids"] == {"a": ["query-a-2", "query-a-0"], "b": ["query-b-3"]}
    assert plan["case_draw_weights"] == pytest.approx(
        {
            "a": 0.40033106216673214,
            "b": 0.29983446891663396,
            "c": 0.29983446891663396,
        },
    )
    assert plan["query_draw_weights"]["a"] == {f"query-a-{index}": 0.25 for index in range(4)}
    assert state == before
    assert sampling.create_sampling_plan(cases(), state, policy) == plan


def test_replay_eligibility_and_unrestricted_next_draw(monkeypatch):
    monkeypatch.setattr(sampling.time, "time_ns", lambda: 100)
    policy = sampling.SamplingPolicy(max_cases=3, max_queries=3)
    eligible = {"b": ["query-b-1", "query-b-3"], "c": ["query-c-2"]}
    plan = sampling.create_sampling_plan(cases(), {}, policy, eligible_query_ids=eligible)
    assert set(plan["case_ids"]) == {"b", "c"}
    assert all(set(values) <= set(eligible[key]) for key, values in plan["query_ids"].items())
    full = sampling.create_sampling_plan(cases(), {}, policy)
    assert set(full["case_draw_weights"]) == {"a", "b", "c"}
    assert full["eligibility"] is None


@pytest.mark.parametrize(
    "options",
    [
        {"max_cases": 0},
        {"max_queries": 0},
        {"max_cases": True},
        {"max_queries": True},
        {"temperature": 0},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"ewma_alpha": 0},
        {"ewma_alpha": 1.1},
        {"uniform_exploration": -0.1},
        {"uniform_exploration": 1.1},
        {"unexpected": 1},
    ],
)
def test_policy_rejects_invalid_limits(options):
    with pytest.raises(ValueError):
        sampling.SamplingPolicy(**options)


def test_uniform_mass_and_unobserved_priority():
    records = {
        "easy": {"ewma_loss": 0.0, "observed_count": 100, "last_selected_round": 10},
        "hard": {"ewma_loss": 1.0, "observed_count": 1, "last_selected_round": 1, "last_error_round": 1},
        "unseen": {},
    }
    weights = sampling._mixed_weights(records, 11, sampling.SamplingPolicy(uniform_exploration=0.3))
    assert sum(weights.values()) == pytest.approx(1)
    assert all(weight >= 0.1 for weight in weights.values())
    assert weights["hard"] > weights["easy"] and weights["unseen"] > weights["easy"]
    uniform = sampling._mixed_weights(records, 11, sampling.SamplingPolicy(uniform_exploration=1))
    assert set(uniform.values()) == {1 / 3}


def state(home):
    return history_module.read_state(home)


@pytest.mark.asyncio
async def test_baseline_then_adaptive_job_updates_only_selected_cases(app, prepared):
    baseline = await app.run_job("home_evaluate")
    assert baseline.success, baseline.answer
    assert state(prepared)["round"] == 1
    assert all(record["ewma_loss"] == 0.5 for record in state(prepared)["cases"].values())
    evaluation = app.context.components["home.evaluation"]["default"]
    evaluation.sampling_policy = sampling.SamplingPolicy(max_cases=1, max_queries=1, ewma_alpha=0.5)
    app.context.components["agent_wrapper"]["judge"].reply.return_value = {
        "result": '{"score": 1}',
        "status": "completed",
        "error": None,
    }
    before = copy.deepcopy(state(prepared))
    commit_candidate(prepared)
    response = await app.run_job("home_evaluate", sampling=True)
    assert response.success, response.answer
    run = Path(response.answer["result_dir"])
    manifest = json.loads((run / "manifest.json").read_text())
    plan = json.loads((prepared / manifest["sampling"]["plan_path"]).read_text())
    selected = manifest["case_ids"][0]
    unselected = ({"1", "2"} - {selected}).pop()
    assert manifest["case_ids"] == plan["case_ids"] and manifest["query_ids"] == plan["query_ids"]
    assert plan["mode"] == "sampled" and plan["round"] == 2
    assert response.answer["case_count"] == response.answer["scored_query_count"] == 1
    after = state(prepared)
    assert after["cases"][selected]["ewma_loss"] == 0.25
    assert after["cases"][selected]["observed_count"] == 2
    assert after["cases"][selected]["successful_observations_after_last_error"] == 1
    assert after["cases"][unselected] == before["cases"][unselected]
    app.context.components["home.evaluation_history"]["default"].record(run)
    assert state(prepared) == after  # A summary is consumed once, including after reopening.
    assert "home_validate" not in app.context.jobs
    assert app.context.jobs["home_evaluate"].parameters["properties"]["sampling"]["type"] == "boolean"
    assert "sampling_policy" not in app.context.jobs["home_evaluate"].parameters["properties"]
    assert "case_ids" not in app.context.jobs["home_evaluate"].parameters["properties"]


@pytest.mark.asyncio
async def test_debug_never_reads_or_changes_sampling_state(app, prepared):
    before = prepared / "sampling/state.json"
    for enabled in (False, True):
        response = await app.run_job("home_evaluate", sampling=enabled, debug=True)
        assert response.success, response.answer
    assert not before.exists()
    assert not list((prepared / "sampling/plans").rglob("*.json"))
    # Corrupt sampling state cannot prevent an unrelated debug smoke test.
    before.write_text("corrupt cached state")
    response = await app.run_job("home_evaluate", sampling=True, debug=True)
    assert response.success and before.read_text() == "corrupt cached state"
    manifest = json.loads((Path(response.answer["result_dir"]) / "manifest.json").read_text())
    assert "sandbox_started_at" not in manifest and "sampling" not in manifest


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["dirty", "factory_constructor", "startup", "selection", "replay"])
async def test_prestart_failure_never_counts(app, prepared, monkeypatch, failure):
    await app.run_job("home_evaluate")
    commit_candidate(prepared)
    previous = (prepared / "sampling/state.json").read_bytes()
    plans = sorted((prepared / "sampling/plans").rglob("*.json"))
    options = {}
    if failure == "dirty":
        (prepared / "code/repo/plugins/beam/src/reme_beam/__init__.py").write_text("dirty")
    elif failure == "selection":
        options["case_ids"] = ["1"]
    elif failure == "replay":
        options["reuse_construction_from"] = "evaluations/not-a-run"
    elif failure == "factory_constructor":

        def fail(*args, **kwargs):
            raise RuntimeError("factory never created")

        monkeypatch.setattr("reme_home.evaluation.component.DockerReMeSandboxFactory", fail)
    else:

        async def fail(*args, **kwargs):
            raise RuntimeError("no sandbox initialized")

        monkeypatch.setattr(FakeFactory, "create_cases", fail)
    response = await app.run_job("home_evaluate", sampling=True, **options)
    assert not response.success
    assert (prepared / "sampling/state.json").read_bytes() == previous
    assert sorted((prepared / "sampling/plans").rglob("*.json")) == plans


@pytest.mark.asyncio
async def test_started_infrastructure_failure_is_not_a_quality_zero(app, prepared, monkeypatch):
    commit_candidate(prepared)

    async def fail(*args, **kwargs):
        raise ConnectionError("construction provider unavailable")

    monkeypatch.setattr("test_evaluation.FakeWorker.run_build", fail)
    response = await app.run_job("home_evaluate", case_ids=["1"])
    assert not response.success
    observed = state(prepared)
    assert observed["round"] == 1
    assert not observed["cases"] and not observed["queries"]
    assert app.context.components["home.evaluation_history"]["default"].list_results() == []


@pytest.mark.asyncio
async def test_started_candidate_failure_does_not_fabricate_a_score(app, prepared, monkeypatch):
    commit_candidate(prepared)
    factory = FakeFactory.create_cases

    async def create(self, *args, **kwargs):
        workers = await factory(self, *args, **kwargs)

        async def empty(_queries):
            return {"success": True, "queries": [{"query_id": "fact:1", "answer": "", "error": None}]}

        for worker in workers:
            worker.run_queries = empty
        return workers

    monkeypatch.setattr(FakeFactory, "create_cases", create)
    response = await app.run_job("home_evaluate", case_ids=["1"])
    assert response.success, response.answer
    observed = state(prepared)
    assert observed["round"] == 1
    record = observed["cases"]["1"]
    assert record["candidate_failure_count"] == 1
    assert record["observed_count"] == 0 and record["ewma_loss"] is None
    query = observed["queries"][sampling._query_key("1", "fact:1")]
    assert query["candidate_failure_count"] == 1 and query["observed_count"] == 0


@pytest.mark.asyncio
async def test_started_cancellation_retains_round_but_no_partial_quality(app, prepared, monkeypatch):
    commit_candidate(prepared)
    monkeypatch.setattr(FakeFactory, "block", asyncio.Event())
    task = asyncio.create_task(app.run_job("home_evaluate", sampling=True, concurrency=1))
    for _ in range(200):
        if FakeFactory.instances and FakeFactory.instances[0].entered.is_set():
            break
        await asyncio.sleep(0.01)
    assert FakeFactory.instances[0].entered.is_set()
    assert state(prepared)["round"] == 1
    concurrent = await app.run_job("home_evaluate", sampling=True)
    assert not concurrent.success  # Formal validations follow dev_meta's serialization boundary.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    observed = state(prepared)
    assert observed["round"] == 1 and not observed["cases"] and not observed["queries"]
    assert not observed["processed_validations"]
    with _initialization_lock(prepared / "metadata/validation.lock"):
        pass


@pytest.mark.asyncio
async def test_replay_sampling_is_limited_to_donor_cases(app, prepared, monkeypatch):
    commit_candidate(prepared)
    from reme_home.evaluation import inputs

    baseline = await app.run_job("home_evaluate", case_ids=["1"])
    assert baseline.success, baseline.answer
    source = Path(baseline.answer["result_dir"])
    commit_candidate(prepared)
    reads = []
    read_json = inputs.read_json

    def track_source_metadata(path):
        if path in {source / "manifest.json", source / "summary.json"}:
            reads.append(path.name)
        return read_json(path)

    monkeypatch.setattr(inputs, "read_json", track_source_metadata)
    app.context.components["home.evaluation"]["default"].sampling_policy = sampling.SamplingPolicy(max_cases=1)
    response = await app.run_job("home_evaluate", sampling=True, reuse_construction_from=str(source))
    assert response.success, response.answer
    assert reads == ["manifest.json", "summary.json"]
    manifest = json.loads((Path(response.answer["result_dir"]) / "manifest.json").read_text())
    assert manifest["case_ids"] == ["1"]
    plan = json.loads((prepared / manifest["sampling"]["plan_path"]).read_text())
    assert plan["eligibility"] == {"1": ["fact:1"]}
    # The next full rebuild must see the entire training set again.
    commit_candidate(prepared)
    response = await app.run_job("home_evaluate", sampling=True)
    assert response.success, response.answer
    manifest = json.loads((Path(response.answer["result_dir"]) / "manifest.json").read_text())
    plan = json.loads((prepared / manifest["sampling"]["plan_path"]).read_text())
    assert set(plan["case_draw_weights"]) == {"1", "2"} and plan["eligibility"] is None


@pytest.mark.asyncio
async def test_sampling_cache_rebuild_and_new_round_after_reopen(app, prepared):
    baseline = await app.run_job("home_evaluate")
    assert baseline.success
    expected = state(prepared)
    (prepared / "sampling/state.json").unlink()
    assert state(prepared) == expected
    assert not (prepared / "sampling/state.json").exists()  # Rebuilding reads never claim a new round.
    with _initialization_lock(prepared / "metadata/workspace.lock"):
        assert history_module.rebuild_sampling_state(prepared) == expected
    commit_candidate(prepared)
    result = await app.run_job("home_evaluate", sampling=True)
    assert result.success, result.answer
    assert state(prepared)["round"] == 2


@pytest.mark.asyncio
async def test_fail_fast_started_round_without_summary_is_not_quality_feedback(app, prepared):
    commit_candidate(prepared)
    app.context.components["agent_wrapper"]["judge"].reply.return_value = {
        "result": "invalid judge output",
        "status": "completed",
        "error": None,
    }
    response = await app.run_job("home_evaluate", sampling=True, fail_fast=True)
    assert not response.success
    observed = state(prepared)
    assert observed["round"] == 1
    assert not observed["processed_validations"] and not observed["cases"]
    assert not observed["queries"]
    (prepared / "sampling/state.json").unlink()
    assert state(prepared) == observed


@pytest.mark.asyncio
async def test_partial_pool_start_failure_counts_started_round(app, prepared, monkeypatch):
    commit_candidate(prepared)
    original = FakeFactory.create_cases

    async def fail_after_first_start(factory, *args, **kwargs):
        await original(factory, *args, **kwargs)
        raise ConnectionError("a later worker failed after first initialized")

    monkeypatch.setattr(FakeFactory, "create_cases", fail_after_first_start)
    result = await app.run_job("home_evaluate", sampling=True)
    assert not result.success
    observed = state(prepared)
    assert observed["round"] == 1 and not observed["processed_validations"]
    assert not observed["cases"] and not observed["queries"]


@pytest.mark.asyncio
async def test_started_callback_is_idempotent(app, prepared):
    commit_candidate(prepared)
    response = await app.run_job("home_evaluate", sampling=True)
    assert response.success
    run = Path(response.answer["result_dir"])
    manifest = json.loads((run / "manifest.json").read_text())
    original = state(prepared)
    component = app.context.components["home.evaluation"]["default"]
    component._sampling_started(run, manifest, None)
    assert state(prepared) == original


@pytest.mark.parametrize("invalid_count", [1, 2])
@pytest.mark.parametrize("damage", ["missing_result", "invalid_result", "checksum", "unsafe_archive"])
def test_replay_eligibility_uses_dev_meta_five_percent_threshold(tmp_path, invalid_count, damage):
    from reme_home.evaluation.inputs import EvaluationSpec, replay_sampling_inputs
    from reme_home.evaluation.artifacts import file_sha256, write_json

    home = tmp_path / "home"
    source = home / "evaluations/init/commit/1"
    fingerprints = {"dataset": "dataset", "construction_config": "config"}
    selected = [
        CaseSpec(
            case_id=f"case-{index}",
            sessions=[],
            queries=[
                QuerySpec(query_id="q", question="?", groundtruth=GroundTruth()),
            ],
        )
        for index in range(20)
    ]
    write_json(
        source / "manifest.json",
        {
            "dataset_kind": "training-set",
            "mode": "scored",
            "status": "completed",
            "commit_sha": "commit",
            "fingerprints": fingerprints,
            "case_ids": [case.case_id for case in selected],
        },
    )
    write_json(source / "summary.json", {})
    for index, case in enumerate(selected):
        folder = source / "cases" / case.case_id
        construction = folder / "memory_construction"
        invalid = index < invalid_count
        entry = "../unsafe" if invalid and damage == "unsafe_archive" else "daily/note.md"
        snapshot = archive(construction / "reme_workspace.tar.gz", {entry: "fixture"})
        result_path = construction / "result.json"
        if invalid and damage in {"missing_result", "invalid_result"}:
            if damage == "invalid_result":
                result_path.write_text("{", encoding="utf-8")
            continue
        write_json(result_path, {"success": True, "workspace_sha256": file_sha256(snapshot)})
        if invalid and damage == "checksum":
            snapshot.write_bytes(b"corrupt")
        write_json(folder / "queries/result.json", {"queries": [{"answer": "answer", "error": None}]})
    spec = EvaluationSpec(reuse_construction_from=str(source))
    if invalid_count == 1:
        _, archives, eligible = replay_sampling_inputs(home, spec, selected, fingerprints)
        assert archives.keys() == eligible.keys() == {case.case_id for case in selected[1:]}
    else:
        with pytest.raises(ValueError, match="5%"):
            replay_sampling_inputs(home, spec, selected, fingerprints)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("artifact", "bad_result", "error"),
    [
        ("memory_construction/result.json", "[]", "Replay source construction failure ratio exceeds 5%"),
        ("memory_construction/reme_workspace.tar.gz", "corrupt", "Replay source construction failure ratio exceeds 5%"),
        ("queries/result.json", "[]", "Replay source has no completed successful answers for 2"),
        ("queries/result.json", "{", "Replay source cannot read answers for case 2"),
    ],
)
async def test_explicit_replay_sampling_reports_invalid_source(app, prepared, artifact, bad_result, error):
    baseline = await app.run_job("home_evaluate")
    assert baseline.success, baseline.answer
    source = Path(baseline.answer["result_dir"])
    (source / "cases" / "2" / artifact).write_text(bad_result, encoding="utf-8")
    commit_candidate(prepared)
    launches = len(FakeFactory.instances)

    response = await app.run_job(
        "home_evaluate",
        sampling=True,
        reuse_construction_from=str(source),
        target_queries={"1": ["fact:1"]},
        sampling_policy={"max_cases": 1, "max_queries": 1},
    )

    assert not response.success and error in str(response.answer)
    assert len(FakeFactory.instances) == launches


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [None, "archive", "branch_during_preflight", "branch_during_copy"])
async def test_explicit_replay_archive_preflight_allows_status_job(app, prepared, monkeypatch, change):
    from reme_home.evaluation import component, inputs

    baseline = await app.run_job("home_evaluate")
    assert baseline.success, baseline.answer
    source = Path(baseline.answer["result_dir"])
    selected_archive = source / "cases/1/memory_construction/reme_workspace.tar.gz"
    repo = prepared / "code/repo"
    commit_candidate(prepared)
    command(repo, "branch", "different-branch")
    launches = len(FakeFactory.instances)
    entered = asyncio.Event()
    resume = Event()
    loop = asyncio.get_running_loop()
    inspect_archive = inputs.inspect_archive

    def inspect(path, max_bytes):
        if path.is_relative_to(source / "cases/2"):
            loop.call_soon_threadsafe(entered.set)
            assert resume.wait(timeout=10)
        return inspect_archive(path, max_bytes)

    monkeypatch.setattr(inputs, "inspect_archive", inspect)
    if change in {"archive", "branch_during_copy"}:
        copyfile = component.shutil.copyfile

        def copy(src, dst, **kwargs):
            if src == selected_archive and change == "archive":
                selected_archive.write_bytes(b"changed after preflight")
            result = copyfile(src, dst, **kwargs)
            if src == selected_archive and change == "branch_during_copy":
                command(repo, "switch", "different-branch")
            return result

        monkeypatch.setattr(component.shutil, "copyfile", copy)

    task = asyncio.create_task(
        app.run_job(
            "home_evaluate",
            sampling=True,
            reuse_construction_from=str(source),
            target_queries={"1": ["fact:1"]},
            sampling_policy={"max_cases": 1, "max_queries": 1},
        ),
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        status = await app.run_job("home_code_status")
        if change == "branch_during_preflight":
            command(repo, "switch", "different-branch")
    finally:
        resume.set()
        response = await task
    assert status.success, status.answer
    if change is None:
        assert response.success, response.answer
        manifest = json.loads((Path(response.answer["result_dir"]) / "manifest.json").read_text())
        assert manifest["query_ids"] == {"1": ["fact:1"]}
        assert len(FakeFactory.instances) == launches + 1
    else:
        error = (
            "Construction source changed after preflight"
            if change == "archive"
            else "Candidate changed during equivalence review"
        )
        assert not response.success and error in str(response.answer)
        assert len(FakeFactory.instances) == launches


@pytest.mark.asyncio
async def test_query_budget_and_draw_order_reach_actual_execution(app, inputs, monkeypatch):
    from unittest.mock import AsyncMock

    source = Path(inputs[0]["benchmark"]["source"])
    for path in source.glob("chats/*/*/probing_questions/probing_questions.json"):
        payload = json.loads(path.read_text())
        payload["fact"] = [
            dict(payload["fact"][0], question=f"question {index}", rubric=["says hello"]) for index in range(4)
        ]
        path.write_text(json.dumps(payload))
    prepared = await app.run_job("home_prepare", spec=inputs[0])
    assert prepared.success
    commit_candidate(Path(app.config.workspace_dir) / "home")
    monkeypatch.setattr("reme_home.evaluation.component.DockerReMeSandboxFactory", FakeFactory)
    monkeypatch.setattr(
        app.context.components["agent_wrapper"]["judge"],
        "reply",
        AsyncMock(return_value={"result": '{"score": 0.5}', "status": "completed", "error": None}),
    )
    monkeypatch.setattr(sampling.time, "time_ns", lambda: 123456)
    component = app.context.components["home.evaluation"]["default"]
    component.sampling_policy = sampling.SamplingPolicy(max_cases=2, max_queries=3)
    response = await app.run_job("home_evaluate", sampling=True, concurrency=1)
    assert response.success, response.answer
    run = Path(response.answer["result_dir"])
    manifest = json.loads((run / "manifest.json").read_text())
    home = Path(app.config.workspace_dir) / "home"
    plan = json.loads((home / manifest["sampling"]["plan_path"]).read_text())
    assert manifest["query_ids"] == plan["query_ids"]
    assert len(manifest["case_ids"]) == 2 and sum(map(len, manifest["query_ids"].values())) == 3
    worker = FakeFactory.instances[-1].workers[0]
    assert [query.query_id for query in worker.queries] == manifest["query_ids"][worker.case_id]
    for case_id, query_ids in manifest["query_ids"].items():
        result = json.loads((run / "cases" / case_id / "queries/result.json").read_text())
        assert [row["query_id"] for row in result["queries"]] == query_ids
    assert response.answer["scored_query_count"] == 3
