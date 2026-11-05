"""Tree-based repetition checks through real Jobs with fake Docker/model boundaries."""

# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from reme_home.evaluation.repetition import retryable_result
from test_evaluation import (  # noqa: F401  # pylint: disable=unused-import
    FakeFactory,
    command,
    home_application,
    preparation_inputs,
    prepared_home,
)


def artifacts(home):
    return {path.relative_to(home): path.read_bytes() for path in home.rglob("*") if path.is_file()}


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [{}, {"sampling": True}, {"record_history": False}, {"case_ids": ["1"]}])
async def test_completed_tree_rejected_before_pool_allocation_or_history_changes(app, prepared, options):
    baseline = await app.run_job("home_evaluate")
    assert baseline.success
    before = artifacts(prepared)
    repeat = await app.run_job("home_evaluate", **options)
    assert not repeat.success and "already been evaluated" in str(repeat.answer)
    assert repeat.answer == {
        "status": "rejected",
        "error_code": "UNCHANGED_CANDIDATE",
        "error": repeat.answer["error"],
        "result_dir": baseline.answer["result_dir"],
    }
    assert "already been evaluated" in repeat.answer["error"]
    assert repeat.metadata["result_dir"] == baseline.answer["result_dir"]
    assert "reme.job_error" not in repeat.metadata
    assert len(FakeFactory.instances) == 1 and artifacts(prepared) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["empty_commit", "branch", "code"])
async def test_tree_identity_across_commits_and_branches(app, prepared, change):
    baseline = await app.run_job("home_evaluate", record_history=False)
    assert baseline.success
    repo = prepared / "code/repo"
    assert (await app.run_job("home_create_candidate_branch", branch_name="next")).success
    if change == "code":
        path = repo / "plugins/beam/src/reme_beam/__init__.py"
        path.write_text("value = 2\n")
        command(repo, "add", ".")
    if change != "branch":
        command(
            repo,
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=test@localhost",
            "commit",
            "--allow-empty",
            "-m",
            change,
        )
    response = await app.run_job("home_evaluate")
    assert response.success is (change == "code")
    assert len(FakeFactory.instances) == (2 if change == "code" else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_fast", [False, True])
async def test_candidate_failures_block_retry_including_partial_results(app, prepared, monkeypatch, fail_fast):
    monkeypatch.setattr(FakeFactory, "fail_build", "2")
    failed = await app.run_job("home_evaluate", fail_fast=fail_fast, concurrency=1)
    assert failed.success is (not fail_fast)
    before = artifacts(prepared)
    retry = await app.run_job("home_evaluate")
    assert not retry.success and "already been evaluated" in str(retry.answer)
    assert len(FakeFactory.instances) == 1 and artifacts(prepared) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_fast", [False, True])
async def test_infrastructure_retry_gets_new_id_and_retains_prior_artifacts(app, prepared, monkeypatch, fail_fast):
    monkeypatch.setattr(FakeFactory, "bad_export", True)
    failed = await app.run_job("home_evaluate", fail_fast=fail_fast)
    assert not failed.success
    manifests = list((prepared / "evaluations").glob("*/*/*/manifest.json"))
    run = manifests[0].parent
    before = artifacts(run)
    monkeypatch.setattr(FakeFactory, "bad_export", False)
    recovered = await app.run_job("home_evaluate")
    assert recovered.success, recovered.answer
    assert Path(recovered.answer["result_dir"]).name == "2"
    assert artifacts(run) == before and len(FakeFactory.instances) == 2
    repeat = await app.run_job("home_evaluate")
    assert not repeat.success and len(FakeFactory.instances) == 2


@pytest.mark.asyncio
async def test_interrupted_run_retries_but_empty_commit_cannot_relabel_failure(app, prepared, monkeypatch):
    original = FakeFactory.create_cases

    async def fail(*args, **kwargs):
        raise ConnectionError("Docker unavailable before startup")

    monkeypatch.setattr(FakeFactory, "create_cases", fail)
    assert not (await app.run_job("home_evaluate")).success
    repo = prepared / "code/repo"
    assert (await app.run_job("home_create_candidate_branch", branch_name="retry")).success
    command(
        repo,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=test@localhost",
        "commit",
        "--allow-empty",
        "-m",
        "empty",
    )
    rejected = await app.run_job("home_evaluate")
    assert not rejected.success and "already been evaluated" in str(rejected.answer)
    command(repo, "reset", "--hard", "HEAD~1")
    monkeypatch.setattr(FakeFactory, "create_cases", original)
    recovered = await app.run_job("home_evaluate")
    assert recovered.success, recovered.answer
    # Switching branches does not overwrite the first attempt.
    assert len(list((prepared / "evaluations").glob("*/*/*/manifest.json"))) == 2


@pytest.mark.asyncio
async def test_debug_can_repeat_and_does_not_block_scored_baseline(app, prepared):
    assert (await app.run_job("home_evaluate", debug=True)).success
    assert (await app.run_job("home_evaluate", debug=True)).success
    assert (await app.run_job("home_evaluate")).success
    assert (await app.run_job("home_evaluate", debug=True)).success


@pytest.mark.asyncio
async def test_cancelled_evaluation_retries_after_pool_cleanup(app, prepared, monkeypatch):
    monkeypatch.setattr(FakeFactory, "block", asyncio.Event())
    task = asyncio.create_task(app.run_job("home_evaluate", concurrency=1))
    try:
        async with asyncio.timeout(5):
            while not FakeFactory.instances:
                await asyncio.sleep(0.01)
            await FakeFactory.instances[0].entered.wait()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert all(worker.closed for worker in FakeFactory.instances[0].workers)
    run = next((prepared / "evaluations").glob("*/*/*/manifest.json")).parent
    before = artifacts(run)
    monkeypatch.setattr(FakeFactory, "block", None)
    recovered = await app.run_job("home_evaluate")
    assert recovered.success and Path(recovered.answer["result_dir"]).name == "2"
    assert artifacts(run) == before


@pytest.mark.asyncio
async def test_replay_and_corrupt_sampling_cache_cannot_bypass_guard(app, prepared):
    baseline = await app.run_job("home_evaluate")
    assert baseline.success
    (prepared / "sampling/state.json").write_text("invalid cache")
    for options in ({"sampling": True}, {"reuse_construction_from": baseline.answer["result_dir"]}):
        repeat = await app.run_job("home_evaluate", **options)
        assert not repeat.success and "already been evaluated" in str(repeat.answer)
    assert len(FakeFactory.instances) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("record_history", [True, False])
@pytest.mark.parametrize("retry_sampling", [True, False])
async def test_sampled_recovery_preserves_cases_queries_and_order(
    app,
    inputs,
    monkeypatch,
    record_history,
    retry_sampling,
):
    source = Path(inputs[0]["benchmark"]["source"])
    for path in source.glob("chats/*/*/probing_questions/probing_questions.json"):
        raw = json.loads(path.read_text())
        raw["fact"] = [dict(raw["fact"][0], question=f"question {i}", rubric=["says hello"]) for i in range(4)]
        path.write_text(json.dumps(raw))
    assert (await app.run_job("home_prepare", spec=inputs[0])).success
    monkeypatch.setattr("reme_home.evaluation.component.DockerReMeSandboxFactory", FakeFactory)
    FakeFactory.instances = []
    monkeypatch.setattr(
        app.context.components["agent_wrapper"]["judge"],
        "reply",
        AsyncMock(return_value={"result": '{"score": 0.5}'}),
    )
    monkeypatch.setattr("reme_home.sampling.time.time_ns", lambda: 1)
    monkeypatch.setattr(FakeFactory, "bad_export", True)
    first = await app.run_job(
        "home_evaluate",
        sampling=True,
        sampling_policy={"max_cases": 1, "max_queries": 2},
        record_history=record_history,
    )
    assert not first.success
    run = Path(first.answer["result_dir"])
    original = artifacts(run)
    old = json.loads((run / "manifest.json").read_text())
    assert len(old["case_ids"]) == 1 and len(next(iter(old["query_ids"].values()))) == 2
    home = Path(app.config.workspace_dir) / "home"
    state_path = home / "sampling/state.json"
    state_before = state_path.read_bytes() if state_path.exists() else None

    def no_draw(*args, **kwargs):
        raise AssertionError("Recovery must not redraw or consult sampling observations")

    monkeypatch.setattr("reme_home.evaluation.component.create_sampling_plan", no_draw)
    monkeypatch.setattr(app.context.components["home.evaluation_history"]["default"], "sampling_state", no_draw)
    monkeypatch.setattr("reme_home.sampling.time.time_ns", lambda: 2)
    monkeypatch.setattr(FakeFactory, "bad_export", False)
    options = {} if retry_sampling else {"case_ids": list({"1", "2"} - set(old["case_ids"]))}
    retry = await app.run_job(
        "home_evaluate",
        sampling=retry_sampling,
        sampling_policy={"max_cases": 2, "max_queries": 1},
        record_history=record_history,
        **options,
    )
    assert retry.success, retry.answer
    new_run = Path(retry.answer["result_dir"])
    new = json.loads((new_run / "manifest.json").read_text())
    assert new_run.name == "2" and artifacts(run) == original
    assert new["case_ids"] == old["case_ids"] and new["query_ids"] == old["query_ids"]
    worker = FakeFactory.instances[-1].workers[0]
    assert [query.query_id for query in worker.queries] == old["query_ids"][worker.case_id]
    plan = json.loads((home / new["sampling"]["plan_path"]).read_text()) if record_history else new["selection_plan"]
    assert plan["case_ids"] == old["case_ids"] and plan["query_ids"] == old["query_ids"]
    assert plan["mode"] == "explicit" and plan["recovery_from"] == run.relative_to(home).as_posix()
    assert "seed" not in plan
    if not record_history:
        assert (state_path.read_bytes() if state_path.exists() else None) == state_before


@pytest.mark.asyncio
async def test_recovery_before_sandbox_start_keeps_original_explicit_selection(app, prepared, monkeypatch):
    original = FakeFactory.create_cases

    async def fail(*args, **kwargs):
        raise ConnectionError("Docker unavailable")

    monkeypatch.setattr(FakeFactory, "create_cases", fail)
    first = await app.run_job("home_evaluate", case_ids=["2", "1"])
    assert not first.success
    old = json.loads(next((prepared / "evaluations").glob("*/*/*/manifest.json")).read_text())
    assert "sampling" not in old and "selection_plan" not in old
    monkeypatch.setattr(FakeFactory, "create_cases", original)
    retry = await app.run_job("home_evaluate", case_ids=["1"])
    assert retry.success
    new = json.loads((Path(retry.answer["result_dir"]) / "manifest.json").read_text())
    assert new["case_ids"] == ["2", "1"] and new["query_ids"] == old["query_ids"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [{"case_ids": None}, {"query_ids": None}, {"query_ids": {}}, {"query_ids": {"1": ["missing"]}}],
)
async def test_invalid_saved_recovery_selection_fails_before_allocation(app, prepared, monkeypatch, mutation):
    monkeypatch.setattr(FakeFactory, "bad_export", True)
    assert not (await app.run_job("home_evaluate", case_ids=["1"])).success
    path = next((prepared / "evaluations").glob("*/*/*/manifest.json"))
    manifest = json.loads(path.read_text())
    path.write_text(json.dumps({**manifest, **mutation}))
    before = artifacts(path.parent)
    retry = await app.run_job("home_evaluate")
    assert not retry.success
    assert len(FakeFactory.instances) == 1 and artifacts(path.parent) == before
    assert len(list((prepared / "evaluations").glob("*/*/*/manifest.json"))) == 1


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"cases": {"a": {"status": "completed"}}}, False),
        ({"cases": [{"status": "candidate_failure"}]}, False),
        ({"cases": [{"status": "infra_error"}]}, True),
        ({"cases": []}, True),
        ({"cases": {"a": {}}, "case_status_counts": {"completed": 0, "candidate_failure": 0}}, True),
    ],
)
def test_legacy_and_incomplete_summaries(tmp_path, payload, expected):
    (tmp_path / "summary.json").write_text(json.dumps(payload))
    assert retryable_result(tmp_path) is expected
