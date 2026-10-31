"""Read-only result Jobs against real Home applications and isolated Git history."""

# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument,protected-access

import asyncio
from pathlib import Path
from threading import Event

import pytest
from reme_home.evaluation import inspection_step as inspection
from reme_home.evaluation.artifacts import write_json

from test_evaluation import (  # noqa: F401  # pylint: disable=unused-import
    FakeFactory,
    home_application,
    preparation_inputs,
    prepared_home,
)
from test_preparation import command

from reme.components.agent_wrapper.as_agent_wrapper import AsAgentWrapper


async def call(app, name, **kwargs):
    response = await app.run_job(name, **kwargs)
    assert response.success, response.answer
    return response.answer


def snapshot(home):
    return {path.relative_to(home): path.read_bytes() for path in home.rglob("*") if path.is_file()}


def write_run(home, run_id="1", *, branch="candidate", commit="a" * 40, **updates):
    run = home / "evaluations" / branch / commit[:7] / run_id
    manifest = {
        "branch_name": branch,
        "commit_sha": commit,
        "validation_id": run_id,
        "case_ids": ["case~a", "missing-case"],
        "query_ids": {"case~a": ["q1", "q2"], "missing-case": ["q3"]},
        "mode": "scored",
        "dataset_kind": "training-set",
        "status": "failed",
        "started_at": f"2026-01-01T00:00:{int(run_id):02d}+00:00",
        "options": {"record_history": False},
        **updates,
    }
    write_json(run / "manifest.json", manifest)
    return run


@pytest.mark.asyncio
async def test_real_evaluation_catalog_and_case_query_navigation_are_read_only(app, prepared):
    scored = await call(app, "home_evaluate", record_history=False)
    debug = await call(app, "home_evaluate", debug=True)
    before = snapshot(prepared)
    page = await call(app, "home_list_validation_results", limit=1)
    assert page["row_count"] == 2 and page["next_offset"] == 1
    assert page["rows"][0]["mode"] == "debug" and page["rows"][0]["record_history"] is False
    last = await call(app, "home_list_validation_results", offset=1, limit=1)
    assert last["next_offset"] is None and last["rows"][0]["validation_path"] == scored["result_dir"]
    assert last["rows"][0]["totals"]["scored_query_count"] == 2
    run = Path(scored["result_dir"])
    result = await call(app, "home_inspect_validation", validation_path=run.relative_to(prepared).as_posix(), limit=1)
    assert result["row_count"] == 2 and result["next_offset"] == 1
    case = result["rows"][0]
    assert case["mean_score"] == 0.5 and case["pointer"].startswith("/cases/")
    result = await call(app, "home_inspect_validation", validation_path=str(run), case_id=case["case_id"])
    query = result["rows"][0]
    assert query["score"] == 0.5 and query["pointer"] == "/queries/0"
    assert "fields" not in query and result["construction"]["failed_jobs"]["rows"] == []
    result = await call(
        app,
        "home_inspect_validation",
        validation_path=str(run),
        case_id=case["case_id"],
        query_id=query["query_id"],
    )
    fields = result["rows"][0]["fields"]
    assert fields["answer"] == "hello" and fields["score"] == 0.5 and "judge_result" in fields
    assert result["query_missing"] is False and result["answer_log"].endswith("answer.log")
    assert snapshot(prepared) == before
    assert app.context.components["home.evaluation_history"]["default"].list_results() == []
    assert debug["result_dir"] != scored["result_dir"]


@pytest.mark.asyncio
async def test_partial_missing_and_corrupt_runs_are_visible_without_rebuilding(app, prepared):
    run = write_run(prepared)
    write_json(
        run / "partial_summary.json",
        {
            "case_count": 1,
            "cases": {
                "case~a": {
                    "mean_score": 0.0,
                    "construction": {"status": "partial_success"},
                    "construction_error_count": 2,
                },
            },
        },
    )
    write_json(run / "failure.json", {"error": "failure" * 300})
    missing = write_run(prepared, "2", mode="debug", status="running")
    corrupt = write_run(prepared, "3")
    (corrupt / "manifest.json").write_text("invalid JSON")
    before = snapshot(prepared)
    result = await call(app, "home_list_validation_results")
    assert result["row_count"] == 3
    assert sum("read_error" in row for row in result["rows"]) == 1
    row = next(row for row in result["rows"] if row["validation_path"] == str(run))
    assert row["summary_path"].endswith("partial_summary.json") and "truncated" in row["failure"]
    result = await call(app, "home_inspect_validation", validation_path=str(run))
    assert result["rows"][0]["case_id"] == "missing-case" and result["rows"][0]["mean_score"] is None
    assert result["rows"][1]["pointer"] == "/cases/case~0a"
    result = await call(app, "home_inspect_validation", validation_path=str(missing), case_id="case~a", query_id="q1")
    assert result["query_missing"] and result["rows"][0]["score"] is None
    assert len(result["missing_files"]) == 2
    result = await call(app, "home_inspect_validation", validation_path=str(missing))
    assert result["summary_missing"] and result["rows_source"] is None and result["row_count"] == 2
    assert snapshot(prepared) == before and not (missing / "partial_summary.json").exists()


@pytest.mark.asyncio
async def test_failure_and_query_paging_previews_and_source_coordinates(app, prepared):
    run = write_run(prepared)
    folder = run / "cases/case~a"
    write_json(
        folder / "memory_construction/result.json",
        {
            "success": False,
            "jobs": [{"job": f"job-{index}", "result": {"success": False, "error": "x" * 900}} for index in range(3)],
        },
    )
    write_json(
        folder / "queries/result.json",
        {
            "queries": [
                {
                    "query_id": "q2",
                    "score": 0,
                    "answer": "a" * 3000,
                    "judge_result": {"success": True},
                    "golden_answer": ["rubric"],
                    "answer_result": None,
                },
                {"query_id": "q1", "score": None, "error": "e" * 900},
            ],
        },
    )
    result = await call(app, "home_inspect_validation", validation_path=str(run), case_id="case~a", limit=1)
    assert result["rows"][0]["query_id"] == "q1" and result["rows"][0]["pointer"] == "/queries/1"
    assert result["next_offset"] == 1 and "truncated" in result["rows"][0]["error"]
    failed_jobs = result["construction"]["failed_jobs"]
    assert failed_jobs["row_count"] == 3 and failed_jobs["next_offset"] == 1
    assert failed_jobs["rows"][0]["pointer"] == "/jobs/0/result"
    result = await call(app, "home_inspect_validation", validation_path=str(run), case_id="case~a", offset=2, limit=1)
    assert result["rows"] == [] and result["construction"]["failed_jobs"]["rows"][0]["job"] == "job-2"
    result = await call(app, "home_inspect_validation", validation_path=str(run), case_id="case~a", query_id="q2")
    assert result["rows"][0]["pointer"] == "/queries/0"
    assert "truncated" in result["rows"][0]["fields"]["answer"]
    assert result["rows"][0]["fields"]["golden_answer"] == ["rubric"]
    assert result["rows"][0]["score"] == 0


@pytest.mark.asyncio
async def test_prefix_resolves_on_branch_history_not_only_evaluated_commits(app, prepared):
    repo = prepared / "code/repo"
    initial = command(repo, "rev-parse", "HEAD")
    await call(app, "home_create_candidate_branch", branch_name="candidate")
    write_run(prepared, commit=initial)
    write_run(prepared, branch="other", commit=initial)
    result = await call(app, "home_list_validation_results", commit_sha=initial[:7].upper())
    assert result["branch_name"] == "candidate" and result["commit_sha"] == initial
    assert result["row_count"] == 1
    # An unevaluated reachable commit can make a short prefix ambiguous.
    tree = command(repo, "rev-parse", "HEAD^{tree}")
    for index in range(500):
        sha = command(
            repo,
            "-c",
            "user.name=Home",
            "-c",
            "user.email=home@local",
            "commit-tree",
            tree,
            "-p",
            initial,
            "-m",
            f"unevaluated {index}",
        )
        if sha[0] == initial[0]:
            break
    else:
        pytest.fail("Could not construct a prefix collision")
    command(repo, "update-ref", "refs/heads/candidate", sha)
    response = await app.run_job("home_list_validation_results", commit_sha=initial[0])
    assert not response.success and "2 matches" in response.answer
    # The very same prefix remains unique on init, despite the other Git object.
    result = await call(app, "home_list_validation_results", branch_name="init", commit_sha=initial[0])
    assert result["commit_sha"] == initial and result["row_count"] == 0
    result = await call(app, "home_list_validation_results", commit_sha=sha)
    assert result["commit_sha"] == sha and result["rows"] == []
    absent = "f" * len(initial)
    assert not (await app.run_job("home_list_validation_results", commit_sha=absent)).success


@pytest.mark.asyncio
async def test_branch_filters_and_deleted_historical_branch_do_not_require_git(app, prepared):
    write_run(prepared, branch="removed")
    write_run(prepared, "2", branch="candidate")
    result = await call(app, "home_list_validation_results", branch_name="removed")
    assert result["row_count"] == 1 and result["rows"][0]["branch_name"] == "removed"
    response = await app.run_job("home_list_validation_results", branch_name="removed", commit_sha="a")
    assert not response.success


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["summary", "options"])
async def test_commit_filtered_catalog_reports_corrupt_run_with_identity(app, prepared, corruption):
    sha = command(prepared / "code/repo", "rev-parse", "HEAD")
    run = write_run(prepared, branch="init", commit=sha, options=None if corruption == "options" else {})
    if corruption == "summary":
        write_json(run / "summary.json", [])
    result = await call(app, "home_list_validation_results", commit_sha=sha[:7])
    assert result["row_count"] == 1
    assert result["rows"][0]["commit_sha"] == sha and "read_error" in result["rows"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "job,kwargs",
    [
        ("home_list_validation_results", {"offset": -1}),
        ("home_list_validation_results", {"limit": 0}),
        ("home_list_validation_results", {"limit": 101}),
        ("home_list_validation_results", {"limit": True}),
        ("home_list_validation_results", {"offset": "1"}),
        ("home_list_validation_results", {"commit_sha": "HEAD"}),
        ("home_list_validation_results", {"commit_sha": ""}),
        ("home_list_validation_results", {"branch_name": "../init"}),
        ("home_inspect_validation", {}),
        ("home_inspect_validation", {"validation_path": "evaluations"}),
        ("home_inspect_validation", {"validation_path": "../outside"}),
        ("home_inspect_validation", {"validation_path": "/tmp/outside"}),
        ("home_inspect_validation", {"validation_path": "evaluations/a/b/c", "query_id": "q"}),
        ("home_inspect_validation", {"validation_path": "evaluations/a/b/c", "case_id": "a/b"}),
        ("home_list_validation_results", {"_allowed_paths": []}),
    ],
)
async def test_invalid_input_fails_closed(app, prepared, job, kwargs):
    assert not (await app.run_job(job, **kwargs)).success


@pytest.mark.asyncio
async def test_unknown_ids_and_symlink_artifacts_are_rejected(app, prepared, tmp_path):
    run = write_run(prepared)
    assert not (await app.run_job("home_inspect_validation", validation_path=str(run), case_id="other")).success
    assert not (
        await app.run_job(
            "home_inspect_validation",
            validation_path=str(run),
            case_id="case~a",
            query_id="other",
        )
    ).success
    secret = tmp_path / "secret.json"
    secret.write_text('{"secret": "must not be read"}')
    (run / "summary.json").symlink_to(secret)
    response = await app.run_job("home_inspect_validation", validation_path=str(run))
    assert not response.success and "must not be read" not in response.answer
    result = await call(app, "home_list_validation_results")
    assert "read_error" in result["rows"][0] and "secret" not in str(result["rows"])
    (run / "summary.json").unlink()
    (run / "cases").symlink_to(tmp_path, target_is_directory=True)
    assert not (await app.run_job("home_inspect_validation", validation_path=str(run), case_id="case~a")).success


@pytest.mark.asyncio
async def test_hidden_case_scope_is_enforced_by_agent_job_injection(app, prepared):
    run = write_run(prepared)
    job = app.context.jobs["home_inspect_validation"]
    tool = AsAgentWrapper._make_tool(job, injected_job_kwargs={"validation_path": str(run), "case_id": "case~a"})
    assert "validation_path" not in tool.input_schema["properties"] and "case_id" not in tool.input_schema["properties"]
    await tool.call(query_id="q1")
    with pytest.raises(ValueError, match="injected"):
        await tool.call(case_id="missing-case")
    with pytest.raises(ValueError, match="injected"):
        await tool.call(validation_path="evaluations/other/commit/1")


@pytest.mark.asyncio
async def test_cancelled_job_drains_read_thread(app, prepared, monkeypatch):
    entered, release, finished = Event(), Event(), Event()

    def read(*args):
        entered.set()
        assert release.wait(5)
        finished.set()
        return {}

    monkeypatch.setattr(inspection, "_list_results", read)
    task = asyncio.create_task(app.run_job("home_list_validation_results"))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()
