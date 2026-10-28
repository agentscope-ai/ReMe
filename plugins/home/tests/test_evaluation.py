"""Real Home Jobs and pool scheduling with isolated fake Docker/model boundaries."""

# pylint: disable=missing-function-docstring,missing-class-docstring,unused-argument,protected-access

import asyncio
import io
import json
from pathlib import Path
import tarfile
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from reme_home.evaluation import component as evaluation
from reme_home.evaluation.artifacts import inspect_archive

# Reuse the real Application, committed repository and tiny source-dataset fixtures.
from test_preparation import (  # noqa: F401  # pylint: disable=unused-import
    home_application,
    preparation_inputs,
    command,
)


def archive(path, files):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as stream:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            stream.addfile(info, io.BytesIO(data))
    return path


class FakeWorker:
    def __init__(self, factory):
        self.factory = factory
        self.container_id = f"fake-{len(factory.workers)}"
        self.reusable = True
        self.case_id = None
        self.queries = []
        self.closed = False

    async def reset_case(self, case_id):
        self.case_id = case_id
        self.factory.events.append((case_id, "reset"))

    async def upload_workspace(self, source):
        inspect_archive(Path(source), 1_000_000)
        self.factory.events.append((self.case_id, "upload"))

    async def run_build(self, jobs):
        self.factory.events.append((self.case_id, "build", jobs))
        await asyncio.sleep(0)
        success = self.case_id != self.factory.fail_build
        result = {"success": success}
        if not success:
            result["exception"] = {"type": "ValueError", "module": "builtins"}
        return {"success": success, "jobs": [{"job": job.job, "result": dict(result)} for job in jobs]}

    async def run_queries(self, queries):
        self.queries = queries
        self.factory.events.append((self.case_id, "answer"))
        if self.factory.block is not None:
            self.factory.entered.set()
            await self.factory.block.wait()
        return {
            "success": True,
            "queries": [
                {
                    "query_id": q.query_id,
                    "answer": "hello",
                    "answer_result": {"success": True, "token_usage": {"answer": {"total_tokens": 2}}},
                    "error": None,
                }
                for q in queries
            ],
        }

    async def export_workspace(self, destination):
        self.factory.events.append((self.case_id, "snapshot"))
        return archive(Path(destination), {"daily/note.md": self.case_id, ".git/HEAD": "ref: refs/heads/main\n"})

    async def export_build_log(self, destination):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_text("session construction log", encoding="utf-8")
        return Path(destination)

    async def read_build_log(self):
        return "failed build log"

    async def export_evaluation(self, destination):
        files = {
            "reme_workspace/daily/note.md": "after-answer",
            "manifest.json": "{}",
            "queries/result.json": '{"queries": [{"query_id": "fact:1", "answer": "forged"}]}',
        }
        for query in self.queries:
            files[f"queries/{query.query_id}/answer.log"] = "answer log"
        if self.factory.bad_export:
            files["../escaped"] = "invalid"
        return archive(Path(destination), files)

    async def export_full(self, destination):
        return archive(Path(destination), {"build_log/build.log": "failed diagnostic"})

    async def close(self):
        self.closed = True


class FakeFactory:
    instances = []
    fail_build = None
    bad_export = False
    block = None

    def __init__(self, candidate, **kwargs):
        self.candidate = candidate
        self.kwargs = kwargs
        self.events = []
        self.workers = []
        self.entered = asyncio.Event()
        self.instances.append(self)

    async def create_cases(self, ids, **kwargs):
        self.workers = [FakeWorker(self) for _ in ids]
        return self.workers


@pytest_asyncio.fixture(name="prepared")
async def prepared_home(app, inputs, monkeypatch):
    spec, _ = inputs
    source = Path(spec["benchmark"]["source"])
    for path in source.glob("chats/*/*/probing_questions/probing_questions.json"):
        raw = json.loads(path.read_text())
        raw["fact"][0]["rubric"] = ["says hello"]
        path.write_text(json.dumps(raw))
    if "lme-judge" in app.config.plugins:
        spec["benchmark"] = {"name": "longmemeval", "source": str(inputs[1]), "all_cases": True}
        spec["runtime"]["plugins"] = []
        spec["candidate"]["paths"] = ["reme/candidate.py"]
    response = await app.run_job("home_prepare", spec=spec)
    assert response.success, response.answer
    FakeFactory.instances = []
    monkeypatch.setattr(evaluation, "DockerReMeSandboxFactory", FakeFactory)
    # Run the real judge Job and Step; mock only its model-facing wrapper.
    wrapper = app.context.components["agent_wrapper"]["judge"]
    monkeypatch.setattr(wrapper, "reply", AsyncMock(return_value={"result": '{"score": 0.5}'}))
    return Path(app.config.workspace_dir) / "home"


@pytest.mark.asyncio
async def test_case_and_query_selection_are_host_injected(app, prepared):
    from reme.components.agent_wrapper import AsAgentWrapper

    job = app.context.jobs["home_evaluate"]
    assert "case_ids" not in job.parameters["properties"]
    assert "query_ids" not in job.parameters["properties"]
    tool = AsAgentWrapper._make_tool(
        job,
        "ctx-1",
        {"case_ids": ["1"], "query_ids": {"1": ["fact:1"]}},
    )
    assert "case_ids" not in tool.input_schema["properties"]
    assert "query_ids" not in tool.input_schema["properties"]
    await tool.call()
    assert {event[0] for event in FakeFactory.instances[0].events} == {"1"}
    assert app.context.components["agent_wrapper"]["judge"].reply.await_count == 1


@pytest.mark.asyncio
async def test_all_cases_and_artifacts_via_real_job(app, prepared):
    response = await app.run_job("home_evaluate", concurrency=1)
    assert response.success, response.answer
    result = response.answer
    assert result["case_count"] == result["scored_query_count"] == 2
    assert result["mean_query_score"] == result["observed_mean_query_score"] == 0.5
    run = Path(result["result_dir"])
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["case_ids"] == ["2", "1"]
    assert {"case_ids", "query_ids", "reuse_construction_from"}.isdisjoint(manifest["options"])
    assert "judge_model" not in manifest
    for case_id in ["1", "2"]:
        folder = run / "cases" / case_id
        assert (folder / "memory_construction/reme_workspace/daily/note.md").read_text() == case_id
        assert (folder / "memory_construction/build.log").is_file()
        assert (folder / "queries/fact:1/answer.log").is_file()
        assert not (folder / "evaluation.tar.gz").exists()
        assert not (folder / "execution.json").exists()
        assert not (folder / "artifacts.json").exists()
        assert not (folder / "queries/answers.json").exists()
        assert not (folder / "failure.json").exists()
        row = json.loads((folder / "queries/result.json").read_text())["queries"][0]
        assert row["golden_answer"] == "hello" and row["score"] == 0.5
        assert row["answer"] == "hello"
    factory = FakeFactory.instances[0]
    for case_id in ["1", "2"]:
        events = [event[1] for event in factory.events if event[0] == case_id]
        assert events.index("snapshot") < events.index("answer")
    assert all(worker.closed for worker in factory.workers)
    second = await app.run_job("home_evaluate", case_ids=["1"])
    assert second.success and Path(second.answer["result_dir"]).name == "2"


@pytest.mark.asyncio
async def test_reuse_tar_runs_current_candidate_without_building(app, prepared):
    baseline = await app.run_job("home_evaluate")
    assert baseline.success, baseline.answer
    repo = prepared / "code/repo"
    (repo / "plugins/beam/src/reme_beam/__init__.py").write_text("value = 2\n")
    command(repo, "add", ".")
    command(repo, "-c", "user.name=Fixture", "-c", "user.email=test@localhost", "commit", "-m", "new answer")
    reused = await app.run_job("home_evaluate", case_ids=["1"], reuse_construction_from=baseline.answer["result_dir"])
    assert reused.success, reused.answer
    assert reused.answer["commit_sha"] != baseline.answer["commit_sha"]
    assert reused.answer["execution_mode"] == "query_only_replay"
    events = FakeFactory.instances[-1].events
    assert ("1", "upload") in events
    assert not any(event[1] == "build" for event in events)
    run = Path(reused.answer["result_dir"])
    memory = run / "cases/1/memory_construction"
    construction = json.loads((memory / "result.json").read_text())
    assert construction["status"] == "reused" and not construction["jobs"]
    assert (memory / "reme_workspace/daily/note.md").read_text() == "1"


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["missing", "corrupt", "failure", "escape"])
async def test_reuse_preflight_rejects_entire_request(app, prepared, mutation):
    baseline = await app.run_job("home_evaluate", case_ids=["1"])
    assert baseline.success, baseline.answer
    run = Path(baseline.answer["result_dir"])
    kwargs = {"reuse_construction_from": str(run), "case_ids": ["1"]}
    if mutation == "missing":
        kwargs.pop("case_ids")  # All training cases, not merely cases present in source.
    elif mutation == "corrupt":
        (run / "cases/1/memory_construction/reme_workspace.tar.gz").write_bytes(b"broken")
    elif mutation == "failure":
        result = run / "cases/1/memory_construction/result.json"
        raw = json.loads(result.read_text())
        raw["success"] = False
        result.write_text(json.dumps(raw))
    else:
        kwargs["reuse_construction_from"] = "../escape"
    count = len(FakeFactory.instances)
    response = await app.run_job("home_evaluate", **kwargs)
    assert not response.success
    assert len(FakeFactory.instances) == count
    assert len(list((prepared / "evaluations").glob("*/*/*/manifest.json"))) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"case_ids": []},
        {"case_ids": ["1", "1"]},
        {"case_ids": ["absent"]},
        {"query_ids": {"1": ["absent"]}},
        {"eval_only": True},
        {"code_ref": "HEAD"},
        {"code_ref": None},
        {"judge_concurrency": 2},
        {"dataset_kind": "debug-cases"},
        {"debug": "true"},
        {"debug": True, "reuse_construction_from": "evaluations/prior"},
        {"debug": True, "case_ids": ["1"]},
        {"_allowed_paths": []},
    ],
)
async def test_invalid_inputs_do_not_start_pool(app, prepared, kwargs):
    response = await app.run_job("home_evaluate", **kwargs)
    assert not response.success
    assert not FakeFactory.instances
    assert not list((prepared / "evaluations").iterdir())


@pytest.mark.asyncio
async def test_build_failure_retains_diagnostics_and_other_cases(app, prepared, monkeypatch):
    monkeypatch.setattr(FakeFactory, "fail_build", "1")
    response = await app.run_job("home_evaluate")
    assert response.success, response.answer
    run = Path(response.answer["result_dir"])
    assert response.answer["scored_query_count"] == 1
    assert response.answer["mean_query_score"] == 0.25
    assert (run / "cases/1/full.tar.gz").is_file()
    assert (run / "cases/1/memory_construction/build.log").is_file()
    assert (run / "cases/1/failure.json").is_file()
    summary = json.loads((run / "summary.json").read_text())
    assert summary["cases"]["1"]["queries"] == {}
    assert summary["query_status_counts"] == {
        "completed": 1,
        "candidate_failure": 0,
        "infra_error": 0,
        "unscored": 1,
    }


@pytest.mark.asyncio
async def test_judge_failure_preserves_answers_and_fail_fast_result(app, prepared, monkeypatch):
    monkeypatch.setattr(
        app.context.components["agent_wrapper"]["judge"],
        "reply",
        AsyncMock(side_effect=RuntimeError("judge unavailable")),
    )
    response = await app.run_job("home_evaluate", case_ids=["1"])
    assert not response.success
    run = Path(response.answer["result_dir"])
    assert response.answer["query_status_counts"]["unscored"] == 1
    assert json.loads((run / "cases/1/queries/result.json").read_text())["queries"][0]["answer"] == "hello"
    assert (run / "cases/1/queries/fact:1/failure.json").is_file()
    failed = await app.run_job("home_evaluate", fail_fast=True)
    assert not failed.success
    assert list((prepared / "evaluations").glob("*/*/*/partial_summary.json"))
    assert all(worker.closed for factory in FakeFactory.instances for worker in factory.workers)


@pytest.mark.asyncio
async def test_cancellation_closes_pool_and_retains_partial_run(app, prepared, monkeypatch):
    monkeypatch.setattr(FakeFactory, "block", asyncio.Event())
    task = asyncio.create_task(app.run_job("home_evaluate", concurrency=1))
    while not FakeFactory.instances:
        await asyncio.sleep(0.01)
    factory = FakeFactory.instances[0]
    await asyncio.wait_for(factory.entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(worker.closed for worker in factory.workers)
    assert list((prepared / "evaluations").glob("*/*/*/failure.json"))
    manifest = json.loads(next((prepared / "evaluations").glob("*/*/*/manifest.json")).read_text())
    assert manifest["status"] == "cancelled"


@pytest.mark.asyncio
async def test_debug_skips_judge_and_invalid_export_keeps_answer(app, prepared, monkeypatch):
    response = await app.run_job("home_evaluate", debug=True)
    assert response.success, response.answer
    assert app.context.components["agent_wrapper"]["judge"].reply.await_count == 0
    assert response.answer["scored_query_count"] == 0
    monkeypatch.setattr(FakeFactory, "bad_export", True)
    response = await app.run_job("home_evaluate", case_ids=["1"])
    assert not response.success
    run = Path(response.answer["result_dir"])
    assert (run / "cases/1/queries/result.json").is_file()
    assert response.answer["scored_query_count"] == 1
    assert not list(prepared.rglob("escaped"))
    assert "artifact" in (run / "cases/1/failure.json").read_text()
    assert response.answer["mean_query_score"] is None
    assert response.answer["case_status_counts"]["infra_error"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "debug,override,expected_fail_fast",
    [(False, None, False), (True, None, True), (True, False, False), (False, True, True)],
)
async def test_mode_controls_data_scoring_and_fail_fast_defaults(app, prepared, debug, override, expected_fail_fast):
    kwargs = {"debug": debug}
    if override is not None:
        kwargs["fail_fast"] = override
    response = await app.run_job("home_evaluate", **kwargs)
    assert response.success, response.answer
    run = Path(response.answer["result_dir"])
    manifest = json.loads((run / "manifest.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    assert manifest["options"]["fail_fast"] is expected_fail_fast
    assert manifest["options"]["debug"] is debug
    assert "dataset_kind" not in manifest["options"]
    assert manifest["dataset_kind"] == ("debug-cases" if debug else "training-set")
    assert manifest["mode"] == ("debug" if debug else "scored")
    assert set(summary["cases"]) == ({"debug-1", "debug-2"} if debug else {"1", "2"})
    assert summary["expected_query_count"] == 2
    assert summary["scored_query_count"] == (0 if debug else 2)
    assert manifest["status"] == "completed"
    assert summary["mean_query_score"] == (None if debug else 0.5)
    assert summary["observed_mean_query_score"] == (None if debug else 0.5)
    assert summary["query_status_counts"]["unscored"] == 0
    for case in summary["cases"].values():
        assert case["mean_score"] == (None if debug else 0.5)
    assert app.context.components["agent_wrapper"]["judge"].reply.await_count == (0 if debug else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_fast", [True, False])
async def test_debug_empty_answers_fail_without_scoring(app, prepared, monkeypatch, fail_fast):
    original = FakeWorker.run_queries

    async def empty_answer(worker, queries):
        result = await original(worker, queries)
        for row in result["queries"]:
            row["answer"] = "  "
        return result

    monkeypatch.setattr(FakeWorker, "run_queries", empty_answer)
    response = await app.run_job("home_evaluate", debug=True, fail_fast=fail_fast, case_ids=["debug-1"])
    assert not response.success
    runs = list((prepared / "evaluations").glob("*/*/*/manifest.json"))
    assert len(runs) == 1
    run = runs[0].parent
    summary = json.loads((run / ("partial_summary.json" if fail_fast else "summary.json")).read_text())
    assert summary["mean_query_score"] is None
    assert summary["observed_mean_query_score"] is None
    assert summary["scored_query_count"] == 0
    assert summary["query_status_counts"]["unscored"] == 0
    assert summary["case_status_counts"]["candidate_failure"] == 1
    assert summary["cases"]["debug-1"]["mean_score"] is None
    assert (run / "cases/debug-1/queries/result.json").is_file()
    assert app.context.components["agent_wrapper"]["judge"].reply.await_count == 0


@pytest.mark.asyncio
async def test_current_head_is_immutable_and_does_not_ship_judge(app, prepared):
    repo = prepared / "code/repo"
    (repo / "plugins/beam/src/reme_beam/__init__.py").write_text("value = 3\n", encoding="utf-8")
    command(repo, "add", ".")
    command(repo, "-c", "user.name=Fixture", "-c", "user.email=test@localhost", "commit", "-m", "change")
    current = command(repo, "rev-parse", "HEAD")
    response = await app.run_job("home_evaluate", case_ids=["1"], query_ids={"1": ["fact:1"]})
    assert response.success, response.answer
    snapshot = FakeFactory.instances[-1].candidate.snapshot
    with tarfile.open(fileobj=io.BytesIO(snapshot.archive), mode="r:gz") as source:
        assert source.extractfile("plugins/beam/src/reme_beam/__init__.py").read() == b"value = 3\n"
        assert "plugins/beam/pyproject.toml" in source.getnames()
        assert not any("reme_home" in Path(name).parts or ".git" in Path(name).parts for name in source.getnames())
    assert response.answer["commit_sha"] == current


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["modified", "staged", "untracked", "runtime", "dataset", "symlink"])
async def test_modified_inputs_are_rejected_before_docker(app, prepared, mutation):
    if mutation in {"modified", "staged"}:
        repo = prepared / "code/repo"
        (repo / "plugins/beam/src/reme_beam/__init__.py").write_text("value = 3\n", encoding="utf-8")
        if mutation == "staged":
            command(repo, "add", ".")
    elif mutation == "untracked":
        (prepared / "code/repo/new.py").write_text("uncommitted", encoding="utf-8")
    elif mutation == "runtime":
        (prepared / "code/runtime/reme/__init__.py").write_text("modified", encoding="utf-8")
    elif mutation == "dataset":
        path = prepared / "datasets/training-set/cases/000000.json"
        raw = json.loads(path.read_text())
        raw["queries"][0]["question"] = "tampered"
        path.write_text(json.dumps(raw), encoding="utf-8")
    else:
        folder = prepared / "evaluations"
        folder.rmdir()
        folder.symlink_to(prepared / "docs", target_is_directory=True)
    before = {path.name: path.read_bytes() for path in (prepared / "evaluations").iterdir()}
    response = await app.run_job("home_evaluate")
    assert not response.success
    assert not FakeFactory.instances
    assert {path.name: path.read_bytes() for path in (prepared / "evaluations").iterdir()} == before


def test_lme_adapter_cutoff_and_memory_checkpoints():
    from reme_home.preparation.data.models import CaseSpec, QuerySpec, SessionSpec
    from reme_home.evaluation.adapters import build_jobs, evaluation_query, accepts_build

    query = QuerySpec(
        query_id="q",
        question="when",
        golden_answer="then",
        metadata={"question_date": "2024/01/02 (Tue) 00:00"},
    )
    case = CaseSpec(
        case_id="c",
        queries=[query],
        sessions=[
            SessionSpec(session_id="future", messages=[{"created_at": "2024-01-03T00:00:00"}]),
            SessionSpec(session_id="past", messages=[{"created_at": "2024-01-01T00:00:00"}]),
        ],
    )
    jobs = build_jobs("longmemeval", case)
    assert [job.job for job in jobs] == ["auto_memory", "index_update"]
    assert jobs[0].arguments["session_id"] == "past"
    assert jobs[1].memory_checkpoint == "session: past"
    assert evaluation_query("longmemeval", query).answer_arguments["query_time"] == "2024-01-02T00:00:00"
    build = {"success": False, "jobs": [{"result": {"success": n > 0}} for n in range(100)]}
    build["jobs"][0]["result"]["exception"] = {"type": "ValueError", "module": "builtins"}
    assert accepts_build(build, 0.03)
    assert not accepts_build(build, 0.01)


@pytest.mark.parametrize("kind", ["link", "duplicate", "oversize"])
def test_unsafe_archives_are_rejected_without_partial_extraction(tmp_path, kind):
    path = tmp_path / "archive.tar.gz"
    with tarfile.open(path, "w:gz") as stream:
        item = tarfile.TarInfo("data")
        item.size = 3
        stream.addfile(item, io.BytesIO(b"abc"))
        if kind == "link":
            link = tarfile.TarInfo("outside")
            link.type = tarfile.SYMTYPE
            link.linkname = "../outside"
            stream.addfile(link)
        elif kind == "duplicate":
            stream.addfile(item, io.BytesIO(b"abc"))
    destination = tmp_path / "extract"
    with pytest.raises(ValueError):
        inspect_archive(path, 2 if kind == "oversize" else 10000, destination)
    assert not destination.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("app", ["lme-judge"], indirect=True)
async def test_lme_evaluation_calls_host_judge_job(app, prepared, monkeypatch):
    wrapper = app.context.components["agent_wrapper"]["judge"]
    monkeypatch.setattr(wrapper, "reply", AsyncMock(side_effect=[{"result": "maybe"}, {"result": "yes"}]))
    response = await app.run_job("home_evaluate")
    assert response.success, response.answer
    assert response.answer["mean_query_score"] == 1
    assert wrapper.reply.await_count == 2
    run = Path(response.answer["result_dir"])
    result = json.loads((run / "cases/q1/queries/result.json").read_text())["queries"][0]
    assert result["judge_result"]["metadata"]["raw_answer_judgement"] == "yes"
    assert "answer_judgement" not in result["judge_result"]["metadata"]
    assert result["score"] == 1
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["judge"]["plugin"] == "lme-judge"
    assert manifest["judge"]["model"] == "qwen3.7-max"
    assert "api_key" not in json.dumps(manifest)


@pytest.mark.asyncio
@pytest.mark.parametrize("app", ["lme-judge"], indirect=True)
async def test_invalid_lme_judge_is_infra_error_after_one_retry(app, prepared):
    wrapper = app.context.components["agent_wrapper"]["judge"]
    wrapper.reply.return_value = {"result": "maybe"}
    response = await app.run_job("home_evaluate")
    assert not response.success
    assert response.answer["mean_query_score"] is None
    assert wrapper.reply.await_count == 2
    run = Path(response.answer["result_dir"])
    assert (run / "cases/q1/queries/result.json").is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["nonsense", '{"score": 2}', '{"score": 0.2}', '{"score": NaN}'])
async def test_invalid_beam_judge_is_infra_error(app, prepared, raw):
    app.context.components["agent_wrapper"]["judge"].reply.return_value = {"result": raw}
    response = await app.run_job("home_evaluate", case_ids=["1"])
    assert not response.success
    assert response.answer["mean_query_score"] is None
    assert response.answer["case_status_counts"]["infra_error"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("plugins", [["home"], ["home", "beam-judge", "lme-judge"]])
async def test_startup_requires_exactly_one_judge(app, tmp_path, plugins):
    from reme.application import Application

    other = Application(
        workspace_dir=str(tmp_path / "other"),
        plugins=plugins,
        service={"backend": "cli"},
        enable_logo=False,
        log_to_file=False,
        log_to_console=False,
    )
    try:
        with pytest.raises(ValueError, match="exactly one"):
            await other.start()
    finally:
        await other.close()


@pytest.mark.asyncio
async def test_preparation_rejects_mismatched_judge_without_writes(app, inputs):
    from test_preparation import snapshot

    spec, lme = inputs
    spec["benchmark"] = {"name": "longmemeval", "source": str(lme), "all_cases": True}
    root = Path(app.config.workspace_dir)
    before = snapshot(root)
    response = await app.run_job("home_prepare", spec=spec)
    assert not response.success
    assert "lme-judge" in str(response.answer)
    assert snapshot(root) == before


@pytest.mark.asyncio
async def test_prepared_workspace_rejects_wrong_judge_on_restart(app, prepared):
    from reme.application import Application

    other = Application(
        workspace_dir=app.config.workspace_dir,
        plugins=["home", "lme-judge"],
        service={"backend": "cli"},
        enable_logo=False,
        log_to_file=False,
        log_to_console=False,
    )
    try:
        with pytest.raises(ValueError, match="requires 'beam-judge'"):
            await other.start()
    finally:
        await other.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("override", [{"enabled": False}, {"steps": [{"backend": "home_prepare_step"}]}])
async def test_startup_rejects_disabled_or_replaced_judge_job(app, tmp_path, override):
    from reme.application import Application

    other = Application(
        workspace_dir=str(tmp_path / "other"),
        plugins=["home", "beam-judge"],
        jobs={"answer_judge": override},
        service={"backend": "cli"},
        enable_logo=False,
        log_to_file=False,
        log_to_console=False,
    )
    try:
        with pytest.raises(ValueError, match="answer_judge"):
            await other.start()
    finally:
        await other.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["answer", "construction"])
@pytest.mark.parametrize("kind", ["timeout", "wrapped_timeout", "http_429", "http_503", "code", "legacy"])
async def test_real_job_errors_reach_worker_and_home_summary(app, prepared, monkeypatch, phase, kind):
    from types import SimpleNamespace
    import httpx
    from deploy.benchmark_docker.worker import _run_job_on_app
    from reme.components.job.base_job import BaseJob

    async def failing_step(_context):
        if kind == "timeout":
            raise TimeoutError("model timed out")
        if kind == "wrapped_timeout":
            try:
                raise TimeoutError("model timed out")
            except TimeoutError as error:
                raise RuntimeError("model invocation failed") from error
        if kind.startswith("http_"):
            request = httpx.Request("POST", "https://example.test")
            response = httpx.Response(int(kind.split("_")[1]), request=request)
            raise httpx.HTTPStatusError("model request failed", request=request, response=response)
        raise ValueError("candidate bug")

    candidate = BaseJob(name="candidate-error", app_context=app.context)
    candidate._build_steps = lambda: [failing_step]
    worker_app = SimpleNamespace(context=app.context, run_job=lambda _name, **kwargs: candidate(**kwargs))
    result = await _run_job_on_app(worker_app, "candidate-error", {})
    if kind == "legacy":
        result.pop("exception")
        result["metadata"] = {}
    else:
        assert result["exception"] == result["metadata"]["reme.job_error"]
    result = json.loads(json.dumps(result))  # The real sandbox boundary is JSON.

    if phase == "answer":

        async def answer(worker, queries):
            return {
                "success": False,
                "queries": [
                    {
                        "query_id": query.query_id,
                        "answer": result["answer"],
                        "answer_result": result,
                        "error": result["error"],
                        **({"exception": result["exception"]} if "exception" in result else {}),
                    }
                    for query in queries
                ],
            }

        monkeypatch.setattr(FakeWorker, "run_queries", answer)
    else:

        async def build(worker, jobs):
            # Two failures are below the 3% threshold. An earlier code failure
            # must not conceal a later infrastructure/unclassified failure.
            return {
                "success": False,
                "jobs": [
                    {"result": {"success": False, "exception": {"type": "ValueError", "module": "builtins"}}},
                    {"result": result},
                    *[{"result": {"success": True}} for _ in range(98)],
                ],
            }

        monkeypatch.setattr(FakeWorker, "run_build", build)

    response = await app.run_job("home_evaluate", case_ids=["1"])
    code_failure = kind == "code"
    assert response.success is code_failure
    summary = response.answer
    assert summary["case_status_counts"]["infra_error"] == (0 if code_failure else 1)
    assert summary["mean_query_score"] == (
        0.5 if code_failure and phase == "construction" else 0 if code_failure else None
    )
    run = Path(summary["result_dir"])
    if not code_failure:
        failure = json.loads((run / "cases/1/failure.json").read_text())
        assert failure["failure_kind"] == "infra_error"
        if kind != "legacy":
            assert failure["exception"] == result["exception"]
    assert all(worker.closed for worker in FakeFactory.instances[-1].workers)


@pytest.mark.asyncio
async def test_late_cleanup_failure_survives_without_duplicate_execution_record(app, prepared, monkeypatch):
    async def close(worker):
        worker.closed = True
        raise RuntimeError("late cleanup failure")

    monkeypatch.setattr(FakeWorker, "close", close)
    response = await app.run_job("home_evaluate", case_ids=["1"])
    assert not response.success
    assert response.answer["mean_query_score"] is None
    assert response.answer["observed_mean_query_score"] == 0.5
    folder = Path(response.answer["result_dir"]) / "cases/1"
    failure = json.loads((folder / "failure.json").read_text())
    assert failure["stage"] == "cleanup" and failure["failure_kind"] == "infra_error"
    assert "late cleanup failure" in failure["error"]
    assert not (folder / "execution.json").exists()
