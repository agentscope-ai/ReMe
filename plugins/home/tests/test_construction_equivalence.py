"""Host-selected query-only replay, construction evidence and sandbox isolation."""

# pylint: disable=missing-function-docstring,missing-class-docstring,unused-argument,protected-access

import asyncio
import os
from pathlib import Path
import sys
from threading import Event
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agentscope.model import ChatUsage, StructuredResponse
from reme_home.agent import candidate_gate, construction_equivalence, optimizer, workflow_common
from reme_home.code.snapshot import read_code_file
from reme_home.evaluation import reuse
from reme_home.evaluation.artifacts import file_sha256, read_json, write_json
from reme_home.evaluation.budget import validation_budget_status
from reme_home.evaluation.inputs import EvaluationSpec

from test_evaluation import (  # noqa: F401  # pylint: disable=unused-import
    FakeFactory,
    command,
    commit_candidate,
    home_application,
    preparation_inputs,
    prepared_home,
    os_sandbox_available,
)
from test_workflows import ScriptedModel, final_report, set_model

from reme.utils.local_backend import LocalProcessBackend


@pytest.fixture(name="sandbox")
def review_sandbox(monkeypatch):
    policies = []

    def create(**kwargs):
        policies.append(kwargs)
        local = LocalProcessBackend()
        return SimpleNamespace(backend=local, exec_shell=local.exec_shell)

    monkeypatch.setattr(workflow_common, "create_case_study_sandbox", create)
    return policies


@pytest.fixture(name="reviewer")
def real_reviewer(app, prepared, monkeypatch):
    wrapper = app.context.components["agent_wrapper"]["home_construction_equivalence"]
    monkeypatch.setattr(wrapper, "reply", MethodType(type(wrapper).reply, wrapper))
    gate = app.context.components["agent_wrapper"]["home_candidate_gate"]
    monkeypatch.setattr(gate, "reply", AsyncMock(return_value={"result": '{"decision":"accept","reason":"Mechanism"}'}))
    return wrapper


def force_draw(monkeypatch, draw):
    monkeypatch.setattr(reuse.random, "SystemRandom", lambda: SimpleNamespace(random=lambda: draw))


@pytest.mark.asyncio
async def test_corrected_equivalence_authorizes_replay(app, prepared, reviewer, sandbox, monkeypatch):
    assert (await app.run_job("home_evaluate")).success
    commit_candidate(prepared)
    model = ScriptedModel(final="Construction is unchanged: only query behavior changes.")
    model.generate_structured_output = AsyncMock(
        return_value=StructuredResponse(
            content={"decision": "construction_unchanged", "reason": "Only query behavior changes"},
            usage=ChatUsage(input_tokens=7, output_tokens=11, time=0),
        ),
    )
    set_model(app, "home_construction_equivalence", model)
    force_draw(monkeypatch, 0.1)
    response = await app.run_job("home_evaluate", sampling=True)
    assert response.success, response.answer
    assert response.answer["execution_mode"] == "query_only_replay"
    report = response.answer["automatic_construction_reuse"]["equivalence_report"]
    assert report["decision"] == "construction_unchanged" and "review_failed" not in report
    assert report["usage"]["input_tokens"] == 9 and report["usage"]["output_tokens"] == 14
    assert read_json(Path(report["review_path"]))["session_id"]
    assert not Path(report["source_path"]).exists()
    model.generate_structured_output.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [False, True])
async def test_large_resources_export_for_both_reviewers(app, prepared, reviewer, sandbox, monkeypatch, changed):
    assert (await app.run_job("home_evaluate")).success
    commit_candidate(prepared)
    repo = prepared / "code/repo"
    relative = "plugins/beam/src/reme_beam/large_resource.bin"
    resource = repo / relative
    resource.write_bytes(b"\0" * 5_000_001)
    command(repo, "add", ".")
    command(repo, "-c", "user.name=Fixture", "-c", "user.email=test@localhost", "commit", "-m", "Large resource")
    source_sha = file_sha256(resource)
    assert (await app.run_job("home_evaluate")).success
    commit_candidate(prepared)
    if changed:
        with resource.open("ab") as output:
            output.write(b"updated")
        command(repo, "add", ".")
        command(repo, "-c", "user.name=Fixture", "-c", "user.email=test@localhost", "commit", "-m", "Update resource")
    candidate_sha = file_sha256(resource)
    # Browsing retains its existing output guard; only complete exports bypass it.
    with pytest.raises(ValueError, match="Code file exceeds 5 MB"):
        read_code_file(repo, {}, path=relative)
    inspected = []

    def inspect_export(category):
        root = next((prepared / "tmp").glob(f"{category}-*"))
        assert file_sha256(root / "base" / relative) == source_sha
        assert file_sha256(root / "candidate" / relative) == candidate_sha
        inspected.append(category)

    async def accept(*args, **kwargs):
        inspect_export("candidate-gate")
        return {"result": '{"decision":"accept","reason":"Fixture export check"}'}

    def finish():
        inspect_export("construction-equivalence")
        return '{"decision":"construction_unchanged","reason":"Fixture export check"}'

    gate = app.context.components["agent_wrapper"]["home_candidate_gate"]
    monkeypatch.setattr(gate, "reply", accept)
    set_model(
        app,
        "home_construction_equivalence",
        ScriptedModel(final=finish),
    )
    force_draw(monkeypatch, 0.1)
    response = await app.run_job("home_evaluate", sampling=True, max_evaluations=10)
    assert response.success, response.answer
    assert response.answer["execution_mode"] == "query_only_replay"
    for report in (
        response.answer["candidate_gate"],
        response.answer["automatic_construction_reuse"]["equivalence_report"],
    ):
        assert not Path(report["source_path"]).exists()
    assert inspected == ["candidate-gate", "construction-equivalence"]


@pytest.mark.asyncio
@pytest.mark.parametrize("draw,selected", [(0.0, True), (0.599999, True), (0.6, False), (0.99, False)])
async def test_equivalent_candidate_uses_host_probability(
    app,
    prepared,
    reviewer,
    sandbox,
    monkeypatch,
    draw,
    selected,
):
    baseline = await app.run_job("home_evaluate")
    source = Path(baseline.answer["result_dir"])
    commit_candidate(prepared)

    def finish():
        evidence = next((prepared / "tmp").glob("construction-equivalence-*"))
        assert (evidence / "runtime/reme/__init__.py").is_file()
        assert not list(evidence.rglob(".git"))
        assert not list(evidence.rglob("queries"))
        assert not (evidence / "manifest.json").exists()
        for case in ["1", "2"]:
            assert (evidence / f"construction/{case}/workspace/daily/note.md").read_text() == case
            assert (evidence / f"construction/{case}/build.log").is_file()
        return '{"decision":"construction_unchanged","reason":"Only query behavior changes"}'

    model = ScriptedModel(
        [("Read", lambda: {"file_path": str(next((prepared / "tmp").glob("construction-equivalence-*/diff.patch")))})],
        final=finish,
    )
    set_model(app, "home_construction_equivalence", model)
    force_draw(monkeypatch, draw)
    result = await app.run_job("home_evaluate", sampling=True, max_evaluations=10)
    assert result.success, result.answer
    decision = result.answer["automatic_construction_reuse"]
    assert decision["selected"] == selected and decision["draw"] == draw and decision["probability"] == 0.6
    assert decision["source_validation"] == str(source)
    report = decision["equivalence_report"]
    assert read_json(Path(report["review_path"]))["decision"] == "construction_unchanged"
    assert report["session_id"] and report["usage"]["input_tokens"] == 4
    assert report["base_commit"] == baseline.answer["commit_sha"]
    assert report["candidate_commit"] == result.answer["commit_sha"]
    assert {tool["function"]["name"] for tool in model.calls[0]["tools"]} == {"Read", "Glob", "Grep", "AutoBash"}
    assert sandbox and all(policy["writable_paths"] == [] for policy in sandbox)
    review_policies = [policy for policy in sandbox if "construction-equivalence-" in policy["cwd"]]
    assert review_policies and all(policy["read_only_paths"] == [report["source_path"]] for policy in review_policies)
    assert not Path(report["source_path"]).exists()
    events = {event[1] for event in FakeFactory.instances[-1].events}
    assert ("upload" in events) == selected and ("build" in events) != selected
    assert result.answer["execution_mode"] == ("query_only_replay" if selected else "full_validation")
    run = Path(result.answer["result_dir"])
    manifest = read_json(run / "manifest.json")
    assert manifest["automatic_construction_reuse"] == decision
    assert validation_budget_status(prepared, 10)["used_validations"] == 1
    optimizer.verify_evaluation(prepared, run, result.answer["commit_sha"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output",
    [
        '{"decision":"construction_changed","reason":"Construction storage changes"}',
        "malformed",
        '{"decision":"construction_unchanged","reason":"x","extra":true}',
        "error",
    ],
)
async def test_changed_or_failed_review_rebuilds_without_drawing(app, prepared, reviewer, sandbox, monkeypatch, output):
    assert (await app.run_job("home_evaluate")).success
    commit_candidate(prepared)
    set_model(app, "home_construction_equivalence", ScriptedModel(final=output))
    if output == "error":
        monkeypatch.setattr(reviewer, "reply", AsyncMock(side_effect=RuntimeError("model offline")))
    monkeypatch.setattr(reuse, "select_query_only", lambda: pytest.fail("Cannot draw without equivalence"))
    response = await app.run_job("home_evaluate", sampling=True)
    assert response.success, response.answer
    decision = response.answer["automatic_construction_reuse"]
    assert decision["selected"] is False and decision["reason"] == "construction_not_equivalent"
    assert decision["equivalence_report"]["decision"] == "construction_changed"
    if output != '{"decision":"construction_changed","reason":"Construction storage changes"}':
        assert decision["equivalence_report"]["review_failed"]
    assert not list((prepared / "tmp").glob("construction-equivalence-*"))
    assert any(event[1] == "build" for event in FakeFactory.instances[-1].events)


@pytest.mark.asyncio
async def test_skip_baseline_debug_plain_calls_and_missing_ancestor(app, prepared, monkeypatch):
    async def unexpected(*args, **kwargs):
        pytest.fail("No reviewer expected")

    monkeypatch.setattr(construction_equivalence, "review_construction", unexpected)
    commit_candidate(prepared)
    response = await app.run_job("home_evaluate", sampling=True)
    assert response.success and response.answer["automatic_construction_reuse"]["reason"] == "no_reusable_ancestor"
    commit_candidate(prepared)
    assert (await app.run_job("home_evaluate", debug=True, sampling=True)).success
    response = await app.run_job("home_evaluate")
    assert response.success and "automatic_construction_reuse" not in response.answer


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["checksum", "failed_construction", "answer", "config", "unfinished", "debug"])
async def test_invalid_nearest_source_falls_back_to_older_ancestor(
    app,
    prepared,
    reviewer,
    sandbox,
    monkeypatch,
    invalid,
):
    baseline = await app.run_job("home_evaluate")
    commit_candidate(prepared)
    recent = await app.run_job("home_evaluate", case_ids=["1"])
    source = Path(recent.answer["result_dir"])
    if invalid == "checksum":
        (source / "cases/1/memory_construction/reme_workspace.tar.gz").write_bytes(b"bad")
    elif invalid == "failed_construction":
        path = source / "cases/1/memory_construction/result.json"
        write_json(path, {**read_json(path), "success": False})
    elif invalid == "answer":
        write_json(source / "cases/1/queries/result.json", {"queries": [{"answer": "", "error": "failed"}]})
    else:
        manifest = read_json(source / "manifest.json")
        if invalid == "config":
            manifest["fingerprints"]["construction_config"] = "different"
        elif invalid == "debug":
            manifest["mode"] = "debug"
        else:
            manifest["status"] = "failed"
        write_json(source / "manifest.json", manifest)
    commit_candidate(prepared)
    set_model(
        app,
        "home_construction_equivalence",
        ScriptedModel(
            final=('{"decision":"construction_unchanged","reason":"Query change"}'),
        ),
    )
    force_draw(monkeypatch, 0.1)
    result = await app.run_job("home_evaluate", sampling=True)
    assert result.success, result.answer
    assert result.answer["automatic_construction_reuse"]["source_validation"] == baseline.answer["result_dir"]


@pytest.mark.asyncio
async def test_nearest_source_limits_replay_but_full_rebuild_restores_all_cases(
    app,
    prepared,
    reviewer,
    sandbox,
    monkeypatch,
):
    assert (await app.run_job("home_evaluate")).success
    commit_candidate(prepared)
    recent = await app.run_job("home_evaluate", case_ids=["1"])
    commit_candidate(prepared)
    set_model(
        app,
        "home_construction_equivalence",
        ScriptedModel(
            final=('{"decision":"construction_unchanged","reason":"Query only"}'),
        ),
    )
    force_draw(monkeypatch, 0.1)
    replay = await app.run_job("home_evaluate", sampling=True)
    assert replay.success, replay.answer
    assert replay.answer["automatic_construction_reuse"]["source_validation"] == recent.answer["result_dir"]
    assert replay.answer["case_count"] == 1
    commit_candidate(prepared)
    force_draw(monkeypatch, 0.8)
    rebuilt = await app.run_job("home_evaluate", sampling=True)
    assert rebuilt.success and rebuilt.answer["case_count"] == 2


@pytest.mark.asyncio
async def test_auto_memory_override_and_whitespace_only_changes(app, prepared, reviewer, sandbox, monkeypatch):
    assert (await app.run_job("home_evaluate")).success
    commit_candidate(prepared)
    repo = prepared / "code/repo"
    path = repo / "plugins/beam/src/reme_beam/auto_memory.py"
    path.write_text("value = 1\n")
    command(repo, "add", ".")
    command(repo, "-c", "user.name=Fixture", "-c", "user.email=test@localhost", "commit", "-m", "construction")
    model = ScriptedModel(final='{"decision":"construction_unchanged","reason":"Query only"}')
    set_model(app, "home_construction_equivalence", model)
    result = await app.run_job("home_evaluate", sampling=True)
    assert result.success, result.answer
    assert result.answer["automatic_construction_reuse"]["equivalence_report"]["decision"] == "construction_changed"
    assert not model.calls and not sandbox
    path.write_text("value  =  1\n\n")
    command(repo, "add", ".")
    command(repo, "-c", "user.name=Fixture", "-c", "user.email=test@localhost", "commit", "-m", "spacing")
    force_draw(monkeypatch, 0.1)
    result = await app.run_job("home_evaluate", sampling=True)
    assert result.success and result.answer["execution_mode"] == "query_only_replay"
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_infrastructure_retry_preserves_replay_and_selection(app, prepared, reviewer, sandbox, monkeypatch):
    assert (await app.run_job("home_evaluate")).success
    commit_candidate(prepared)
    model = ScriptedModel(final='{"decision":"construction_unchanged","reason":"Query only"}')
    set_model(app, "home_construction_equivalence", model)
    force_draw(monkeypatch, 0.1)
    monkeypatch.setattr(FakeFactory, "bad_export", True)
    failed = await app.run_job("home_evaluate", sampling=True, max_evaluations=10)
    assert not failed.success
    failed_manifest = read_json(Path(failed.answer["result_dir"]) / "manifest.json")
    assert (await app.run_job("home_create_candidate_branch", branch_name="retry-branch")).success
    monkeypatch.setattr(FakeFactory, "bad_export", False)
    monkeypatch.setattr(reuse, "select_query_only", lambda: pytest.fail("Recovery must not redraw"))
    recovered = await app.run_job("home_evaluate", sampling=True, max_evaluations=10)
    assert recovered.success, recovered.answer
    manifest = read_json(Path(recovered.answer["result_dir"]) / "manifest.json")
    assert manifest["automatic_construction_reuse"] == {
        **failed_manifest["automatic_construction_reuse"],
        "branch_name": "retry-branch",
    }
    assert manifest["case_ids"] == failed_manifest["case_ids"] and manifest["query_ids"] == failed_manifest["query_ids"]
    assert recovered.answer["execution_mode"] == "query_only_replay" and len(model.calls) == 1
    assert validation_budget_status(prepared, 10)["used_validations"] == 2


@pytest.mark.asyncio
async def test_review_checkout_race_and_cancellation_do_not_sample(app, prepared, reviewer, sandbox, monkeypatch):
    assert (await app.run_job("home_evaluate")).success
    commit_candidate(prepared)
    before = (prepared / "sampling/state.json").read_bytes()
    entered, release = asyncio.Event(), asyncio.Event()

    async def paused(*args, **kwargs):
        entered.set()
        await release.wait()
        return {"result": '{"decision":"construction_unchanged","reason":"Query only"}'}

    monkeypatch.setattr(reviewer, "reply", paused)
    task = asyncio.create_task(app.run_job("home_evaluate", sampling=True))
    await asyncio.wait_for(entered.wait(), timeout=10)
    concurrent = await app.run_job("home_evaluate", sampling=True)
    assert not concurrent.success
    commit_candidate(prepared)
    release.set()
    response = await task
    assert not response.success and "changed during equivalence" in str(response.answer)
    assert len(FakeFactory.instances) == 1 and (prepared / "sampling/state.json").read_bytes() == before
    entered.clear()
    release.clear()
    task = asyncio.create_task(app.run_job("home_evaluate", sampling=True))
    await asyncio.wait_for(entered.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not list((prepared / "tmp").glob("construction-equivalence-*"))
    assert (prepared / "sampling/state.json").read_bytes() == before
    release.set()
    assert (await app.run_job("home_evaluate", sampling=True)).success


@pytest.mark.asyncio
async def test_sandbox_failure_stops_before_model_sampling_or_budget(app, prepared, reviewer, monkeypatch):
    assert (await app.run_job("home_evaluate")).success
    commit_candidate(prepared)
    before = (prepared / "sampling/state.json").read_bytes()
    model = ScriptedModel()
    set_model(app, "home_construction_equivalence", model)

    def unavailable(**kwargs):
        raise RuntimeError("sandbox unavailable")

    monkeypatch.setattr(workflow_common, "create_case_study_sandbox", unavailable)
    response = await app.run_job("home_evaluate", sampling=True, max_evaluations=10)
    assert not response.success and "sandbox unavailable" in str(response.answer)
    assert not model.calls and len(FakeFactory.instances) == 1
    assert not list((prepared / "tmp").glob("construction-equivalence-*"))
    assert validation_budget_status(prepared, 10)["used_validations"] == 0
    assert (prepared / "sampling/state.json").read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["candidate-gate", "construction-equivalence"])
@pytest.mark.parametrize("stage", ["export", "report"])
async def test_partial_export_and_report_failures_clean_review_files(
    app,
    prepared,
    reviewer,
    sandbox,
    monkeypatch,
    kind,
    stage,
):
    assert (await app.run_job("home_evaluate")).success
    commit_candidate(prepared)
    if stage == "report":

        def fail_report(*args, **kwargs):
            raise OSError("report write failed")

        monkeypatch.setattr(workflow_common, "save_record", fail_report)
    elif kind == "candidate-gate":
        original = candidate_gate.export_blob

        def fail_blob(*args, **kwargs):
            original(*args, **kwargs)
            raise OSError("partial source export failed")

        monkeypatch.setattr(candidate_gate, "export_blob", fail_blob)
    else:
        original = construction_equivalence.inspect_archive

        def fail_archive(*args, **kwargs):
            original(*args, **kwargs)
            raise OSError("partial construction export failed")

        monkeypatch.setattr(construction_equivalence, "inspect_archive", fail_archive)
    response = await app.run_job(
        "home_evaluate",
        sampling=True,
        **({"max_evaluations": 10} if kind == "candidate-gate" else {}),
    )
    assert not response.success and "failed" in str(response.answer)
    assert not list((prepared / "tmp").glob(f"{kind}-*"))
    assert len(FakeFactory.instances) == 1


@pytest.mark.asyncio
async def test_cancel_during_export_waits_for_writer_and_preserves_other_scratch(tmp_path):
    (tmp_path / "tmp").mkdir()
    unrelated = tmp_path / "tmp/optimizer-keep"
    unrelated.mkdir()
    marker = unrelated / "notes.txt"
    marker.write_text("keep")
    started, release = Event(), Event()
    roots = []

    def prepare(root):
        (root / "partial.txt").write_text("partial")
        started.set()
        if not release.wait(timeout=10):
            raise TimeoutError("test writer was not released")
        (root / "finished.txt").write_text("finished")

    async def review():
        async with candidate_gate.review_export(tmp_path, "construction-equivalence") as root:
            roots.append(root)
            await construction_equivalence.blocking(prepare, root)

    task = asyncio.create_task(review())
    try:
        assert await asyncio.to_thread(started.wait, 10)
        task.cancel()
        await asyncio.sleep(0)
        assert roots[0].is_dir()  # The owned writer is still running.
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not roots[0].exists()
    assert marker.read_text() == "keep"


@pytest.mark.asyncio
async def test_optimizer_final_selection_accepts_automatic_replay(app, prepared, reviewer, sandbox, monkeypatch):
    baseline = await app.run_job("home_evaluate")
    assert baseline.success
    set_model(
        app,
        "home_construction_equivalence",
        ScriptedModel(
            final=('{"decision":"construction_unchanged","reason":"Query only"}'),
        ),
    )
    target = prepared / "code/repo/plugins/beam/src/reme_beam/__init__.py"
    model = ScriptedModel(
        [
            ("home_create_candidate_branch", {"branch_name": "retrieval"}),
            ("Read", {"file_path": str(target)}),
            ("Edit", {"file_path": str(target), "old_string": "value = 1", "new_string": "value = 2"}),
            ("home_commit_candidate", {"message": "Improve retrieval"}),
            ("home_evaluate", {"debug": False}),
        ],
        final=lambda: final_report(prepared, "retrieval"),
    )
    set_model(app, "home_optimizer", model)
    force_draw(monkeypatch, 0.1)
    result = await app.run_job("home_optimize")
    assert result.success, result.answer
    manifest = read_json(Path(result.answer["evaluation_path"]) / "manifest.json")
    assert manifest["execution_mode"] == "query_only_replay"
    schema = next(tool for tool in model.calls[0]["tools"] if tool["function"]["name"] == "home_evaluate")
    assert set(schema["function"]["parameters"]["properties"]) == {"debug"}


@pytest.mark.asyncio
@pytest.mark.usefixtures("os_sandbox_available")
async def test_real_equivalence_sandbox_allows_only_copied_construction_and_source(
    app,
    prepared,
    tmp_path,
    monkeypatch,
):
    source = await app.run_job("home_evaluate")
    assert source.success
    commit_candidate(prepared)
    spec = EvaluationSpec(sampling=True)
    context = reuse.reuse_context(prepared, spec, {**os.environ, **app.context.app_config.environment})
    root = tmp_path / "review-export"
    root.mkdir()
    scope = construction_equivalence.prepare_equivalence_scope(
        prepared,
        context["context"],
        context["spec"],
        context["cases"],
        context["fingerprints"],
        root=root,
    )
    backend = workflow_common.WorkflowSandbox(scope, memory_limit_mib=0)
    root = Path(scope["scratch"])
    evidence = root / "construction/1/workspace/daily/note.md"
    assert await backend.read_file(str(evidence)) == b"1"
    assert await backend.read_file(str(root / "runtime/reme/__init__.py")) is not None
    assert (await backend.stat(str(evidence))).size_bytes == 1
    assert "diff.patch" in await backend.list_dir(str(root))
    private = tmp_path / "private.txt"
    private.write_text("private")
    source_run = Path(source.answer["result_dir"])
    for forbidden in [
        private,
        prepared / "metadata/preparation.json",
        prepared / "code/repo/.git/config",
        prepared / "code/runtime/reme/__init__.py",
        prepared / "datasets/training-set/cases/1.json",
        source_run / "summary.json",
        source_run / "cases/1/queries/result.json",
        source_run / "cases/1/memory_construction/build.log",
    ]:
        with pytest.raises(OSError):
            await backend.read_file(str(forbidden))
        with pytest.raises(OSError):
            await backend.stat(str(forbidden))
    for forbidden in [evidence, root / "diff.patch", root / "new.txt", private]:
        with pytest.raises(OSError):
            await backend.write_file(str(forbidden), b"forbidden")
    (root / "escape").symlink_to(private)
    with pytest.raises(OSError):
        await backend.read_file(str(root / "escape"))
    assert not list(root.rglob(".git")) and not list(root.rglob("queries"))
    network = await backend.exec_shell(
        [sys.executable, "-I", "-B", "-c", "import socket; socket.socket().bind(('127.0.0.1', 0))"],
        timeout=10,
    )
    assert not network.ok()
    monkeypatch.setenv("HOME_REVIEW_PRIVATE_TOKEN", "private-marker")
    environment = await backend.exec_shell(
        [sys.executable, "-I", "-B", "-c", "import os; print(os.environ.get('HOME_REVIEW_PRIVATE_TOKEN', 'absent'))"],
        timeout=10,
    )
    assert environment.ok() and environment.stdout.strip() == b"absent"
