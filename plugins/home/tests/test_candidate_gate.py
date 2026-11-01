"""Admission decisions, provenance, concurrency and real OS sandbox boundaries."""

# pylint: disable=missing-function-docstring,missing-class-docstring,unused-argument,protected-access

import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agentscope.model import ChatUsage, StructuredResponse
from agentscope.tool import LocalBackend
from reme_home.agent import candidate_gate, workflow_common
from reme_home.evaluation.artifacts import read_json
from reme_home.evaluation.budget import validation_budget_status
from reme_home.preparation.code import git

from test_evaluation import (  # noqa: F401  # pylint: disable=unused-import
    FakeFactory,
    commit_candidate,
    home_application,
    preparation_inputs,
    prepared_home,
    rebuild_review,
    os_sandbox_available,
)
from test_workflows import ScriptedModel, set_model


@pytest.fixture(name="sandbox")
def gate_sandbox(monkeypatch):
    policies = []

    def create(**kwargs):
        policies.append(kwargs)
        local = LocalBackend()
        return SimpleNamespace(backend=local, exec_shell=local.exec_shell)

    monkeypatch.setattr(workflow_common, "create_case_study_sandbox", create)
    return policies


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["reject", "accept"])
async def test_gate_before_sampling_preserves_budget_and_pins_code(app, prepared, sandbox, monkeypatch, decision):
    assert (await app.run_job("home_evaluate")).success
    initial = git(prepared / "code/repo", "rev-parse", "HEAD")
    commit_candidate(prepared)
    commit = git(prepared / "code/repo", "rev-parse", "HEAD")
    before = (prepared / "sampling/state.json").read_bytes()

    def finish():
        source = next((prepared / "tmp").glob("candidate-gate-*"))
        relative = "plugins/beam/src/reme_beam/__init__.py"
        assert (source / "base" / relative).read_text() == "value = 1\n"
        assert "Next test candidate" in (source / "candidate" / relative).read_text()
        assert "Next test candidate" in (source / "diff.patch").read_text()
        assert not list(source.rglob(".git"))
        return f'{{"decision":"{decision}","reason":"Inspected the committed mechanism."}}'

    model = ScriptedModel(
        [("Read", lambda: {"file_path": str(next((prepared / "tmp").glob("candidate-gate-*/diff.patch")))})],
        final=finish,
    )
    set_model(app, "home_candidate_gate", model)
    draws = []
    from reme_home.evaluation import component

    original = component.create_sampling_plan

    def draw(*args, **kwargs):
        assert model.calls  # Gate completed before any sampling.
        draws.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(component, "create_sampling_plan", draw)
    response = await app.run_job("home_evaluate", max_validations=10, sampling=True)
    report = response.answer["candidate_gate"]
    assert report["base_commit"] == initial and report["candidate_commit"] == commit
    saved = read_json(Path(report["review_path"]))
    assert saved["decision"] == decision and saved["session_id"]
    assert saved["usage"]["input_tokens"] == 4
    assert len(model.calls) == 2
    assert {tool["function"]["name"] for tool in model.calls[0]["tools"]} == {"Read", "Glob", "Grep", "AutoBash"}
    assert sandbox and all(policy["writable_paths"] == [] for policy in sandbox)
    gate_policies = [policy for policy in sandbox if "candidate-gate-" in policy["cwd"]]
    assert gate_policies and all(policy["read_only_paths"] == [report["source_path"]] for policy in gate_policies)
    assert not Path(report["source_path"]).exists()
    if decision == "reject":
        assert not response.success
        assert response.answer["error_code"] == "CANDIDATE_GATE_REJECTED"
        assert response.answer["result_dir"] is None
        assert len(FakeFactory.instances) == 1 and not draws
        assert (prepared / "sampling/state.json").read_bytes() == before
        assert validation_budget_status(prepared, 10)["used_validations"] == 0
    else:
        assert response.success, response.answer
        assert len(FakeFactory.instances) == 2 and draws
        manifest = read_json(Path(response.answer["result_dir"]) / "manifest.json")
        assert manifest["candidate_gate"] == report
        assert validation_budget_status(prepared, 10)["used_validations"] == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures("rebuild_review")
async def test_baseline_debug_unbudgeted_and_late_phase_skip_gate(app, prepared, monkeypatch):
    async def unexpected(*args, **kwargs):
        raise AssertionError("Gate must be skipped")

    monkeypatch.setattr(candidate_gate, "review_candidate", unexpected)
    assert (await app.run_job("home_evaluate", max_validations=10)).success  # Baseline.
    commit_candidate(prepared)
    assert (await app.run_job("home_evaluate", debug=True, max_validations=10)).success
    assert (await app.run_job("home_evaluate")).success  # Ordinary explicit evaluations remain ungated.
    commit_candidate(prepared)
    result = await app.run_job("home_evaluate", max_validations=2)  # One previous non-init start: second half.
    assert result.success, result.answer
    assert "candidate_gate" not in result.answer
    commit_candidate(prepared)
    result = await app.run_job("home_evaluate", max_validations=2)
    assert result.answer["error_code"] == "VALIDATION_BUDGET_EXHAUSTED"


@pytest.mark.asyncio
async def test_nearest_ancestor_and_infrastructure_retry(app, prepared, sandbox, monkeypatch):
    assert (await app.run_job("home_evaluate")).success
    commit_candidate(prepared)
    first = await app.run_job("home_evaluate")
    assert first.success
    base = first.answer["commit_sha"]
    commit_candidate(prepared)
    set_model(app, "home_candidate_gate", ScriptedModel(final='{"decision":"accept","reason":"New mechanism"}'))
    monkeypatch.setattr(FakeFactory, "bad_export", True)
    failed = await app.run_job("home_evaluate", max_validations=10, sampling=True)
    assert not failed.success
    assert failed.answer["candidate_gate"]["base_commit"] == base

    async def unexpected(*args, **kwargs):
        raise AssertionError("Infrastructure retry must not rerun admission")

    monkeypatch.setattr(candidate_gate, "review_candidate", unexpected)
    monkeypatch.setattr(FakeFactory, "bad_export", False)
    recovered = await app.run_job("home_evaluate", max_validations=10, sampling=True)
    assert recovered.success, recovered.answer
    repeat = await app.run_job("home_evaluate", max_validations=10)
    assert repeat.answer["error_code"] == "UNCHANGED_CANDIDATE"


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ["invalid", '{"decision":"reject","reason":"x","extra":true}', "error"])
async def test_model_failure_allows_validation_with_saved_diagnostics(app, prepared, sandbox, monkeypatch, output):
    commit_candidate(prepared)
    wrapper = set_model(app, "home_candidate_gate", ScriptedModel(final=output))
    if output == "error":

        async def broken(*args, **kwargs):
            raise RuntimeError("model unavailable")

        monkeypatch.setattr(wrapper, "reply", broken)
    response = await app.run_job("home_evaluate", max_validations=10)
    assert response.success, response.answer
    report = response.answer["candidate_gate"]
    assert report["decision"] == "accept" and report["review_failed"]
    assert read_json(Path(report["review_path"]))["review_failed"]
    assert not list((prepared / "tmp").glob("candidate-gate-*"))


@pytest.mark.asyncio
async def test_corrected_gate_rejects_before_spending_budget(app, prepared, sandbox):
    commit_candidate(prepared)
    model = ScriptedModel(final="Reject this candidate: only prompt tuning.")
    model.generate_structured_output = AsyncMock(
        return_value=StructuredResponse(
            content={"decision": "reject", "reason": "Only prompt tuning"},
            usage=ChatUsage(input_tokens=7, output_tokens=11, time=0),
        ),
    )
    set_model(app, "home_candidate_gate", model)
    response = await app.run_job("home_evaluate", max_validations=10)
    assert not response.success
    assert response.answer["error_code"] == "CANDIDATE_GATE_REJECTED"
    report = read_json(Path(response.answer["candidate_gate"]["review_path"]))
    assert report["decision"] == "reject" and "review_failed" not in report
    assert report["usage"]["input_tokens"] == 9 and report["usage"]["output_tokens"] == 14
    assert report["session_id"] and not Path(report["source_path"]).exists()
    assert validation_budget_status(prepared, 10)["used_validations"] == 0
    assert not FakeFactory.instances
    model.generate_structured_output.assert_awaited_once()


@pytest.mark.asyncio
async def test_sandbox_setup_failure_never_calls_model_or_evaluates(app, prepared, monkeypatch):
    commit_candidate(prepared)
    model = ScriptedModel()
    set_model(app, "home_candidate_gate", model)

    def unavailable(**kwargs):
        raise RuntimeError("sandbox unavailable")

    monkeypatch.setattr(workflow_common, "create_case_study_sandbox", unavailable)
    result = await app.run_job("home_evaluate", max_validations=10)
    assert not result.success and "sandbox unavailable" in str(result.answer)
    assert not model.calls and not FakeFactory.instances
    assert validation_budget_status(prepared, 10)["used_validations"] == 0
    assert not list((prepared / "tmp").glob("candidate-gate-*"))


@pytest.mark.asyncio
async def test_review_change_race_and_cancellation_release_locks(app, prepared, sandbox, monkeypatch):
    commit_candidate(prepared)
    entered, release = asyncio.Event(), asyncio.Event()
    wrapper = app.context.components["agent_wrapper"]["home_candidate_gate"]

    async def paused(*args, **kwargs):
        entered.set()
        await release.wait()
        return {"result": '{"decision":"accept","reason":"Mechanism change"}'}

    monkeypatch.setattr(wrapper, "reply", paused)
    task = asyncio.create_task(app.run_job("home_evaluate", max_validations=10))
    await asyncio.wait_for(entered.wait(), timeout=10)
    concurrent = await app.run_job("home_evaluate", max_validations=10)
    assert not concurrent.success  # Formal lock spans review and evaluation.
    commit_candidate(prepared)
    release.set()
    result = await task
    assert not result.success and "changed during admission" in str(result.answer)
    assert not FakeFactory.instances
    entered.clear()
    release.clear()
    task = asyncio.create_task(app.run_job("home_evaluate", max_validations=10))
    await asyncio.wait_for(entered.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not list((prepared / "tmp").glob("candidate-gate-*"))
    assert validation_budget_status(prepared, 10)["used_validations"] == 0
    release.set()
    result = await app.run_job("home_evaluate", max_validations=10)
    assert result.success, result.answer


@pytest.mark.asyncio
@pytest.mark.usefixtures("os_sandbox_available")
async def test_real_gate_sandbox_denies_evidence_live_code_writes_and_network(app, prepared, tmp_path):
    commit_candidate(prepared)
    context = candidate_gate.admission_context(prepared)
    root = tmp_path / "review-export"
    root.mkdir()
    scope = candidate_gate.prepare_review_scope(prepared, context, root=root)
    backend = workflow_common.WorkflowSandbox(scope, memory_limit_mib=0)
    root = Path(scope["scratch"])
    target = root / "candidate/plugins/beam/src/reme_beam/__init__.py"
    assert b"Next test candidate" in await backend.read_file(str(target))
    assert (await backend.stat(str(target))).size_bytes > 0
    assert "diff.patch" in await backend.list_dir(str(root))
    outside = tmp_path / "private.txt"
    outside.write_text("private")
    for path in [
        outside,
        prepared / "metadata/preparation.json",
        prepared / "code/repo/.git/config",
        prepared / "code/repo/plugins/beam/src/reme_beam/__init__.py",
        prepared / "datasets/training-set/cases/1.json",
    ]:
        with pytest.raises(OSError):
            await backend.read_file(str(path))
        with pytest.raises(OSError):
            await backend.stat(str(path))
    for path in (target, root / "diff.patch", root / "new.txt", outside):
        with pytest.raises(OSError):
            await backend.write_file(str(path), b"forbidden")
    # Symlinks inside allowed exports must not bypass OS containment.
    (root / "escape").symlink_to(outside)
    with pytest.raises(OSError):
        await backend.read_file(str(root / "escape"))
    network = await backend.exec_shell(
        [sys.executable, "-I", "-B", "-c", "import socket; socket.socket().bind(('127.0.0.1', 0))"],
        timeout=10,
    )
    assert not network.ok()
