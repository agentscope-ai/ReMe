"""Admission decisions, provenance, concurrency and real OS sandbox boundaries."""

# pylint: disable=missing-function-docstring,missing-class-docstring,unused-argument,protected-access

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from agentscope.model import ChatUsage, StructuredResponse
from reme_home.agent import candidate_gate, workflow_common
from reme_home.evaluation import component
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
    host_network_probe_fixture,
)
from test_workflows import ScriptedModel, set_model

from reme.utils.local_backend import LocalProcessBackend


@pytest.fixture(name="sandbox")
def gate_sandbox(monkeypatch):
    policies = []

    def create(**kwargs):
        policies.append(kwargs)
        local = LocalProcessBackend()
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
    original = component.create_sampling_plan

    def draw(*args, **kwargs):
        assert model.calls  # Gate completed before any sampling.
        draws.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(component, "create_sampling_plan", draw)
    response = await app.run_job("home_evaluate", max_evaluations=10, sampling=True)
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
    assert (await app.run_job("home_evaluate", max_evaluations=10)).success  # Baseline.
    baseline_repeat = await app.run_job("home_evaluate", max_evaluations=10)
    assert baseline_repeat.answer["error_code"] == "UNCHANGED_CANDIDATE"
    commit_candidate(prepared)
    assert (await app.run_job("home_evaluate", debug=True, max_evaluations=10)).success
    assert (await app.run_job("home_evaluate")).success  # Ordinary explicit evaluations remain ungated.
    commit_candidate(prepared)
    result = await app.run_job("home_evaluate", max_evaluations=2)  # One previous non-init start: second half.
    assert result.success, result.answer
    assert "candidate_gate" not in result.answer
    commit_candidate(prepared)
    result = await app.run_job("home_evaluate", max_evaluations=2)
    assert result.answer["error_code"] == "VALIDATION_BUDGET_EXHAUSTED"


@pytest.mark.asyncio
@pytest.mark.parametrize("exemption", ["baseline", "retry", "retry_same_commit"])
async def test_gate_exemption_is_bound_to_checkout(app, prepared, monkeypatch, exemption):
    commit_candidate(prepared)
    if exemption == "baseline":
        origin, destination = "init", "test-candidate"
    else:
        monkeypatch.setattr(FakeFactory, "bad_export", True)
        failed = await app.run_job("home_evaluate")
        assert not failed.success and failed.answer["result_dir"]
        monkeypatch.setattr(FakeFactory, "bad_export", False)
        origin, destination = "test-candidate", "next-candidate"
        assert (await app.run_job("home_create_candidate_branch", branch_name=destination)).success
        if exemption == "retry":
            commit_candidate(prepared)
    assert (await app.run_job("home_switch_candidate_branch", branch_name=origin)).success

    evaluation = app.context.components["home.evaluation"]["default"]
    evaluate = evaluation._evaluate
    gate = AsyncMock(side_effect=AssertionError("Gate must be skipped for the original checkout"))
    monkeypatch.setattr(candidate_gate, "review_candidate", gate)
    launches = len(FakeFactory.instances)
    budget = validation_budget_status(prepared, 10)
    attempts = set((prepared / "evaluations").glob("*/*/*/manifest.json"))

    async def switch_after_gate(*args):
        assert (await app.run_job("home_switch_candidate_branch", branch_name=destination)).success
        return await evaluate(*args)

    monkeypatch.setattr(evaluation, "_evaluate", switch_after_gate)
    result = await app.run_job("home_evaluate", max_evaluations=10)
    assert not result.success and "changed during admission" in str(result.answer)
    gate.assert_not_awaited()
    assert len(FakeFactory.instances) == launches
    assert validation_budget_status(prepared, 10) == budget
    assert set((prepared / "evaluations").glob("*/*/*/manifest.json")) == attempts


@pytest.mark.asyncio
@pytest.mark.usefixtures("rebuild_review")
@pytest.mark.parametrize("intervening", ["debug_only", "debug", "unrecorded", "infra_error"])
async def test_admission_base_requires_usable_budgeted_ancestor(app, prepared, monkeypatch, intervening):
    repo = prepared / "code/repo"
    base = git(repo, "rev-parse", "HEAD")
    used = 0
    if intervening != "debug_only":
        # A candidate failure is still meaningful formal evidence, even at score zero.
        commit_candidate(prepared)
        with monkeypatch.context() as patch:
            patch.setattr(FakeFactory, "fail_build", "1")
            formal = await app.run_job("home_evaluate", case_ids=["1"])
        assert formal.success and formal.answer["case_status_counts"]["candidate_failure"] == 1
        assert formal.answer["mean_query_score"] == 0
        base = formal.answer["commit_sha"]
        used = 1

    commit_candidate(prepared)
    if intervening == "infra_error":
        with monkeypatch.context() as patch:
            patch.setattr(FakeFactory, "bad_export", True)
            result = await app.run_job("home_evaluate")
        assert not result.success and result.answer["case_status_counts"]["infra_error"] == 2
        used += 1
    elif intervening == "unrecorded":
        assert (await app.run_job("home_evaluate", record_history=False)).success
    else:
        assert (await app.run_job("home_evaluate", debug=True)).success

    assert validation_budget_status(prepared, 10)["used_validations"] == used
    commit_candidate(prepared)

    async def reject(wrapper, home, context, budget, **kwargs):
        return {**context, "decision": "reject", "reason": "Stop after capturing the admission context"}

    monkeypatch.setattr(candidate_gate, "review_candidate", reject)
    response = await app.run_job("home_evaluate", max_evaluations=10)
    assert response.answer["error_code"] == "CANDIDATE_GATE_REJECTED"
    assert response.answer["candidate_gate"]["base_commit"] == base


@pytest.mark.asyncio
@pytest.mark.usefixtures("rebuild_review")
async def test_admission_base_uses_latest_evaluation_at_equal_ancestry_distance(app, prepared):
    repo = prepared / "code/repo"
    # The left lineage is longer, but both evaluated tips are direct parents of the merge.
    commit_candidate(prepared)
    commit_candidate(prepared)
    assert (await app.run_job("home_evaluate")).success
    assert (await app.run_job("home_switch_candidate_branch", branch_name="init")).success
    assert (await app.run_job("home_create_candidate_branch", branch_name="right")).success
    (repo / "plugins/beam/src/right.py").write_text("right = True\n", encoding="utf-8")
    assert (await app.run_job("home_commit_candidate", message="Right change")).success
    right = await app.run_job("home_evaluate")
    assert right.success
    assert (await app.run_job("home_create_candidate_branch", branch_name="unmerged")).success
    commit_candidate(prepared)
    assert (await app.run_job("home_evaluate")).success
    assert (await app.run_job("home_switch_candidate_branch", branch_name="right")).success
    assert (await app.run_job("home_merge_candidate_branch", branch_name="test-candidate")).success
    assert (await app.run_job("home_commit_candidate", message="Combine branches")).success

    _, context = candidate_gate.admission_context(prepared)
    assert context["base_commit"] == right.answer["commit_sha"]


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
    failed = await app.run_job("home_evaluate", max_evaluations=10, sampling=True)
    assert not failed.success
    assert failed.answer["candidate_gate"]["base_commit"] == base

    async def unexpected(*args, **kwargs):
        raise AssertionError("Infrastructure retry must not rerun admission")

    monkeypatch.setattr(candidate_gate, "review_candidate", unexpected)
    monkeypatch.setattr(FakeFactory, "bad_export", False)
    lookup = Mock(wraps=candidate_gate.previous_candidate_result)
    monkeypatch.setattr(candidate_gate, "previous_candidate_result", lookup)
    monkeypatch.setattr(component, "previous_candidate_result", lookup)
    recovered = await app.run_job("home_evaluate", max_evaluations=10, sampling=True)
    assert recovered.success, recovered.answer
    lookup.assert_called_once_with(prepared, recovered.answer["commit_sha"])
    repeat = await app.run_job("home_evaluate", max_evaluations=10)
    assert repeat.answer["error_code"] == "UNCHANGED_CANDIDATE"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output",
    ["invalid", '{"decision":"reject","reason":"x","extra":true}', "error", "failed", "interrupted", "waiting"],
)
async def test_model_failure_allows_validation_with_saved_diagnostics(app, prepared, sandbox, monkeypatch, output):
    commit_candidate(prepared)
    wrapper = set_model(app, "home_candidate_gate", ScriptedModel(final=output))
    if output == "error":

        async def broken(*args, **kwargs):
            raise RuntimeError("model unavailable")

        monkeypatch.setattr(wrapper, "reply", broken)
    elif output in {"failed", "interrupted", "waiting"}:
        monkeypatch.setattr(
            wrapper,
            "reply",
            AsyncMock(
                return_value={
                    "result": '{"decision":"reject","reason":"Unfinished judgement"}',
                    "status": output,
                    "error": None,
                },
            ),
        )
    response = await app.run_job("home_evaluate", max_evaluations=10)
    assert response.success, response.answer
    report = response.answer["candidate_gate"]
    assert report["decision"] == "accept" and report["review_failed"]
    if output in {"failed", "interrupted", "waiting"}:
        assert output in report["reason"]
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
    response = await app.run_job("home_evaluate", max_evaluations=10)
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
    result = await app.run_job("home_evaluate", max_evaluations=10)
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
        return {"result": '{"decision":"accept","reason":"Mechanism change"}', "status": "completed", "error": None}

    monkeypatch.setattr(wrapper, "reply", paused)
    task = asyncio.create_task(app.run_job("home_evaluate", max_evaluations=10))
    await asyncio.wait_for(entered.wait(), timeout=10)
    concurrent = await app.run_job("home_evaluate", max_evaluations=10)
    assert not concurrent.success  # Formal lock spans review and evaluation.
    commit_candidate(prepared)
    release.set()
    result = await task
    assert not result.success and "changed during admission" in str(result.answer)
    assert not FakeFactory.instances
    entered.clear()
    release.clear()
    task = asyncio.create_task(app.run_job("home_evaluate", max_evaluations=10))
    await asyncio.wait_for(entered.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not list((prepared / "tmp").glob("candidate-gate-*"))
    assert validation_budget_status(prepared, 10)["used_validations"] == 0
    release.set()
    result = await app.run_job("home_evaluate", max_evaluations=10)
    assert result.success, result.answer


@pytest.mark.asyncio
@pytest.mark.usefixtures("os_sandbox_available")
async def test_real_gate_sandbox_denies_evidence_live_code_writes_and_network(
    app,
    prepared,
    tmp_path,
    host_network_probe,
):
    commit_candidate(prepared)
    _, context = candidate_gate.admission_context(prepared)
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
    # Binding inside an isolated network namespace may succeed. Connecting to
    # a live host listener tests the boundary the sandbox actually promises.
    network = await backend.exec_shell(host_network_probe, timeout=10)
    assert not network.ok()
