"""Formal budget accounting through real evaluation Jobs and fake Docker boundaries."""

# pylint: disable=missing-function-docstring,missing-class-docstring,unused-argument,protected-access

import asyncio

import pytest
from reme_home.agent.optimizer import OptimizerAgentWrapper, validation_budget_tool
from reme_home.evaluation import component as evaluation
from reme_home.evaluation.artifacts import write_json
from reme_home.evaluation.budget import validation_budget_status
from reme_home.evaluation.inputs import EvaluationSpec
from test_evaluation import (  # noqa: F401  # pylint: disable=unused-import
    FakeFactory,
    commit_candidate,
    home_application,
    preparation_inputs,
    prepared_home,
)


def seed_history(home, keys):
    write_json(
        home / "sampling/state.json",
        {
            "schema_version": 2,
            "round": len(keys),
            "successful_round": 0,
            "cases": {},
            "queries": {},
            "started_validations": {key: index for index, key in enumerate(keys, 1)},
            "processed_validations": [],
        },
    )


@pytest.mark.parametrize(
    "used,phase,remaining",
    [
        (0, "structural_exploration", 10),
        (4, "structural_exploration", 6),
        (5, "evidence_driven_refinement", 5),
        (9, "evidence_driven_refinement", 1),
        (10, "exhausted", 0),
        (11, "exhausted", 0),
    ],
)
def test_dev_meta_phase_boundaries(tmp_path, used, phase, remaining):
    seed_history(tmp_path, ["init/first/1", *(f"candidate/hash/{index}" for index in range(used))])
    result = validation_budget_status(tmp_path, 10)
    assert result["used_validations"] == used
    assert result["remaining_validations"] == remaining
    assert result["optimization_phase"] == phase
    assert result["gated_validation_count"] == 5
    assert result["next_validation_guidance"]


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.5, "10", None])
def test_invalid_optimizer_budgets_rejected(invalid):
    wrapper = OptimizerAgentWrapper(name="optimizer", max_validations=invalid)
    with pytest.raises(ValueError, match="positive integer"):
        wrapper.validation_limit()
    if invalid is not None:
        with pytest.raises(ValueError):
            EvaluationSpec(max_validations=invalid)


def test_budgeted_evaluation_cannot_disable_started_history():
    with pytest.raises(ValueError, match="record_history"):
        EvaluationSpec(max_validations=10, record_history=False)
    assert EvaluationSpec(debug=True, max_validations=1, record_history=False).debug
    assert OptimizerAgentWrapper(name="optimizer").validation_limit() == 10


@pytest.mark.asyncio
async def test_budget_tool_reports_persisted_count_without_mutation(tmp_path):
    seed_history(tmp_path, ["init/base/1", "candidate/hash/1"])
    before = (tmp_path / "sampling/state.json").read_bytes()
    tool = validation_budget_tool(tmp_path, 3)
    assert tool.name == "get_validation_budget"
    result = await tool._func()
    assert result["remaining_validations"] == 2
    assert result["optimization_phase"] == "evidence_driven_refinement"
    assert (tmp_path / "sampling/state.json").read_bytes() == before


@pytest.mark.asyncio
async def test_exhaustion_blocks_before_pool_and_survives_cache_rebuild(app, prepared):
    baseline = await app.run_job("home_evaluate")
    assert baseline.success
    assert validation_budget_status(prepared, 1)["used_validations"] == 0
    commit_candidate(prepared)
    result = await app.run_job("home_evaluate", max_validations=1)
    assert result.success, result.answer
    assert result.answer["validation_budget"]["remaining_validations"] == 0
    count = len(FakeFactory.instances)
    # A fresh status lookup and cache rebuild both retain the charge.
    (prepared / "sampling/state.json").unlink()
    assert validation_budget_status(prepared, 1)["used_validations"] == 1
    commit_candidate(prepared)
    rejected = await app.run_job("home_evaluate", max_validations=1)
    assert not rejected.success
    assert rejected.answer["error_code"] == "VALIDATION_BUDGET_EXHAUSTED"
    assert rejected.answer["result_dir"] is None
    assert len(FakeFactory.instances) == count
    debug = await app.run_job("home_evaluate", debug=True, max_validations=1)
    assert debug.success, debug.answer
    assert "validation_budget" not in debug.answer
    assert validation_budget_status(prepared, 1)["used_validations"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["candidate_failure", "infra_failure", "cancelled", "prestart_failure"])
async def test_charge_depends_on_first_sandbox_start_not_result(app, prepared, monkeypatch, outcome):
    commit_candidate(prepared)
    if outcome == "candidate_failure":
        monkeypatch.setattr(FakeFactory, "fail_build", "1")
    elif outcome == "infra_failure":
        monkeypatch.setattr(FakeFactory, "bad_export", True)
    elif outcome == "cancelled":
        monkeypatch.setattr(FakeFactory, "block", asyncio.Event())
    else:

        class PrestartFailureFactory(FakeFactory):
            async def create_cases(self, ids, **kwargs):
                raise RuntimeError("sandbox initialization failed")

        monkeypatch.setattr(evaluation, "DockerReMeSandboxFactory", PrestartFailureFactory)
    task = asyncio.create_task(app.run_job("home_evaluate", max_validations=1))
    if outcome == "cancelled":
        while not FakeFactory.instances:
            await asyncio.sleep(0)
        await FakeFactory.instances[-1].entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await task
    expected = int(outcome != "prestart_failure")
    assert validation_budget_status(prepared, 1)["used_validations"] == expected
    (prepared / "sampling/state.json").unlink(missing_ok=True)
    assert validation_budget_status(prepared, 1)["used_validations"] == expected
