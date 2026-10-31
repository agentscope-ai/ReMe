"""Optimize Home candidates and verify their baseline and final evaluation evidence."""

import asyncio
import json
import math
import time
from pathlib import Path
from typing import Callable

from agentscope.message import HintBlock
from agentscope.state import AgentState
from agentscope.tool import Edit, FunctionTool
from pydantic import Field

from reme.steps.base_step import BaseStep, Ref

from ..code.repository import validate_repository, require_clean, validate_tree, status
from ..evaluation.artifacts import contained, read_json, write_json
from ..evaluation.component import blocking
from ..evaluation.inputs import EvaluationSpec, load_inputs
from ..evaluation.history import read_state
from ..evaluation.budget import validate_max_validations, validation_budget_status
from ..preparation.spec import SpecModel
from ..workspace import HomeWorkspace, _initialization_lock, _validate_directory
from .diagnostic import diagnostic_tool, evaluation_paths
from .bounded import TimeBudgetReminder, _format_minutes, _remove_hint_by_identity
from .compression import RecoverableCompressionAgent
from .workflow_common import required_job, save_record, workflow_reply
from .wrapper import CaseStudyAgentWrapper

MUTATION_JOBS = (
    "home_create_candidate_branch",
    "home_switch_candidate_branch",
    "home_merge_candidate_branch",
    "home_commit_candidate",
)
PENDING_CYCLE_TRANSITIONS = "home.optimizer.pending_cycle_transitions"


class ValidationCycleTimer:
    """Emit escalating reminders as a scored-validation cycle ages."""

    def __init__(self, time_budget_minutes: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        if (
            not isinstance(time_budget_minutes, (int, float))
            or isinstance(time_budget_minutes, bool)
            or not math.isfinite(time_budget_minutes)
            or time_budget_minutes <= 0
        ):
            raise ValueError("time_budget_minutes must be a positive number")
        self.time_budget_minutes = float(time_budget_minutes)
        self._clock = clock
        self._started_at = self._clock()
        self._last_reminded_ratio = 0.0
        self._next_reminder_is_persistent = False
        self.context_compression_count = 0

    def reset(self) -> None:
        """Start a fresh cycle after a scored validation finishes."""

        self._started_at = self._clock()
        self._last_reminded_ratio = 0.0
        self._next_reminder_is_persistent = False
        self.context_compression_count = 0

    def record_context_compression(self) -> None:
        """Record one completed compression since the last scored validation."""

        self.context_compression_count += 1

    def next_reminder(self) -> TimeBudgetReminder | None:
        """Return one reminder for the latest newly reached deadline."""

        elapsed_minutes = max(0.0, (self._clock() - self._started_at) / 60.0)
        elapsed_ratio = elapsed_minutes / self.time_budget_minutes
        reached_ratio: float | None = None
        if elapsed_ratio >= 1.0:
            reached_ratio = math.floor(elapsed_ratio * 10 + 1e-9) / 10
        else:
            for threshold in (0.9, 0.75, 0.5):
                if elapsed_ratio >= threshold:
                    reached_ratio = threshold
                    break
        if reached_ratio is None or reached_ratio <= self._last_reminded_ratio:
            return None
        self._last_reminded_ratio = reached_ratio
        persistent = self._next_reminder_is_persistent
        self._next_reminder_is_persistent = not persistent
        elapsed = _format_minutes(elapsed_minutes)
        percentage = round(reached_ratio * 100)
        timing = (
            f"The optimizer agent run has used {elapsed} since the last scored validation "
            f"({percentage}% of its {_format_minutes(self.time_budget_minutes)} soft time budget). "
        )
        compression = (
            f"It has completed {self.context_compression_count} context compressions since the last "
            "scored validation. "
        )
        if elapsed_ratio >= 1.0:
            return TimeBudgetReminder(
                content=(
                    f"<system-reminder>{timing}{compression}"
                    "Prioritize completing the code changes, committing them, passing the smoke test, and starting "
                    "validation as soon as possible.</system-reminder>"
                ),
                persistent=persistent,
            )
        remaining = _format_minutes(self.time_budget_minutes - elapsed_minutes)
        return TimeBudgetReminder(
            content=(
                f"<system-reminder>{timing}{compression}"
                f"Complete the code changes, commit them, pass the smoke test, and start validation within {remaining}."
                "</system-reminder>"
            ),
            persistent=persistent,
        )


class OptimizerAgent(RecoverableCompressionAgent):
    """Add Meta-ReMe's per-validation reminders to Home's bounded agent."""

    def __init__(self, *args, validation_cycle_timer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_cycle_timer = validation_cycle_timer

    async def _reasoning(self, *args, **kwargs):
        """Deliver hints after tool results without splitting the SDK's reply message."""
        transitions = self.state.middle_context.pop(PENDING_CYCLE_TRANSITIONS, [])
        for content in transitions:
            self.state.append_context(
                self.name,
                [HintBlock(source="meta_reme_validation_cycle_transition", hint=content)],
            )
        timer = self.validation_cycle_timer
        reminder = timer.next_reminder() if timer is not None else None
        message = None
        if reminder is not None:
            message = HintBlock(source="meta_reme_validation_timer", hint=reminder.content)
            self.state.append_context(self.name, [message])
        try:
            async for item in super()._reasoning(*args, **kwargs):
                yield item
        finally:
            if reminder is not None and not reminder.persistent:
                _remove_hint_by_identity(self.state.context, message)

    async def compress_context(self, *args, **kwargs):
        """Count completed context replacements in the current validation cycle."""
        summary_before, context_before = self.state.summary, self.state.context
        await super().compress_context(*args, **kwargs)
        if self.validation_cycle_timer is not None and (
            self.state.summary is not summary_before or self.state.context is not context_before
        ):
            self.validation_cycle_timer.record_context_compression()


class OptimizerAgentWrapper(CaseStudyAgentWrapper):
    """Keep each optimizer's tools and cycle timer bound to one reply invocation."""

    def create_cycle_timer(self, options=None):
        """Use the legacy time setting as a fallback; explicit null disables reminders."""
        options = self._merged_kwargs(options or {})
        budget = options.get("validation_cycle_time_budget_minutes", options.get("time_budget_minutes"))
        return None if budget is None else ValidationCycleTimer(budget)

    def validation_limit(self):
        """Keep the formal count limit host-owned and independent of elapsed time."""
        return validate_max_validations(self._merged_kwargs({}).get("max_validations", 10))

    def _create_agent(self, *, options, **kwargs):
        timer = (
            options["_validation_cycle_timer"]
            if "_validation_cycle_timer" in options
            else self.create_cycle_timer(options)
        )
        return OptimizerAgent(
            validation_cycle_timer=timer,
            fallback_directory=options.get("_context_fallback_directory", self.session_path / "context-offloads"),
            **kwargs,
        )


def validation_cycle_transition(manifest, validation_number, remaining):
    """Expire old reminders and announce the next cycle with its remaining budget."""
    commit = manifest["commit_sha"][:7]
    return HintBlock(
        source="meta_reme_validation_cycle_transition",
        hint=(
            f"<system-reminder>Formal validation {validation_number} completed successfully for commit {commit}. "
            f"The previous cycle timer has been reset, and all timer reminders for cycle {validation_number} "
            f"are now expired. Remaining formal validations: {remaining}. Start cycle {validation_number + 1}."
            "</system-reminder>"
        ),
    )


def completed_validation(home, result, started_before):
    """Match dev_meta's new-count plus completed-status boundary, independent of scores."""
    if not isinstance(result, dict) or result.get("status") not in {"success", "completed"}:
        return None
    run = contained(home, result["result_dir"], directory=True)
    key = run.relative_to(contained(home, "evaluations", directory=True)).as_posix()
    started = read_state(home)["started_validations"]
    # The initial full baseline precedes optimizer creation in dev_meta and
    # never consumes its validation budget. Home records it in sampling history.
    if key in started_before or key not in started or key.startswith("init/"):
        return None
    manifest = read_json(contained(run, "manifest.json"))
    if (
        manifest.get("status") != "completed"
        or manifest.get("mode") != "scored"
        or manifest.get("dataset_kind") != "training-set"
        or not manifest.get("sandbox_started_at")
    ):
        return None
    return manifest, sum(not name.startswith("init/") for name in started)


class OptimizationInput(SpecModel):
    """The host owns runtime settings; the caller supplies only the optimization focus."""

    instruction: str = Field(default="Improve the candidate using the completed baseline evaluation.", min_length=1)


class OptimizationResult(SpecModel):
    """An agent claim that must be checked against Git and saved formal evidence."""

    branch_name: str = Field(min_length=1)
    commit_sha: str = Field(pattern="^[0-9a-f]{40}$")
    evaluation_path: str = Field(min_length=1)
    report: str = Field(min_length=1)


def parse_result(text):
    """Accept a JSON object, optionally enclosed in one Markdown JSON fence."""
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return OptimizationResult.model_validate(json.loads(text))


EVALUATION_OPTIONS = {
    "concurrency",
    "timeout_seconds",
    "base_image",
    "fail_fast",
    "max_construction_failure_rate",
    "max_artifact_bytes",
}


def evaluation_options(raw):
    """Validate host-only execution settings without exposing selection or replay."""
    if not isinstance(raw, dict) or set(raw) - EVALUATION_OPTIONS:
        raise ValueError("evaluation_options may contain only Home execution settings")
    return EvaluationSpec.model_validate(raw).model_dump(exclude_unset=True)


def verify_evaluation(home, path, commit, *, baseline=False):
    """Require completed end-to-end evidence with no infrastructure failures."""
    run = Path(evaluation_paths(home, [str(path)])[0])
    manifest = read_json(contained(run, "manifest.json"))
    summary = read_json(contained(run, "summary.json"))
    required = {
        "commit_sha": commit,
        "status": "completed",
        "success": True,
        "mode": "scored",
        "dataset_kind": "training-set",
        "execution_mode": "full_validation",
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("Evaluation must match the commit and complete without infrastructure failures")
    if (
        not manifest.get("sandbox_started_at")
        or not summary.get("case_count")
        or summary.get("scoring_enabled") is not True
        or any(summary.get(key, {}).get("infra_error") != 0 for key in ("case_status_counts", "query_status_counts"))
    ):
        raise ValueError("Evaluation must match the commit and complete without infrastructure failures")
    _record, dataset, cases = load_inputs(home, EvaluationSpec())
    if manifest.get("fingerprints", {}).get("dataset") != dataset.normalized_fingerprint:
        raise ValueError("Evaluation dataset no longer matches the prepared dataset")
    selected = manifest.get("case_ids", [])
    queries = manifest.get("query_ids", {})
    expected = {case.case_id: {q.query_id for q in case.queries} for case in cases}
    if (
        not selected
        or len(set(selected)) != len(selected)
        or set(selected) - expected.keys()
        or set(queries) != set(selected)
        or any(not ids or len(set(ids)) != len(ids) or set(ids) - expected[key] for key, ids in queries.items())
    ):
        raise ValueError("Evaluation has an invalid case/query selection")
    if baseline and (set(selected) != set(expected) or any(set(queries[key]) != ids for key, ids in expected.items())):
        raise ValueError("Initial baseline must cover every prepared training case and query")
    rows = summary.get("cases", {})
    if (
        set(rows) != set(selected)
        or summary["case_count"] != len(selected)
        or summary.get("expected_query_count") != sum(len(ids) for ids in queries.values())
    ):
        raise ValueError("Evaluation summary has incomplete case coverage")
    # Failed construction can legitimately omit answers; the recorded candidate
    # failure is evidence, not an infrastructure failure or a missing result.
    for key, row in rows.items():
        if set(row.get("queries", {})) != set(queries[key]) and row.get("construction", {}).get("status") != "failure":
            raise ValueError(f"Evaluation summary has incomplete query coverage: {key}")
    return {"evaluation_path": str(run), "manifest": manifest, "summary": summary}


def baseline_state(home):
    """Inspect a clean repository and locate a usable initial full evaluation."""
    _validate_directory(home, home / "metadata")
    with _initialization_lock(home / "metadata/workspace.lock"):
        repo, record = validate_repository(home)
        require_clean(repo)
        validate_tree(repo, record, "HEAD")
        initial = record["code"]["initial_commit"]
        current = status(repo, record)
        for manifest in sorted((home / "evaluations/init").glob("*/*/manifest.json"), reverse=True):
            try:
                evidence = verify_evaluation(home, manifest.parent, initial, baseline=True)
                return current, evidence
            except (ValueError, OSError, KeyError, TypeError):
                continue
        if current["current_branch"] != "init" or current["commit_sha"] != initial:
            raise ValueError("A complete initial baseline is missing; check out clean init before home_optimize")
        return current, None


def verify_final(home, result):
    """Check the actual checkout and evidence; never repair or discard agent edits."""
    _validate_directory(home, home / "metadata")
    with _initialization_lock(home / "metadata/workspace.lock"):
        repo, record = validate_repository(home)
        require_clean(repo)
        validate_tree(repo, record, "HEAD")
        current = status(repo, record)
        if current["current_branch"] != result.branch_name or current["commit_sha"] != result.commit_sha:
            raise ValueError("Final branch/commit does not match the checked-out HEAD")
        evidence = verify_evaluation(home, result.evaluation_path, result.commit_sha)
        if evidence["manifest"]["branch_name"] != result.branch_name:
            raise ValueError("Final evaluation belongs to a different branch")
        return evidence["evaluation_path"]


def validation_budget_tool(home, max_validations):
    """Expose dev_meta's read-only count, phase and next-candidate guidance."""
    validate_max_validations(max_validations)

    async def get_validation_budget():
        """Return started formal validations, remaining count and optimization guidance."""
        return await blocking(validation_budget_status, home, max_validations)

    return FunctionTool(
        get_validation_budget,
        name="get_validation_budget",
        description=(
            "Report how many formal optimizer validations have started at least one sandbox, how many remain, "
            "and whether the next candidate should emphasize structural code changes or evidence-driven refinement. "
            "Debug-case smoke validations, the initial baseline and failures before sandbox startup are excluded."
        ),
        is_read_only=True,
    )


def evaluation_tool(wrapper, options, *, home, timer=None, max_validations=10):
    """Only debug mode is model-selected; all formal runs rebuild construction."""
    job = required_job(wrapper, "home_evaluate")
    validate_max_validations(max_validations)

    async def evaluate(debug: bool = False, **kwargs):
        """Run debug after committing; after debug passes, run a sampled formal evaluation.

        Debug checks execution, not answer quality. Inspect infrastructure failures
        separately; retries keep the existing commit and selection recovery rules.
        """
        if not isinstance(debug, bool):
            raise ValueError("debug must be boolean")
        if set(kwargs) - {"_agent_state"}:
            raise ValueError("Optimizer state is host-owned and cannot be supplied by the model")
        state = kwargs.get("_agent_state")
        if not isinstance(state, AgentState):
            raise TypeError("Cycle reminders require the host-injected AgentState")
        started_before = await blocking(lambda: set(read_state(home)["started_validations"])) if not debug else set()
        response = await job(
            **options,
            **({"max_validations": max_validations} if not debug else {}),
            debug=debug,
            sampling=not debug,
            record_history=True,
            case_ids=None,
            query_ids=None,
            reuse_construction_from=None,
        )
        if not debug:
            budget = await blocking(validation_budget_status, home, max_validations)
            if isinstance(response.answer, dict):
                response.answer = {**response.answer, "validation_budget": budget}
            completed = await blocking(completed_validation, home, response.answer, started_before)
            if completed is not None:
                manifest, validation_number = completed
                if timer is not None:
                    timer.reset()
                transition = validation_cycle_transition(manifest, validation_number, budget["remaining_validations"])
                state.middle_context.setdefault(PENDING_CYCLE_TRANSITIONS, []).append(transition.hint)
        return {"success": response.success, "answer": response.answer, "metadata": response.metadata}

    tool = FunctionTool(evaluate, name="home_evaluate", is_concurrency_safe=False, is_state_injected=True)
    tool.input_schema = {
        "type": "object",
        "properties": {"debug": {"type": "boolean", "default": False}},
        "additionalProperties": False,
    }
    return tool


class HomeOptimizeStep(BaseStep):
    """Require a full initial baseline, run the coding agent, and verify its final claim."""

    workspace: HomeWorkspace = Ref(HomeWorkspace, "home.workspace")
    agent_wrapper: OptimizerAgentWrapper = Ref(OptimizerAgentWrapper, "agent_wrapper")

    async def execute(self):
        """One optimizer per workspace; never hold the evaluation/Git lock across model calls."""
        if self.context.get("_allowed_paths") is not None:
            raise ValueError("Home optimization requires access to its managed workspace")
        if not self.workspace.is_started or not self.agent_wrapper.is_started:
            raise RuntimeError("Start the Home workspace and optimizer wrapper first")
        spec = OptimizationInput.model_validate(
            {key: self.context[key] for key in OptimizationInput.model_fields if key in self.context},
        )
        options = evaluation_options(self.kwargs.get("evaluation_options", {}))
        self.agent_wrapper.validation_limit()
        home = self.workspace.home_path
        _validate_directory(home, self.workspace.state_path)
        with _initialization_lock(self.workspace.state_path / "optimization.lock"):
            _current, baseline = await blocking(baseline_state, home)
            if baseline is None:
                response = await required_job(self.agent_wrapper, "home_evaluate")(
                    **options,
                    debug=False,
                    sampling=False,
                    record_history=True,
                    case_ids=None,
                    query_ids=None,
                    reuse_construction_from=None,
                )
                if not response.success:
                    raise RuntimeError(f"Initial baseline failed: {response.answer}")
                current, _ = await blocking(baseline_state, home)
                baseline = await blocking(
                    verify_evaluation,
                    home,
                    response.answer["result_dir"],
                    current["commit_sha"],
                    baseline=True,
                )
            return await self._optimize(home, spec, baseline, options)

    async def _optimize(self, home, spec, baseline, options):
        payload = {"status": "running", "instruction": spec.instruction, "baseline_path": baseline["evaluation_path"]}
        report_path = await blocking(save_record, home, "optimization-reports", payload)
        self.context.response.metadata["optimization_report_path"] = report_path
        try:
            timer = self.agent_wrapper.create_cycle_timer()
            max_validations = self.agent_wrapper.validation_limit()
            budget = await blocking(validation_budget_status, home, max_validations)
            reply = await workflow_reply(
                self.agent_wrapper,
                home,
                "optimizer",
                f"Begin optimization.\n{spec.instruction}\n\n"
                f"Completed full initial baseline: {baseline['evaluation_path']}\n"
                f"Formal validation budget: {json.dumps(budget)}\n"
                "Return exactly one JSON object with branch_name, commit_sha (full SHA), evaluation_path and report. "
                "Leave that evaluated branch and commit checked out with no uncommitted changes.",
                tool_factory=lambda backend: [
                    Edit(backend=backend),
                    diagnostic_tool(self.agent_wrapper),
                    validation_budget_tool(home, max_validations),
                    evaluation_tool(
                        self.agent_wrapper,
                        options,
                        home=home,
                        timer=timer,
                        max_validations=max_validations,
                    ),
                ],
                job_names=MUTATION_JOBS,
                agent_options={
                    "_validation_cycle_timer": timer,
                    "_context_fallback_directory": contained(home, "logs/context-offloads", directory=True),
                },
                memory_limit_mib=self.kwargs.get("memory_limit_mib"),
            )
            payload.update(
                agent_report=reply.get("result"),
                session_id=reply.get("session_id"),
                usage=reply.get("usage"),
            )
            result = parse_result(reply["result"])
            evidence_path = await blocking(verify_final, home, result)
            payload.update(result.model_dump(), evaluation_path=evidence_path, status="completed")
            self.context.response.answer = {**payload, "optimization_report_path": report_path}
            self.context.response.metadata.update({key: reply[key] for key in ("session_id", "usage") if key in reply})
        except BaseException as exc:
            payload.update(status="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed", error=str(exc))
            raise
        finally:
            await blocking(write_json, contained(home, report_path), payload)
        return self.context.response
