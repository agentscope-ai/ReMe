"""Optimize Home candidates and verify their baseline and final evaluation evidence."""

import asyncio
import json
import time
from pathlib import Path
from typing import Callable

from agentscope.message import HintBlock
from agentscope.state import AgentState
from agentscope.tool import Edit, FunctionTool
from pydantic import Field

from reme.steps.base_step import BaseStep, Ref
from reme.utils.async_utils import complete_in_thread as blocking

from ..code.repository import (
    EVALUATED_PREFIX,
    resolve_commit_prefix,
    selection_branch,
    validate_repository,
    require_clean,
    validate_tree,
    status,
)
from ..evaluation.artifacts import contained, read_json, write_json
from ..evaluation.inputs import EvaluationSpec, evaluation_head, load_inputs
from ..evaluation.history import read_state
from ..evaluation.budget import validate_max_evaluations, validation_budget_status
from ..evaluation.reuse import verify_case_reuse
from ..preparation.spec import SpecModel
from ..workspace import HomeWorkspace, _initialization_lock, _validate_directory
from .diagnostic import diagnostic_tool, evaluation_paths
from .draft import create_draft_tools
from .bounded import RunTimeBudgetTimer, _format_minutes, _remove_hint_by_identity
from .compression import RecoverableCompressionAgent
from .structured_report import parse_json_object
from .workflow_common import required_job, save_record, workflow_reply
from .wrapper import CaseStudyAgentWrapper

MUTATION_JOBS = (
    "home_create_candidate_branch",
    "home_switch_candidate_branch",
    "home_merge_candidate_branch",
    "home_commit_candidate",
)
PENDING_CYCLE_TRANSITIONS = "home.optimizer.pending_cycle_transitions"
FINAL_SELECTION_HINT = (
    "Final selection: use home_list_validation_results to locate eligible evaluated commits. "
    "Read their paired_comparisons.json files beside summary.json, or use home_compare_validations "
    "for the shortlisted runs. The last run's table compares only that run against history; compare "
    "historical finalists directly when needed. Prefer shared-question score, regressions, coverage "
    "and query token cost; unknown values are not zero and different snapshots do not isolate answer changes. "
    "Read original artifacts only if the compact evidence is insufficient. Leave the selected evaluated "
    "branch checked out and return branch_name, commit_sha (unique prefix accepted), evaluation_path and report."
)
VALIDATION_BUDGET_EXHAUSTED_HINT = (
    "<system-reminder>The formal validation budget is exhausted. All previous validation-cycle timer reminders "
    "are expired. Select the best candidate using existing evaluation evidence; do not develop new candidates "
    "or run further evaluations. Complete any necessary evidence and repository checks, and leave the selected "
    "evaluated branch and full commit checked out with no uncommitted changes. Return exactly one JSON object "
    "with branch_name, commit_sha, evaluation_path and report. Put all summary and explanation in report; "
    "do not include Markdown fences or text outside the JSON object. " + FINAL_SELECTION_HINT + "</system-reminder>"
)


def final_selection_tool(timer=None):
    """Reveal selection guidance only when the optimizer elects to finish early."""

    async def begin_final_selection():
        """Enter final selection when ready to finish optimization early."""
        if timer is not None:
            timer.stop()
        return FINAL_SELECTION_HINT

    return FunctionTool(begin_final_selection, name="begin_final_selection", is_concurrency_safe=False)


class ValidationCycleTimer(RunTimeBudgetTimer):
    """Emit escalating reminders as a scored-validation cycle ages."""

    reminder_thresholds = (0.90, 0.75, 0.50)

    def __init__(self, time_budget_minutes: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__(time_budget_minutes, agent_label="optimizer", clock=clock)
        self.start()

    def reset(self) -> None:
        """Start a fresh cycle after a scored validation finishes."""

        self.start()

    def _reminder_content(self, elapsed_minutes: float, reached_ratio: float) -> str:
        """Keep optimizer guidance tied to the last scored validation."""
        elapsed_ratio = elapsed_minutes / self.time_budget_minutes
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
            return (
                f"<system-reminder>{timing}{compression}"
                "Prioritize completing the code changes, committing them, passing the smoke test, and starting "
                "validation as soon as possible.</system-reminder>"
            )
        remaining = _format_minutes(self.time_budget_minutes - elapsed_minutes)
        return (
            f"<system-reminder>{timing}{compression}"
            f"Complete the code changes, commit them, pass the smoke test, and start validation within {remaining}."
            "</system-reminder>"
        )


class OptimizerAgent(RecoverableCompressionAgent):
    """Add per-validation reminders to HoMe's bounded agent."""

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
        options = self._merged_kwargs({})
        if "max_validations" in options:
            raise ValueError("Use max_evaluations instead of max_validations")
        return validate_max_evaluations(options.get("max_evaluations", 10))

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
    commit_sha: str = Field(pattern="^[0-9a-fA-F]{1,40}$")
    evaluation_path: str = Field(min_length=1)
    report: str = Field(min_length=1)


def parse_result(text):
    """Parse the final report using the shared formatting recovery rules."""
    return OptimizationResult.model_validate(parse_json_object(text, report_name="optimizer"))


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


def _infrastructure_free(manifest, summary):
    return manifest.get("success") is True and all(
        isinstance(counts := summary.get(key), dict) and counts.get("infra_error") == 0
        for key in ("case_status_counts", "query_status_counts")
    )


def verify_evaluation(home, path, commit, *, baseline=False):
    """Accept completed baseline observations; final evidence must be infrastructure-free."""
    run = Path(evaluation_paths(home, [str(path)])[0])
    manifest = read_json(contained(run, "manifest.json"))
    if baseline and (manifest.get("branch_name"), manifest.get("commit_sha")) != ("init", commit):
        raise ValueError("Initial baseline evidence must match the init branch and initial commit")
    summary = read_json(contained(run, "summary.json"))
    required = {
        "commit_sha": commit,
        "status": "completed",
        "mode": "scored",
        "dataset_kind": "training-set",
    }
    completed = (
        all(manifest.get(key) == value for key, value in required.items())
        and manifest.get("sandbox_started_at")
        and summary.get("case_count")
        and summary.get("scoring_enabled") is True
    )
    if not completed or (not baseline and not _infrastructure_free(manifest, summary)):
        raise ValueError("Evaluation must match the commit and complete without infrastructure failures")
    mode = manifest.get("execution_mode")
    reused = bool(manifest.get("construction_sources"))
    if reused:
        if baseline:
            raise ValueError("Initial baseline cannot reuse construction")
    elif mode != "full_validation":
        raise ValueError("Reused evidence requires host-selected per-case construction sources")
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
    # Failed construction may omit answers. Baseline infrastructure errors remain
    # observations; final evidence already passed the strict checks above.
    for key, row in rows.items():
        observed = set(row.get("queries", {}))
        failure_path = contained(run, f"cases/{key}/failure.json")
        failure = read_json(failure_path) if baseline and failure_path.is_file() else {}
        unobserved_answers = (
            baseline and failure.get("stage") == "answer" and failure.get("failure_kind") == "infra_error"
        )
        if observed - set(queries[key]) or (
            observed != set(queries[key])
            and row.get("construction", {}).get("status") != "failure"
            and not unobserved_answers
        ):
            raise ValueError(f"Evaluation summary has incomplete query coverage: {key}")
    if reused:
        verify_case_reuse(home, run, manifest, commit)
    return {"evaluation_path": str(run), "manifest": manifest, "summary": summary}


def baseline_state(home, *, evaluation_path=None):
    """Check a clean repository and return verified initial evidence, or None."""
    _validate_directory(home, home / "metadata")
    with _initialization_lock(home / "metadata/workspace.lock"):
        repo, record = validate_repository(home)
        require_clean(repo)
        validate_tree(repo, record, "HEAD")
        initial = record["code"]["initial_commit"]
        current = status(repo, record)
        if evaluation_path is not None:
            if current["current_branch"] != "init" or current["commit_sha"] != initial:
                raise ValueError("New initial baseline requires clean init at the initial commit")
            return verify_evaluation(home, evaluation_path, initial, baseline=True)
        # Attempt IDs are allocated in order under the initial commit's directory.
        attempts = (
            path.parent
            for path in (home / "evaluations/init").glob("*/*/manifest.json")
            if path.parent.name.isascii() and path.parent.name.isdecimal()
        )
        fallback = None
        for run in sorted(attempts, key=lambda path: int(path.name), reverse=True):
            try:
                evidence = verify_evaluation(home, run, initial, baseline=True)
            except (ValueError, OSError, KeyError, TypeError):
                continue
            if _infrastructure_free(evidence["manifest"], evidence["summary"]):
                return evidence
            if fallback is None:
                fallback = evidence
        if fallback is not None:
            return fallback
        if current["current_branch"] != "init" or current["commit_sha"] != initial:
            raise ValueError("A complete initial baseline is missing; check out clean init before home_optimize")
        return None


def _verify_final_checkout(home, result):
    """Resolve and validate the final checkout while holding the workspace lock."""
    _validate_directory(home, home / "metadata")
    with _initialization_lock(home / "metadata/workspace.lock"):
        repo, record = validate_repository(home)
        require_clean(repo)
        validate_tree(repo, record, "HEAD")
        current = status(repo, record)
        result.commit_sha = resolve_commit_prefix(repo, result.commit_sha)
        result.branch_name = selection_branch(repo, result.branch_name)
        if current["current_branch"] != result.branch_name or current["commit_sha"] != result.commit_sha:
            raise ValueError("Final branch/commit does not match the checked-out HEAD")


def verify_final(home, result):
    """Scan immutable evidence outside the lock, then recheck the actual checkout."""
    _verify_final_checkout(home, result)
    evidence = verify_evaluation(home, result.evaluation_path, result.commit_sha)
    if result.branch_name not in {evidence["manifest"]["branch_name"], EVALUATED_PREFIX + result.commit_sha}:
        raise ValueError("Final evaluation belongs to a different branch")
    _verify_final_checkout(home, result)
    return evidence["evaluation_path"]


def validation_budget_tool(home, max_evaluations):
    """Expose dev_meta's read-only count, phase and next-candidate guidance."""
    validate_max_evaluations(max_evaluations)

    async def get_validation_budget():
        """Return started formal validations, remaining count and optimization guidance."""
        return await blocking(validation_budget_status, home, max_evaluations)

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


def evaluation_checkout(home):
    """Bind the host-owned sampling decision to one actual checkout."""
    with _initialization_lock(home / "metadata/workspace.lock"):
        repo, record = validate_repository(home)
        commit, branch = evaluation_head(repo)
        return {
            "expected_commit": commit,
            "expected_branch": branch,
            "sampling": branch != "init" and commit != record["code"]["initial_commit"],
        }


def evaluation_tool(wrapper, options, *, home, timer=None, max_evaluations=10, budget_exhausted=False):
    """Expose optional formal targets; execution limits and construction replay remain host-owned."""
    job = required_job(wrapper, "home_evaluate")
    validate_max_evaluations(max_evaluations)

    async def evaluate(debug: bool = False, target_queries: dict[str, list[str] | None] | None = None, **kwargs):
        """Run debug after committing; after debug passes, run a sampled formal evaluation.

        Debug checks execution, not answer quality. Inspect infrastructure failures
        separately; retries keep the existing commit and selection recovery rules.
        """
        nonlocal budget_exhausted
        if not isinstance(debug, bool):
            raise ValueError("debug must be boolean")
        if set(kwargs) - {"_agent_state"}:
            raise ValueError("Optimizer state is host-owned and cannot be supplied by the model")
        state = kwargs.get("_agent_state")
        if not isinstance(state, AgentState):
            raise TypeError("Cycle reminders require the host-injected AgentState")
        started_before = await blocking(lambda: set(read_state(home)["started_validations"])) if not debug else set()
        checkout = await blocking(evaluation_checkout, home) if not debug else {"sampling": False}
        if target_queries and not debug and not checkout["sampling"]:
            raise ValueError(
                "Targeted evaluation requires a candidate branch; init retries cover the full training set",
            )
        response = await job(
            **options,
            **({"max_evaluations": max_evaluations} if not debug else {}),
            debug=debug,
            **checkout,
            record_history=True,
            case_ids=None,
            query_ids=None,
            reuse_construction_from=None,
            **({"target_queries": target_queries} if not debug and target_queries is not None else {}),
        )
        if isinstance(response.answer, dict):
            response.answer = {key: value for key, value in response.answer.items() if key != "paired_comparisons"}
        if not debug:
            budget = await blocking(validation_budget_status, home, max_evaluations)
            if isinstance(response.answer, dict):
                response.answer = {**response.answer, "validation_budget": budget}
            if budget["remaining_validations"] == 0:
                if timer is not None:
                    timer.stop()
                if not budget_exhausted:
                    state.middle_context.setdefault(PENDING_CYCLE_TRANSITIONS, []).append(
                        VALIDATION_BUDGET_EXHAUSTED_HINT,
                    )
                    budget_exhausted = True
            else:
                completed = await blocking(completed_validation, home, response.answer, started_before)
                if completed is not None:
                    manifest, validation_number = completed
                    if timer is not None:
                        timer.reset()
                    transition = validation_cycle_transition(
                        manifest,
                        validation_number,
                        budget["remaining_validations"],
                    )
                    state.middle_context.setdefault(PENDING_CYCLE_TRANSITIONS, []).append(transition.hint)
        return {"success": response.success, "answer": response.answer, "metadata": response.metadata}

    tool = FunctionTool(evaluate, name="home_evaluate", is_concurrency_safe=False, is_state_injected=True)
    tool.input_schema = {
        "type": "object",
        "properties": {
            "debug": {"type": "boolean", "default": False},
            "target_queries": {
                "type": ["object", "null"],
                "additionalProperties": {"type": ["array", "null"], "items": {"type": "string"}},
                "description": (
                    "Formal targets keyed by case ID: [] prioritizes a case with sampled questions; null targets "
                    "all its questions; a list targets those query IDs. After a change, you may target relevant "
                    "cases and questions to check whether it has the intended effect. "
                    "Ignored for debug."
                ),
            },
        },
        "additionalProperties": False,
    }
    return tool


class HomeOptimizeStep(BaseStep):
    """Collect a full initial baseline, run the coding agent, and verify its final claim."""

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
            baseline = await blocking(baseline_state, home)
            if baseline is None:
                response = await required_job(self.agent_wrapper, "home_evaluate")(
                    **{**options, "fail_fast": False},
                    debug=False,
                    sampling=False,
                    expected_commit=read_json(home / "metadata/preparation.json")["code"]["initial_commit"],
                    expected_branch="init",
                    record_history=True,
                    case_ids=None,
                    query_ids=None,
                    reuse_construction_from=None,
                )
                if not isinstance(response.answer, dict) or response.answer.get("status") != "completed":
                    raise RuntimeError(f"Initial baseline failed: {response.answer}")
                baseline = await blocking(
                    baseline_state,
                    home,
                    evaluation_path=response.answer["result_dir"],
                )
            return await self._optimize(home, spec, baseline, options)

    async def _optimize(self, home, spec, baseline, options):
        payload = {"status": "running", "instruction": spec.instruction, "baseline_path": baseline["evaluation_path"]}
        report_path = await blocking(save_record, home, "optimization-reports", payload)
        self.context.response.metadata["optimization_report_path"] = report_path
        try:
            timer = self.agent_wrapper.create_cycle_timer()
            max_evaluations = self.agent_wrapper.validation_limit()
            budget = await blocking(validation_budget_status, home, max_evaluations)
            budget_exhausted = budget["remaining_validations"] == 0
            if budget_exhausted and timer is not None:
                timer.stop()
            reply = await workflow_reply(
                self.agent_wrapper,
                home,
                "optimizer",
                f"Begin optimization.\n{spec.instruction}\n\n"
                f"Completed full initial baseline: {baseline['evaluation_path']}\n"
                f"Baseline case statuses: {json.dumps(baseline['summary']['case_status_counts'])}\n"
                f"Formal validation budget: {json.dumps(budget)}\n"
                f"{VALIDATION_BUDGET_EXHAUSTED_HINT if budget_exhausted else ''}",
                tool_factory=lambda backend: [
                    *create_draft_tools()[0],
                    Edit(backend=backend),
                    diagnostic_tool(self.agent_wrapper),
                    validation_budget_tool(home, max_evaluations),
                    final_selection_tool(timer),
                    evaluation_tool(
                        self.agent_wrapper,
                        options,
                        home=home,
                        timer=timer,
                        max_evaluations=max_evaluations,
                        budget_exhausted=budget_exhausted,
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
