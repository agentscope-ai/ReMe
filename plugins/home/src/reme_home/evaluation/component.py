"""Home evaluation orchestration using the shared Docker pool and a host judge."""

import asyncio
import os
import shutil
import traceback
from contextlib import nullcontext

from deploy.benchmark_docker import BenchmarkDockerPool, CasePlan, DockerReMeSandboxFactory, SourceCandidate

from reme.components.base_component import BaseComponent
from reme.utils.async_utils import complete_in_thread as blocking
from reme.utils.async_utils import finish_cleanup

from ..preparation.data.models import utc_now
from ..code.repository import preserve_evaluated, validate_repository
from .adapters import accepts_build, build_jobs, evaluation_query
from .artifacts import (
    contained,
    file_sha256,
    inspect_archive,
    read_json,
    summarize,
    write_json,
    construction_exception,
    answer_exception,
    failure_kind,
)
from .inputs import EvaluationSpec, load_inputs, replay_sampling_inputs, reuse_inputs, snapshot_code
from .compaction import compact_construction, compact_query
from ..judge import HomeJudge
from ..workspace import HomeWorkspace, _initialization_lock, _validate_directory
from ..sampling import SamplingPolicy, create_sampling_plan
from .history import HomeEvaluationHistory
from .budget import validation_budget_status
from .repetition import previous_candidate_result
from .reuse import (
    automatic_reuse,
    case_reuse_inputs,
    construction_fingerprints,
    reuse_context,
    reuse_response,
    verify_candidate,
)


def selection_summary(plan):
    """Keep tool feedback bounded; full requests and dropped IDs stay in the saved plan."""
    if plan is None:
        return None
    targets = plan.get("targets", {})
    invalid_cases = len(targets.get("invalid_case_ids", []))
    invalid_queries = sum(len(ids) for ids in targets.get("invalid_query_ids", {}).values())
    notices = []
    if invalid_cases or invalid_queries:
        notices.append(f"Ignored {invalid_cases} invalid case IDs and {invalid_queries} invalid query IDs.")
    fallback_count = sum(
        bool(targets.get("requested", {}).get(key)) and set(targets["requested"][key]) <= set(ids)
        for key, ids in targets.get("invalid_query_ids", {}).items()
    )
    if fallback_count:
        notices.append(f"Used case-only sampling for {fallback_count} cases with no valid target query IDs.")
    if targets.get("requested") and not targets.get("retained_case_ids"):
        notices.append("No valid target cases remained; used automatic selection.")
    if plan.get("recovery_from"):
        notices.append("Restored the previous selection and memory sources; new targets were not applied.")
    return {
        "case_count": len(plan["case_ids"]),
        "query_count": sum(len(ids) for ids in plan["query_ids"].values()),
        "target_case_count": len(targets.get("retained_case_ids", [])),
        "target_query_count": sum(len(ids) for ids in targets.get("retained_query_ids", {}).values()),
        "invalid_case_count": invalid_cases,
        "invalid_query_count": invalid_queries,
        "trimmed_case_count": len(targets.get("trimmed_case_ids", [])),
        "trimmed_query_count": sum(len(ids) for ids in targets.get("trimmed_query_ids", {}).values()),
        "notices": notices,
    }


def selected_cases(cases, plan):
    """Apply a host-owned selection while preserving its case and question order."""
    by_id = {case.case_id: case for case in cases}
    result = []
    for key in plan["case_ids"]:
        case = by_id[key]
        queries = {query.query_id: query for query in case.queries}
        result.append(case.model_copy(update={"queries": [queries[qid] for qid in plan["query_ids"][key]]}))
    return result


def retain_sampled_reuse(reuse, cases, plan):
    """Keep reviewed provenance only for the final selection, without another draw."""
    reuse["target_case_ids"] = plan.get("targets", {}).get("retained_case_ids", [])
    had_sources = bool(reuse.get("case_sources"))
    retained = {case.case_id for case in cases}
    for field in ("case_sources", "rebuild_reasons"):
        if field in reuse:
            reuse[field] = {key: value for key, value in reuse[field].items() if key in retained}
    reuse["selected"] = bool(reuse.get("case_sources"))
    if had_sources and not reuse["selected"]:
        reuse["reason"] = "no_reusable_snapshot_for_sampled_cases"
    rebuild_reasons = reuse.setdefault("rebuild_reasons", {})
    for case_id in retained - reuse.get("case_sources", {}).keys():
        rebuild_reasons.setdefault(case_id, "no_compatible_reviewed_snapshot")


class HomeEvaluation(BaseComponent):
    """Own evaluation tasks; the application owns the host judge Job lifecycle."""

    component_type = "home.evaluation"

    def __init__(self, sampling_policy: dict | None = None, **kwargs):
        from ..agent.wrapper import BoundedAgentWrapper

        if "sampling" in kwargs:
            raise ValueError("Use sampling_policy for Component defaults; sampling is a Job boolean")
        super().__init__(**kwargs)
        self.workspace = self.bind("default", HomeWorkspace, optional=False)
        self.judge = self.bind("default", HomeJudge, optional=False)
        self.history = self.bind("default", HomeEvaluationHistory, optional=False)
        self.candidate_gate = self.bind("home_candidate_gate", BoundedAgentWrapper, optional=False)
        self.construction_equivalence = self.bind("home_construction_equivalence", BoundedAgentWrapper, optional=False)
        self._pending: set[asyncio.Task] = set()
        self.sampling_policy = SamplingPolicy.model_validate(sampling_policy if sampling_policy is not None else {})

    async def _close(self):
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)

    async def evaluate(self, **kwargs) -> dict:
        """Run through the component lifecycle, including cancellation cleanup."""
        if not self.is_started:
            raise RuntimeError("Start the Home application before evaluation")
        spec = EvaluationSpec.model_validate(kwargs)
        override = spec.sampling_policy.model_dump(exclude_unset=True) if spec.sampling_policy is not None else {}
        policy = SamplingPolicy.model_validate({**self.sampling_policy.model_dump(), **override})
        spec = spec.model_copy(update={"sampling_policy": policy})
        task = asyncio.create_task(self._serialized_evaluate(spec))
        self._pending.add(task)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel()
            try:
                await finish_cleanup(asyncio.gather(task, return_exceptions=True))
            except asyncio.CancelledError:
                pass
            raise
        finally:
            self._pending.discard(task)

    async def _serialized_evaluate(self, spec):
        """Match dev_meta's serialized formal validations; debug never owns sampling state."""
        _validate_directory(self.workspace_path.resolve(), self.workspace.state_path)
        lock = nullcontext() if spec.debug else _initialization_lock(self.workspace.state_path / "validation.lock")
        with lock:
            budgeted = spec.max_evaluations is not None and not spec.debug
            if budgeted:
                budget = await blocking(validation_budget_status, self.workspace.home_path, spec.max_evaluations)
                if budget["remaining_validations"] < 1:
                    return {
                        "status": "rejected",
                        "error_code": "VALIDATION_BUDGET_EXHAUSTED",
                        "error": "Optimizer validation budget is exhausted; no sandbox was started.",
                        "validation_budget": budget,
                        "result_dir": None,
                    }
            admission = None
            admission_checkout = None
            if budgeted and budget["used_validations"] < budget["gated_validation_count"]:
                from ..agent.candidate_gate import admission_context, review_candidate

                with _initialization_lock(self.workspace.state_path / "workspace.lock"):
                    admission_checkout, context = await blocking(admission_context, self.workspace.home_path)
                if context is not None:
                    admission = await review_candidate(
                        self.candidate_gate,
                        self.workspace.home_path,
                        context,
                        budget,
                        memory_limit_mib=self.kwargs.get("candidate_gate_memory_limit_mib"),
                    )
                    if admission["decision"] == "reject":
                        return {
                            "status": "rejected",
                            "error_code": "CANDIDATE_GATE_REJECTED",
                            "error": admission["reason"],
                            "candidate_gate": admission,
                            "validation_budget": budget,
                            "result_dir": None,
                        }
            result = await self._evaluate(spec, admission, admission_checkout)
            if budgeted:
                result["validation_budget"] = await blocking(
                    validation_budget_status,
                    self.workspace.home_path,
                    spec.max_evaluations,
                )
            return result

    def _prepare(self, spec, environment, admission_checkout=None):
        home = self.workspace.home_path
        recovery_plan = None
        previous_manifest = None
        _validate_directory(self.workspace_path.resolve(), self.workspace.state_path)
        with _initialization_lock(self.workspace.state_path / "workspace.lock"):
            record, dataset, cases = load_inputs(home, spec)
            snapshot, commit, branch, checkout_branch = snapshot_code(home, record)
            if admission_checkout is not None and (commit, branch) != (
                admission_checkout.commit,
                admission_checkout.branch,
            ):
                raise ValueError("Candidate changed during admission; retry on the intended commit and branch")
            previous = None
            if not spec.debug:
                # The admission check already looked up recovery for this non-baseline checkout.
                previous = (
                    admission_checkout.previous_result
                    if admission_checkout is not None and commit != record["code"]["initial_commit"]
                    else previous_candidate_result(home, commit)
                )
            if previous is not None:
                manifest = read_json(contained(home, previous / "manifest.json"))
                previous_manifest = manifest
                recovery_spec = EvaluationSpec.model_validate(
                    {
                        **spec.model_dump(),
                        "sampling": False,
                        "target_queries": None,
                        "case_ids": manifest["case_ids"],
                        "query_ids": manifest["query_ids"],
                    },
                )
                if (
                    recovery_spec.case_ids is None
                    or recovery_spec.query_ids is None
                    or set(recovery_spec.query_ids) != set(recovery_spec.case_ids)
                ):
                    raise ValueError("Recovery requires saved case IDs and query IDs for every case")
                # Validate saved IDs against the prepared dataset, including queries
                # outside the caller's new selection. Never silently drop a case.
                record, dataset, cases = load_inputs(home, recovery_spec)
                recovery_plan = {
                    "mode": "explicit",
                    "case_ids": manifest["case_ids"],
                    "query_ids": manifest["query_ids"],
                    "policy": spec.sampling_policy.model_dump(),
                    "recovery_from": previous.relative_to(home).as_posix(),
                }
                if "target_selection" in manifest:
                    recovery_plan["targets"] = manifest["target_selection"]
        # Automatic reuse reviews all available cases before final sampling.
        defer_sampling = spec.reuse_construction_from is None
        sampling_state = (
            self.history.sampling_state()
            if spec.sampling and not spec.debug and previous is None and not defer_sampling
            else None
        )
        # The validation lock serializes formal runs while replay preflights archives.
        plan = recovery_plan
        fingerprints = construction_fingerprints(record, dataset, spec, environment)
        if previous_manifest is not None:
            sources = previous_manifest.get("construction_sources", {})
            if not sources and previous_manifest.get("execution_mode") in {"query_only_replay", "mixed_validation"}:
                raise ValueError("Recovery requires saved per-case construction sources")
            sources, archives = case_reuse_inputs(home, spec, cases, fingerprints, sources)
        elif spec.sampling and not spec.debug and not defer_sampling:
            source, verified_archives, eligibility = replay_sampling_inputs(home, spec, cases, fingerprints)
            plan = create_sampling_plan(
                cases,
                sampling_state,
                spec.sampling_policy,
                eligible_query_ids=eligibility,
                target_queries=spec.target_queries,
            )
            cases = selected_cases(cases, plan)
            sources = {case.case_id: source for case in cases}
            archives = {case.case_id: verified_archives[case.case_id] for case in cases}
        else:
            source, archives = reuse_inputs(home, spec, cases, fingerprints)
            sources = {case.case_id: source for case in cases} if source else {}
        return (
            record,
            dataset,
            cases,
            snapshot,
            commit,
            branch,
            checkout_branch,
            fingerprints,
            sources,
            archives,
            plan,
            previous_manifest,
        )

    def _verify_candidate(self, commit, branch, checkout_branch=None):
        """Reject checkout changes between selection, review and execution."""
        with _initialization_lock(self.workspace.state_path / "workspace.lock"):
            verify_candidate(self.workspace.home_path, commit, branch, checkout_branch=checkout_branch)

    def _sampling_started(self, run, manifest, plan, policy=None, *, record_history=True):
        """The factory invokes this only after a real sandbox successfully initializes."""
        if not record_history:
            manifest["sandbox_started_at"] = utc_now().isoformat()
            if plan is not None:
                # Keep draw evidence local without claiming a started-history round.
                manifest["selection_plan"] = plan
            write_json(run / "manifest.json", manifest)
            return
        try:
            self.history.record_started(run, policy if policy is not None else self.sampling_policy, plan)
        finally:
            # Cache publication can fail after the durable start was written.
            # Preserve those fields when the caller later saves failure status.
            manifest.update(read_json(run / "manifest.json"))

    def _allocate(self, branch, commit, manifest):
        """Allocate ascending IDs under a collision-safe commit prefix."""
        home = self.workspace.home_path
        with _initialization_lock(self.workspace.state_path / "workspace.lock"):
            if manifest.get("mode") == "scored":
                repo, _ = validate_repository(home)
                manifest["evaluated_branch"] = preserve_evaluated(repo, commit)
            parent = contained(home, f"evaluations/{branch}", directory=True)
            parent.mkdir(parents=True, exist_ok=True)
            width = 7
            while True:
                folder = contained(parent, commit[:width], directory=True)
                prior = list(folder.glob("*/manifest.json")) if folder.exists() else []
                if all(read_json(path).get("commit_sha") == commit for path in prior):
                    break
                width += 1
                if width > len(commit):
                    raise ValueError("Invalid evaluation commit directory")
            folder.mkdir(exist_ok=True)
            numbers = [int(path.name) for path in folder.iterdir() if path.name.isdigit()]
            run = folder / str(max(numbers, default=0) + 1)
            run.mkdir()
            manifest["validation_id"] = run.name
            write_json(run / "manifest.json", manifest)
            return run

    async def _select_automatic_reuse(self, spec, cases, fingerprints, commit, branch, deferred_sampling):
        """Bound targeted reviews and produce the final selection before starting a sandbox."""
        sampling_plan = None
        targeted = deferred_sampling and bool(set(spec.target_queries or {}) & {case.case_id for case in cases})
        with _initialization_lock(self.workspace.state_path / "workspace.lock"):
            context = await blocking(
                reuse_context,
                self.workspace.home_path,
                spec,
                cases,
                fingerprints,
                commit,
                branch,
                replay_selection=deferred_sampling and not targeted,
                targeted=targeted,
            )
        review_cases = None
        if targeted:
            sampling_state = await blocking(self.history.sampling_state)
            sampling_plan = create_sampling_plan(
                cases,
                sampling_state,
                spec.sampling_policy,
                target_queries=spec.target_queries,
                reusable_case_ids=set(context.get("candidate_sources", {})) if context else None,
            )
            review_cases = selected_cases(cases, sampling_plan)
        reuse = await automatic_reuse(
            self.construction_equivalence,
            self.workspace.home_path,
            context,
            memory_limit_mib=self.kwargs.get("construction_equivalence_memory_limit_mib"),
            cases=review_cases,
        )
        await blocking(self._verify_candidate, commit, branch)
        if deferred_sampling:
            replay = reuse.get("replay_selection") if reuse and reuse["selected"] else None
            if targeted:
                approved = (reuse or {}).get("case_sources", {})
                if sampling_plan["prefer_reusable_background"] and not (
                    approved and set(sampling_plan["targets"]["retained_case_ids"]) <= approved.keys()
                ):
                    sampling_plan = create_sampling_plan(
                        cases,
                        sampling_state,
                        spec.sampling_policy,
                        target_queries=spec.target_queries,
                        seed=sampling_plan["seed"],
                    )
            elif replay is not None:
                sampling_plan = {
                    "mode": "replay",
                    "policy": spec.sampling_policy.model_dump(),
                    "source_validation": reuse["source_validation"],
                    **replay,
                }
                if spec.target_queries:
                    sampling_plan["targets"] = {
                        "requested": spec.target_queries,
                        "invalid_case_ids": list(spec.target_queries),
                        "invalid_query_ids": {},
                        "retained_case_ids": [],
                        "retained_query_ids": {},
                        "trimmed_case_ids": [],
                        "trimmed_query_ids": {},
                    }
            else:
                sampling_plan = create_sampling_plan(
                    cases,
                    await blocking(self.history.sampling_state),
                    spec.sampling_policy,
                    target_queries=spec.target_queries,
                    reusable_case_ids=set(reuse["case_sources"]) if reuse and reuse["selected"] else None,
                )
            cases = selected_cases(cases, sampling_plan)
            if reuse is not None:
                retain_sampled_reuse(reuse, cases, sampling_plan)
        return cases, sampling_plan, reuse

    async def _evaluate(self, spec, admission=None, admission_checkout=None):
        environment = {**os.environ, **self.app_context.app_config.environment}
        prepared = await blocking(self._prepare, spec, environment, admission_checkout)
        (
            record,
            dataset,
            cases,
            snapshot,
            commit,
            branch,
            checkout_branch,
            fingerprints,
            sources,
            archives,
            sampling_plan,
            previous,
        ) = prepared
        reuse = None
        deferred_sampling = spec.sampling and not spec.debug and sampling_plan is None and previous is None
        if previous is not None:
            saved = previous.get("automatic_construction_reuse")
            if saved is not None:
                reuse = {**saved, "candidate_commit": commit, "branch_name": branch}
        elif (
            not spec.debug
            and spec.reuse_construction_from is None
            and (spec.sampling or spec.max_evaluations is not None)
        ):
            cases, sampling_plan, reuse = await self._select_automatic_reuse(
                spec,
                cases,
                fingerprints,
                commit,
                branch,
                deferred_sampling,
            )
            if reuse is not None and reuse["selected"]:
                sources, archives = await blocking(
                    case_reuse_inputs,
                    self.workspace.home_path,
                    spec,
                    cases,
                    fingerprints,
                    reuse["case_sources"],
                )
        await blocking(self._verify_candidate, commit, branch, checkout_branch)
        reuse_feedback = reuse_response(reuse) if reuse is not None else None
        # Each case can load a verified snapshot or build its own memory.
        jobs = {case.case_id: build_jobs(case) for case in cases if case.case_id not in sources}
        mode = (
            "full_validation"
            if not sources
            else ("query_only_replay" if len(sources) == len(cases) else "mixed_validation")
        )
        judge = None if spec.debug else self.judge
        run = None
        try:
            manifest = {
                "branch_name": branch,
                "commit_sha": commit,
                "case_ids": [case.case_id for case in cases],
                "query_ids": {case.case_id: [q.query_id for q in case.queries] for case in cases},
                "dataset": dataset.dataset,
                "dataset_kind": spec.dataset_kind,
                "mode": "debug" if judge is None else "scored",
                "execution_mode": mode,
                "fingerprints": fingerprints,
                "options": spec.model_dump(
                    mode="json",
                    exclude={
                        "case_ids",
                        "query_ids",
                        "target_queries",
                        "reuse_construction_from",
                        "debug",
                        "sampling_policy",
                    },
                ),
                "judge": self.judge.identity(dataset.dataset),
                "status": "running",
                "started_at": utc_now().isoformat(),
            }
            if sources:
                manifest["construction_sources"] = sources
            if sampling_plan is not None and "targets" in sampling_plan:
                manifest["target_selection"] = sampling_plan["targets"]
            if "test_evaluation" in record:
                manifest["test_evaluation"] = record["test_evaluation"]
            if admission is not None:
                manifest["candidate_gate"] = admission
            if reuse is not None:
                manifest["automatic_construction_reuse"] = reuse
            run = self._allocate(branch, commit, manifest)
            plans = []
            for case in cases:
                source = sources.get(case.case_id)
                folder = run / "cases" / case.case_id
                memory = folder / "memory_construction"
                memory.mkdir(parents=True)
                if source:
                    # Retain an independent pre-query snapshot and inherited log.
                    archive, expected_sha = archives[case.case_id]
                    await blocking(shutil.copyfile, archive, memory / "reme_workspace.tar.gz")
                    actual_sha = await blocking(file_sha256, memory / "reme_workspace.tar.gz")
                    if actual_sha != expected_sha:
                        raise ValueError(f"Construction source changed after preflight: {case.case_id}")
                    log = contained(archive.parent, "build.log")
                    if log.is_file():
                        await blocking(shutil.copyfile, log, memory / "build.log")
                    write_json(
                        memory / "result.json",
                        {
                            "success": True,
                            "status": "reused",
                            "source": source,
                            "workspace_sha256": actual_sha,
                            "jobs": [],
                        },
                    )
                plans.append(
                    CasePlan(
                        case_id=case.case_id,
                        build_jobs=jobs.get(case.case_id, ()),
                        queries=tuple(evaluation_query(q) for q in case.queries),
                        workspace_source=memory / "reme_workspace.tar.gz" if source else None,
                        export_workspace_to=None if source else memory / "reme_workspace.tar.gz",
                        export_build_log_to=None if source else memory / "build.log",
                        export_evaluation_to=folder / "evaluation.tar.gz",
                        export_diagnostics_to=folder / "full.tar.gz",
                    ),
                )
            if sources:
                await blocking(self._verify_candidate, commit, branch, checkout_branch)
            factory = DockerReMeSandboxFactory(
                SourceCandidate(
                    snapshot,
                    base_image=spec.base_image,
                    install_projects=tuple(record["code"]["install_projects"]),
                ),
                config=record["code"]["config"],
                plugins=tuple(record["code"]["plugins"]),
                command_timeout=spec.timeout_seconds,
                env=environment,
                on_sandbox_started=(
                    None
                    if spec.debug
                    else lambda: blocking(
                        self._sampling_started,
                        run,
                        manifest,
                        sampling_plan,
                        spec.sampling_policy,
                        record_history=spec.record_history,
                    )
                ),
            )
            semaphore = asyncio.Semaphore(spec.concurrency)
            by_id = {case.case_id: case for case in cases}

            async def completed(plan, result):
                try:
                    await self._persist_case(run, by_id[plan.case_id], result, sources.get(plan.case_id), spec)
                except Exception as error:
                    result.error = (
                        result.error + "; " if result.error else ""
                    ) + f"artifact persistence failed: {error}"
                    write_json(
                        run / "cases" / plan.case_id / "failure.json",
                        {"stage": "artifacts", "error": result.error, "failure_kind": "infra_error"},
                    )
                await self._score_case(
                    run,
                    by_id[plan.case_id],
                    dataset.dataset,
                    judge,
                    semaphore,
                    spec,
                    (result.queries or {}).get("queries", []),
                )

            results = await BenchmarkDockerPool(factory, concurrency=spec.concurrency).run(
                plans,
                on_case_finished=completed,
                fail_fast=spec.fail_fast,
                accept_build=lambda build: accepts_build(build, spec.max_construction_failure_rate),
            )
            # Pool teardown can attach late cleanup failures after callbacks.
            for result in results:
                if result.error:
                    failure = run / "cases" / result.case_id / "failure.json"
                    if not failure.exists():
                        write_json(failure, {"stage": "cleanup", "error": result.error, "failure_kind": "infra_error"})
                    else:
                        previous = read_json(failure)
                        if previous.get("error") != result.error:
                            previous.update(cleanup_error=result.error, failure_kind="infra_error")
                            write_json(failure, previous)
            summary = summarize(run, cases, scoring_required=not spec.debug)
            manifest.update(
                status="completed",
                success=summary["case_status_counts"]["infra_error"] == 0,
                finished_at=utc_now().isoformat(),
            )
            if not spec.debug and spec.record_history:
                await blocking(self.history.publish_completed, run, manifest, summary)
            else:
                write_json(run / "manifest.json", manifest)
                write_json(run / "summary.json", summary, compact_leaf_containers=True)
            comparison = None
            if not spec.debug:
                try:
                    from .comparison import publish_comparisons

                    comparison = await blocking(publish_comparisons, self.workspace.home_path, run)
                except Exception:
                    self.logger.exception(f"Paired comparison publication failed for completed evaluation {run}")
            return {
                "result_dir": str(run),
                "evaluated_branch": manifest.get("evaluated_branch"),
                **({"paired_comparisons": comparison} if spec.debug or comparison is not None else {}),
                "commit_sha": commit,
                "execution_mode": manifest["execution_mode"],
                "record_history": spec.record_history and not spec.debug,
                "status": "completed",
                "selection": selection_summary(sampling_plan),
                "selection_plan_path": manifest.get("sampling", {}).get("plan_path"),
                **({"candidate_gate": admission} if admission is not None else {}),
                **({"automatic_construction_reuse": reuse_feedback} if reuse_feedback is not None else {}),
                **{key: value for key, value in summary.items() if key != "cases"},
            }
        except BaseException as error:
            # A full summary is the durable completion boundary. Bookkeeping
            # errors or cancellation must not demote an already completed run.
            completed = run is not None and manifest.get("status") == "completed" and (run / "summary.json").is_file()
            if completed:
                raise
            if run is not None:
                manifest.update(
                    status="cancelled" if isinstance(error, asyncio.CancelledError) else "failed",
                    success=False,
                    finished_at=utc_now().isoformat(),
                )
                write_json(run / "manifest.json", manifest)
                write_json(
                    run / "failure.json",
                    {
                        "stage": "evaluation",
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                        "failed_at": utc_now().isoformat(),
                    },
                )
                partial = summarize(
                    run,
                    cases,
                    scoring_required=not spec.debug,
                )
                write_json(run / "partial_summary.json", partial, compact_leaf_containers=True)
            if isinstance(error, asyncio.CancelledError) or run is None:
                raise
            raise RuntimeError(f"Evaluation failed; artifacts: {run}: {error}") from error

    @staticmethod
    async def _persist_case(run, case, result, source, spec):
        folder = run / "cases" / case.case_id
        memory = folder / "memory_construction"
        # Freeze every returned answer in the sole authoritative query file
        # before extraction or judging can fail. Unjudged answers stay present.
        by_id = {row.get("query_id"): row for row in (result.queries or {}).get("queries", [])}
        rows = [
            compact_query(
                {
                    **by_id[query.query_id],
                    "query_id": query.query_id,
                    "question": query.question,
                    "groundtruth": query.groundtruth.model_dump(mode="json"),
                    "question_type": query.metadata.get("question_type"),
                    "score": None,
                    "judge_result": None,
                },
            )
            for query in case.queries
            if query.query_id in by_id
        ]
        write_json(folder / "queries/result.json", {"queries": rows}, compact_leaf_containers=True)
        if result.error:
            stage = result.failure_stage or "execution"
            exception = result.exception
            if exception is None:
                exception = (
                    answer_exception(result.queries or {})
                    if stage == "answer"
                    else construction_exception(result.build or {})
                )
            write_json(
                folder / "failure.json",
                {
                    "stage": stage,
                    "error": result.error,
                    "exception": exception,
                    "failure_kind": failure_kind(exception, stage),
                },
            )
        if source is None:
            build = result.build or {"success": False, "error": result.error, "jobs": []}
            accepted = accepts_build(build, spec.max_construction_failure_rate)
            construction = {
                **build,
                "status": "success" if build.get("success") else "partial_success" if accepted else "failure",
            }
            archive = memory / "reme_workspace.tar.gz"
            if archive.is_file():
                construction["workspace_sha256"] = await blocking(file_sha256, archive)
            write_json(memory / "result.json", compact_construction(construction), compact_leaf_containers=True)
        archive = memory / "reme_workspace.tar.gz"
        if archive.is_file():
            await blocking(inspect_archive, archive, spec.max_artifact_bytes, memory / "reme_workspace")
        evaluation = folder / "evaluation.tar.gz"
        if evaluation.is_file():
            extracted = folder / "exported"
            await blocking(inspect_archive, evaluation, spec.max_artifact_bytes, extracted)
            # Keep per-query logs beside host authoritative results. Container
            # query scores are intentionally ignored; judging happens outside it.
            for query in case.queries:
                log = extracted / "queries" / query.query_id / "answer.log"
                if log.is_file():
                    target = folder / "queries" / query.query_id / "answer.log"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    await blocking(shutil.copyfile, log, target)
            # Container results cannot overwrite host-frozen answers or scores.
            await blocking(shutil.rmtree, extracted)
            evaluation.unlink()

    @staticmethod
    async def _score_case(run, case, dataset, judge, semaphore, spec, answers):
        folder = run / "cases" / case.case_id
        by_id = {row.get("query_id"): row for row in answers}
        rows = read_json(folder / "queries/result.json").get("queries", [])
        positions = {row["query_id"]: index for index, row in enumerate(rows)}
        case_failure = read_json(folder / "failure.json") if (folder / "failure.json").exists() else {}
        for query in case.queries:
            raw = by_id.get(query.query_id, {})
            if not raw and case_failure:
                continue  # Construction failures leave queries unassessed.
            row = {
                **raw,
                "query_id": query.query_id,
                "question": query.question,
                "groundtruth": query.groundtruth.model_dump(mode="json"),
                "question_type": query.metadata.get("question_type"),
                "score": None,
                "judge_result": None,
            }
            if not isinstance(row.get("answer"), str) or not row["answer"].strip() or row.get("error") is not None:
                row["error"] = row.get("error") or "No successful answer (construction or execution failed)"
                row["failure_stage"] = "answer"
                row["failure_kind"] = (
                    failure_kind(row["exception"], "answer")
                    if row.get("exception")
                    else case_failure.get("failure_kind")
                    or ("candidate_failure" if raw and not raw.get("error") else failure_kind(None, "answer"))
                )
                row["infrastructure_error"] = row["failure_kind"] == "infra_error"
            elif judge is not None:
                try:
                    async with semaphore:
                        row["judge_result"], row["score"] = await judge.evaluate(dataset, query, row["answer"])
                except Exception as error:
                    row.update(
                        error=f"{type(error).__name__}: {error}",
                        judge_result=getattr(error, "result", None),
                        failure_stage="judge",
                        failure_kind="infra_error",
                        infrastructure_error=True,
                    )
            compact = compact_query(row)
            if query.query_id in positions:
                rows[positions[query.query_id]] = compact
            else:
                rows.append(compact)
            # Publish each score; a later judge failure cannot erase frozen answers.
            write_json(folder / "queries/result.json", {"queries": rows}, compact_leaf_containers=True)
            if row.get("error") is not None:
                write_json(folder / "queries" / query.query_id / "failure.json", compact, compact_leaf_containers=True)
                if spec.fail_fast:
                    raise RuntimeError(f"Query {case.case_id}/{query.query_id} failed: {row['error']}")
