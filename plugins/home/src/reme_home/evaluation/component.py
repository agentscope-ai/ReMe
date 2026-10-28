"""Home evaluation orchestration using the shared Docker pool and a host judge."""

import asyncio
import os
import shutil
import traceback

from deploy.benchmark_docker import BenchmarkDockerPool, CasePlan, DockerReMeSandboxFactory, SourceCandidate

from reme.components.base_component import BaseComponent

from ..preparation.data.models import fingerprint, utc_now
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
from .inputs import EvaluationSpec, load_inputs, reuse_inputs, snapshot_code
from .compaction import compact_construction, compact_query
from ..judge import HomeJudge
from ..workspace import HomeWorkspace, _initialization_lock, _validate_directory


async def blocking(function, *args, **kwargs):
    """Await owned filesystem work even when the calling job is cancelled."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.gather(task, return_exceptions=True)
        raise


class HomeEvaluation(BaseComponent):
    """Own evaluation tasks; the application owns the host judge Job lifecycle."""

    component_type = "home.evaluation"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.workspace = self.bind("default", HomeWorkspace, optional=False)
        self.judge = self.bind("default", HomeJudge, optional=False)
        self._pending: set[asyncio.Task] = set()

    async def _close(self):
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)

    async def evaluate(self, **kwargs) -> dict:
        """Run through the component lifecycle, including cancellation cleanup."""
        if not self.is_started:
            raise RuntimeError("Start the Home application before evaluation")
        spec = EvaluationSpec.model_validate(kwargs)
        task = asyncio.create_task(self._evaluate(spec))
        self._pending.add(task)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        finally:
            self._pending.discard(task)

    def _prepare(self, spec, environment):
        home = self.workspace.home_path
        _validate_directory(self.workspace_path.resolve(), self.workspace.state_path)
        with _initialization_lock(self.workspace.state_path / "workspace.lock"):
            record, dataset, cases = load_inputs(home, spec)
            judge_identity = self.judge.identity(dataset.dataset)
            snapshot, commit, branch = snapshot_code(home, record)
            fingerprints = {
                "dataset": dataset.normalized_fingerprint,
                "code": snapshot.sha256,
                "trusted_judge": judge_identity["fingerprint"],
                "construction_config": fingerprint(
                    {
                        "runtime": record["code"]["runtime_fingerprint"],
                        "config": record["code"]["config"],
                        "plugins": record["code"]["plugins"],
                        "image": spec.base_image,
                        "models": {
                            k: v for k, v in environment.items() if "MODEL" in k and "KEY" not in k and "JUDGE" not in k
                        },
                    },
                ),
            }
            source, archives = reuse_inputs(home, spec, cases, fingerprints)
            # Validate adapters before allocating results or starting containers.
            jobs = {case.case_id: build_jobs(dataset.dataset, case) for case in cases} if source is None else {}
            for case in cases:
                for query in case.queries:
                    if dataset.dataset == "beam" and spec.dataset_kind == "training-set":
                        rubric = query.metadata.get("rubric", query.golden_answer)
                        if not isinstance(rubric, list) or not rubric:
                            raise ValueError(f"BEAM query {query.query_id} has no scoring rubric")
                    evaluation_query(dataset.dataset, query)
            return record, dataset, cases, snapshot, commit, branch, fingerprints, source, archives, jobs

    def _allocate(self, branch, commit, manifest):
        """Allocate ascending IDs under a collision-safe commit prefix."""
        home = self.workspace.home_path
        with _initialization_lock(self.workspace.state_path / "workspace.lock"):
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

    async def _evaluate(self, spec):
        environment = {**os.environ, **self.app_context.app_config.environment}
        prepared = await blocking(self._prepare, spec, environment)
        record, dataset, cases, snapshot, commit, branch, fingerprints, source, archives, jobs = prepared
        judge = None if spec.debug else self.judge
        run = None
        try:
            manifest = {
                "schema_version": 1,
                "branch_name": branch,
                "commit_sha": commit,
                "case_ids": [case.case_id for case in cases],
                "query_ids": {case.case_id: [q.query_id for q in case.queries] for case in cases},
                "dataset": dataset.dataset,
                "dataset_kind": spec.dataset_kind,
                "mode": "debug" if judge is None else "scored",
                "execution_mode": "query_only_replay" if source else "full_validation",
                "construction_source": source,
                "fingerprints": fingerprints,
                "options": spec.model_dump(mode="json", exclude={"case_ids", "query_ids", "reuse_construction_from"}),
                "judge": self.judge.identity(dataset.dataset),
                "status": "running",
                "started_at": utc_now().isoformat(),
            }
            run = self._allocate(branch, commit, manifest)
            plans = []
            for case in cases:
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
                        queries=tuple(evaluation_query(dataset.dataset, q) for q in case.queries),
                        workspace_source=memory / "reme_workspace.tar.gz" if source else None,
                        export_workspace_to=None if source else memory / "reme_workspace.tar.gz",
                        export_build_log_to=None if source else memory / "build.log",
                        export_evaluation_to=folder / "evaluation.tar.gz",
                        export_diagnostics_to=folder / "full.tar.gz",
                    ),
                )
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
            )
            semaphore = asyncio.Semaphore(spec.concurrency)
            by_id = {case.case_id: case for case in cases}

            async def completed(plan, result):
                try:
                    await self._persist_case(run, by_id[plan.case_id], result, source, spec)
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
            manifest.update(status="completed", finished_at=utc_now().isoformat())
            write_json(run / "manifest.json", manifest)
            write_json(run / "summary.json", summary, compact_leaf_containers=True)
            return {
                "result_dir": str(run),
                "commit_sha": commit,
                "execution_mode": manifest["execution_mode"],
                "status": "completed",
                **{key: value for key, value in summary.items() if key != "cases"},
            }
        except BaseException as error:
            if run is not None:
                manifest.update(
                    status="cancelled" if isinstance(error, asyncio.CancelledError) else "failed",
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
                    "golden_answer": query.golden_answer,
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
                "golden_answer": query.golden_answer,
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
