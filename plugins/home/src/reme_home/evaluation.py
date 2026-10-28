"""Home evaluation orchestration using the shared Docker pool and a host judge."""

import asyncio
from dataclasses import asdict
import os
import shutil
import traceback

from deploy.benchmark_docker import BenchmarkDockerPool, CasePlan, DockerReMeSandboxFactory, SourceCandidate

from reme.components.base_component import BaseComponent

from .data_preparation.models import fingerprint, utc_now
from .evaluation_adapters import accepts_build, build_jobs, evaluation_query
from .evaluation_artifacts import contained, file_sha256, inspect_archive, read_json, summarize, write_json
from .evaluation_inputs import load_inputs, reuse_inputs, snapshot_code
from .evaluation_failures import construction_exception, failure_kind
from .evaluation_judge import HomeJudge
from .evaluation_spec import EvaluationSpec
from .workspace import HomeWorkspace, _initialization_lock, _validate_directory


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
        judge = self.judge if spec.dataset_kind == "training-set" else None
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
                "options": spec.model_dump(mode="json"),
                "judge": self.judge.identity(dataset.dataset),
                "judge_model": self.judge.identity(dataset.dataset)["model"],
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
                await self._score_case(run, by_id[plan.case_id], dataset.dataset, judge, semaphore, spec)

            results = await BenchmarkDockerPool(factory, concurrency=spec.concurrency).run(
                plans,
                on_case_finished=completed,
                fail_fast=spec.fail_fast,
                accept_build=lambda build: accepts_build(build, spec.max_construction_failure_rate),
            )
            # Pool teardown can attach late cleanup failures after callbacks.
            for result in results:
                execution = asdict(result)
                execution["workspace_snapshot"] = str(result.workspace_snapshot) if result.workspace_snapshot else None
                write_json(run / "cases" / result.case_id / "execution.json", execution)
                if result.error:
                    failure = run / "cases" / result.case_id / "failure.json"
                    if not failure.exists():
                        write_json(failure, {"stage": "cleanup", "error": result.error, "failure_kind": "infra_error"})
                    else:
                        previous = read_json(failure)
                        if previous.get("error") != result.error:
                            previous.update(cleanup_error=result.error, failure_kind="infra_error")
                            write_json(failure, previous)
            summary = summarize(run, cases, status="completed")
            summary.update(
                execution_complete=True,
                execution_mode=manifest["execution_mode"],
                scoring_required=judge is not None,
            )
            if judge is None:
                summary["valid_for_comparison"] = False
            manifest.update(status="completed", finished_at=utc_now().isoformat())
            write_json(run / "manifest.json", manifest)
            write_json(run / "summary.json", summary)
            return {
                "result_dir": str(run),
                "commit_sha": commit,
                "execution_mode": manifest["execution_mode"],
                **{key: value for key, value in summary.items() if key != "cases"},
            }
        except BaseException as error:
            if run is not None:
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
                    status="cancelled" if isinstance(error, asyncio.CancelledError) else "failed",
                )
                partial["execution_complete"] = False
                write_json(run / "partial_summary.json", partial)
            if isinstance(error, asyncio.CancelledError) or run is None:
                raise
            raise RuntimeError(f"Evaluation failed; artifacts: {run}: {error}") from error

    @staticmethod
    async def _persist_case(run, case, result, source, spec):
        folder = run / "cases" / case.case_id
        memory = folder / "memory_construction"
        execution = asdict(result)
        execution["workspace_snapshot"] = str(result.workspace_snapshot) if result.workspace_snapshot else None
        write_json(folder / "execution.json", execution)
        write_json(folder / "queries/answers.json", result.queries or {"queries": []})
        if result.error:
            stage = result.failure_stage or "execution"
            exception = result.exception or construction_exception(result.build or {})
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
            write_json(memory / "result.json", construction)
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
            # Do not merge the container's queries tree: it could overwrite
            # host-frozen answers.json or host-only judge results. The complete
            # untrusted tree remains available in evaluation.tar.gz for inspection.
            await blocking(shutil.rmtree, extracted)
        integrity = {}
        for relative in ("memory_construction/reme_workspace.tar.gz", "evaluation.tar.gz", "full.tar.gz"):
            artifact = folder / relative
            if artifact.is_file():
                integrity[relative] = await blocking(file_sha256, artifact)
        write_json(folder / "artifacts.json", {"sha256": integrity})

    @staticmethod
    async def _score_case(run, case, dataset, judge, semaphore, spec):
        folder = run / "cases" / case.case_id
        answers = read_json(folder / "queries/answers.json").get("queries", [])
        by_id = {row.get("query_id"): row for row in answers}
        rows = []
        for query in case.queries:
            raw = by_id.get(query.query_id, {})
            row = {
                **raw,
                "query_id": query.query_id,
                "question": query.question,
                "golden_answer": query.golden_answer,
                "question_type": query.metadata.get("question_type"),
                "score": None,
                "judge_result": None,
            }
            if not isinstance(row.get("answer"), str) or not row["answer"].strip() or row.get("error"):
                row["error"] = row.get("error") or "No successful answer (construction or execution failed)"
                row["failure_stage"] = "answer"
                case_failure = read_json(folder / "failure.json") if (folder / "failure.json").exists() else {}
                row["failure_kind"] = (
                    failure_kind(row["exception"], "answer")
                    if row.get("exception")
                    else case_failure.get("failure_kind") or failure_kind(None, "answer")
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
            rows.append(row)
            # Publish each score; a later judge failure cannot erase frozen answers.
            write_json(folder / "queries/result.json", {"queries": rows})
            if row.get("error"):
                write_json(folder / "queries" / query.query_id / "failure.json", row)
                if spec.fail_fast:
                    raise RuntimeError(f"Query {case.case_id}/{query.query_id} failed: {row['error']}")
