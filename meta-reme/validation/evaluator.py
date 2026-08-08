"""Concurrent validation orchestration for prepared Meta-ReMe workspaces."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tarfile
import tempfile
import traceback
from typing import Any, Callable, Iterator
from uuid import uuid4

import yaml

from models import CaseSpec, DomainSpec, fingerprint, model_fingerprint, utc_now
from workspace import Workspace
from sandbox import DockerReMeSandboxFactory, SourceCandidate, SourceSnapshot

from .adapters import build_jobs, evaluation_queries
from .scheduler import QueryCasePlan, QueryLease, QueryScheduler

SANDBOX_ENV_NAMES = (
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_BACKEND",
    "LLM_MODEL_NAME",
    "BENCH_MODEL_NAME",
    "JUDGE_MODEL_NAME",
    "EMBEDDING_API_KEY",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_BACKEND",
    "EMBEDDING_MODEL_NAME",
)

FactoryBuilder = Callable[..., DockerReMeSandboxFactory]


class ValidationError(RuntimeError):
    """Raised when validation inputs or workspace state are invalid."""


def run_validation(
    workspace_root: Path,
    case_ids: list[str],
    code_id: str,
    concurrency: int,
    *,
    validation_id: str | None = None,
    factory_builder: FactoryBuilder = DockerReMeSandboxFactory,
    environment: dict[str, str] | None = None,
) -> Path:
    """Synchronously validate cases; use ``run_validation_async`` in async applications."""

    return asyncio.run(
        run_validation_async(
            workspace_root,
            case_ids,
            code_id,
            concurrency,
            validation_id=validation_id,
            factory_builder=factory_builder,
            environment=environment,
        ),
    )


async def run_validation_async(
    workspace_root: Path,
    case_ids: list[str],
    code_id: str,
    concurrency: int,
    *,
    validation_id: str | None = None,
    factory_builder: FactoryBuilder = DockerReMeSandboxFactory,
    environment: dict[str, str] | None = None,
) -> Path:
    """Asynchronously validate one immutable code revision against prepared cases."""

    if concurrency < 1:
        raise ValidationError("concurrency must be at least 1")
    if not case_ids:
        raise ValidationError("at least one case_id is required")
    if len(case_ids) != len(set(case_ids)):
        raise ValidationError("case_ids must be unique")
    _validate_code_id(code_id)

    root = Path(workspace_root).resolve()
    domain = _load_domain(root)
    workspace = Workspace.open(root, domain)
    cases = _load_cases(workspace, case_ids)
    repository = workspace.path("code/repo/reme")
    run_id = validation_id or uuid4().hex
    run_root = _validation_root(workspace, code_id, run_id)

    with workspace.acquire_lock():
        commit_sha = _resolve_branch_commit(repository, code_id)
        snapshot_context = _source_snapshot(repository, commit_sha)
        with snapshot_context as snapshot:
            return await _execute_validation(
                workspace,
                run_root,
                domain,
                cases,
                code_id,
                commit_sha,
                run_id,
                concurrency,
                factory_builder,
                environment,
                snapshot,
            )


async def _execute_validation(
    workspace: Workspace,
    run_root: Path,
    domain: DomainSpec,
    cases: list[CaseSpec],
    code_id: str,
    commit_sha: str,
    run_id: str,
    concurrency: int,
    factory_builder: FactoryBuilder,
    environment: dict[str, str] | None,
    snapshot: SourceSnapshot,
) -> Path:
    """Run validation after resolving and freezing the branch-backed code revision."""

    candidate = SourceCandidate(snapshot, base_image=domain.sandbox.image)
    env = dict(environment) if environment is not None else _sandbox_environment()
    fingerprints = {
        "dataset": _dataset_fingerprint(workspace),
        "code": snapshot.sha256,
        "config": model_fingerprint(domain),
        "model": fingerprint({name: env.get(name) for name in SANDBOX_ENV_NAMES if "MODEL" in name}),
        "image": fingerprint(domain.sandbox.image),
    }
    run_root.mkdir(parents=True, exist_ok=False)
    workspace.atomic_write_json(
        run_root.relative_to(workspace.root) / "manifest.json",
        {
            "schema_version": 1,
            "validation_id": run_id,
            "code_id": code_id,
            "commit_sha": commit_sha,
            "case_ids": [case.case_id for case in cases],
            "concurrency": concurrency,
            "container_reuse": True,
            "scheduling": {
                "construction_barrier": True,
                "query_unit": "single_query_lease",
                "workspace_affinity": "case_id",
            },
            "dataset": domain.dataset.name.value,
            "fingerprints": fingerprints,
            "started_at": utc_now().isoformat(),
        },
    )
    factory = factory_builder(
        candidate,
        env=env,
        config=f"{domain.bundle_target}.yaml",
        command_timeout=domain.sandbox.timeout_seconds,
    )
    results = await _run_cases(workspace, run_root, factory, domain, cases, concurrency)
    summary = _summarize(run_id, code_id, commit_sha, [case.case_id for case in cases], results, fingerprints)
    workspace.atomic_write_json(run_root.relative_to(workspace.root) / "summary.json", summary)
    return run_root


async def _run_cases(
    workspace: Workspace,
    run_root: Path,
    factory: DockerReMeSandboxFactory,
    domain: DomainSpec,
    cases: list[CaseSpec],
    concurrency: int,
) -> list[dict[str, Any]]:
    """Run strict construction and query phases with reusable worker sandboxes."""

    queue: asyncio.Queue[tuple[int, CaseSpec]] = asyncio.Queue()
    for index, case in enumerate(cases):
        queue.put_nowait((index, case))
    results: list[dict[str, Any] | None] = [None] * len(cases)
    plans: list[QueryCasePlan] = []
    workers = [_WorkerState(worker_id=index) for index in range(concurrency)]

    async def run_construction_worker(worker: _WorkerState) -> None:
        while True:
            try:
                index, case = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            result, plan = await _construct_case_with_retries(
                workspace,
                run_root,
                factory,
                domain,
                index,
                case,
                worker,
            )
            if result is not None:
                results[index] = result
            if plan is not None:
                plans.append(plan)

    try:
        construction_outcomes = await asyncio.gather(
            *(run_construction_worker(worker) for worker in workers),
            return_exceptions=True,
        )
        _raise_worker_failure(construction_outcomes)

        scheduler = QueryScheduler(plans, max_retries=domain.sandbox.max_retries)
        query_outcomes = await asyncio.gather(
            *(_run_query_worker(workspace, factory, domain, scheduler, worker) for worker in workers),
            return_exceptions=True,
        )
        _raise_worker_failure(query_outcomes)

        for plan in plans:
            context = plan.context
            assert isinstance(context, _ConstructedCase)
            results[plan.case_index] = _finalize_constructed_case(workspace, plan, context)
    finally:
        await asyncio.gather(
            *(_best_effort_close(worker.sandbox_case) for worker in workers if worker.sandbox_case is not None),
        )

    if any(result is None for result in results):
        raise RuntimeError("validation worker exited without publishing every case result")
    return [result for result in results if result is not None]


@dataclass
class _WorkerState:
    """Host-side state retained by one reusable sandbox worker."""

    worker_id: int
    sandbox_case: Any | None = None
    loaded_case_id: str | None = None


@dataclass(frozen=True)
class _ConstructedCase:
    """Artifacts needed to run and finalize one successfully built case."""

    case_spec: CaseSpec
    attempt_number: int
    attempt_root: Path
    build: dict[str, Any]
    memory_workspace: Path
    memory_sha256: str


def _raise_worker_failure(outcomes: list[Any]) -> None:
    failure = next((outcome for outcome in outcomes if isinstance(outcome, BaseException)), None)
    if failure is not None:
        raise failure


async def _construct_case_with_retries(
    workspace: Workspace,
    run_root: Path,
    factory: DockerReMeSandboxFactory,
    domain: DomainSpec,
    case_index: int,
    case: CaseSpec,
    worker: _WorkerState,
) -> tuple[dict[str, Any] | None, QueryCasePlan | None]:
    last_result: dict[str, Any] | None = None
    for attempt_number in range(1, domain.sandbox.max_retries + 2):
        result, context = await _construct_case(
            workspace,
            run_root,
            factory,
            domain,
            case,
            attempt_number,
            worker,
        )
        if context is not None:
            try:
                queries = tuple(evaluation_queries(domain.dataset.name, case))
            except Exception as exc:
                result = {
                    "case_id": case.case_id,
                    "attempt": context.attempt_number,
                    "status": "infra_error",
                    "stage": "schedule_queries",
                    "completed_stages": ["prepare", "construct_memory"],
                    "build": context.build,
                    "queries": [],
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "container_id": getattr(worker.sandbox_case, "container_id", None),
                    "artifact_sha256": {"memory_workspace": context.memory_sha256},
                }
                workspace.atomic_write_json(context.attempt_root.relative_to(workspace.root) / "failure.json", result)
                return _publish_case_result(workspace, context.attempt_root, result), None
            return None, QueryCasePlan(
                case_index=case_index,
                case_id=case.case_id,
                queries=queries,
                context=context,
            )
        assert result is not None
        last_result = result
        if result["status"] != "infra_error":
            return result, None
        await _retire_worker_sandbox(worker)
    assert last_result is not None
    return last_result, None


async def _construct_case(
    workspace: Workspace,
    run_root: Path,
    factory: DockerReMeSandboxFactory,
    domain: DomainSpec,
    case_spec: CaseSpec,
    attempt_number: int,
    worker: _WorkerState,
) -> tuple[dict[str, Any] | None, _ConstructedCase | None]:
    case_root = workspace.entity_path((run_root / "cases").relative_to(workspace.root), case_spec.case_id)
    attempt_root = workspace.entity_path(case_root.relative_to(workspace.root), f"attempt-{attempt_number}")
    attempt_root.mkdir(parents=True, exist_ok=False)
    completed_stages: list[str] = []
    build: dict[str, Any] | None = None
    try:
        if worker.sandbox_case is None:
            worker.sandbox_case = await factory.create_case(case_spec.case_id)
        else:
            await worker.sandbox_case.reset_case(case_spec.case_id)
        worker.loaded_case_id = None
        completed_stages.append("prepare")
        build = await worker.sandbox_case.run_build(build_jobs(domain.dataset.name, case_spec))
        completed_stages.append("construct_memory")
        memory_root = attempt_root / "memory_construction"
        memory_root.mkdir(parents=False, exist_ok=False)
        workspace.atomic_write_json(memory_root.relative_to(workspace.root) / "result.json", build)
        memory_workspace = await worker.sandbox_case.export_workspace(memory_root / "reme_workspace.tar.gz")
        _extract_artifacts(
            memory_workspace,
            memory_root / "reme_workspace",
            domain.sandbox.max_artifact_bytes,
        )
        await worker.sandbox_case.export_build_log(memory_root / "build.log")
        memory_sha256 = _sha256_file(memory_workspace)
        if not build.get("success"):
            result = {
                "case_id": case_spec.case_id,
                "attempt": attempt_number,
                "status": "candidate_failure",
                "completed_stages": completed_stages,
                "build": build,
                "queries": [],
                "error": "memory construction failed",
                "artifact_sha256": {
                    "memory_workspace": memory_sha256,
                },
            }
            await _best_effort_export(worker.sandbox_case, attempt_root / "full.tar.gz")
            return _publish_case_result(workspace, attempt_root, result), None

        worker.loaded_case_id = case_spec.case_id
        return None, _ConstructedCase(
            case_spec=case_spec,
            attempt_number=attempt_number,
            attempt_root=attempt_root,
            build=build,
            memory_workspace=memory_workspace,
            memory_sha256=memory_sha256,
        )
    except Exception as exc:  # A failed container/worker must not cancel sibling cases.
        if worker.sandbox_case is not None:
            await _best_effort_export(worker.sandbox_case, attempt_root / "full.tar.gz")
        result = {
            "case_id": case_spec.case_id,
            "attempt": attempt_number,
            "status": "infra_error",
            "stage": "construction",
            "completed_stages": completed_stages,
            "build": build,
            "queries": [],
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "container_id": getattr(worker.sandbox_case, "container_id", None),
        }
        workspace.atomic_write_json(attempt_root.relative_to(workspace.root) / "failure.json", result)
        return _publish_case_result(workspace, attempt_root, result), None


async def _run_query_worker(
    workspace: Workspace,
    factory: DockerReMeSandboxFactory,
    domain: DomainSpec,
    scheduler: QueryScheduler,
    worker: _WorkerState,
) -> None:
    while True:
        lease = await scheduler.claim(worker.worker_id, worker.loaded_case_id)
        if lease is None:
            return
        context = lease.plan.context
        assert isinstance(context, _ConstructedCase)
        query_id = str(getattr(lease.query, "query_id", ""))
        try:
            await _load_query_workspace(factory, worker, lease.plan, context)
            result = await worker.sandbox_case.run_query(lease.query)
            await _export_query_artifacts(workspace, domain, worker.sandbox_case, context, query_id, lease)
            await scheduler.complete(lease, result)
        except Exception as exc:  # Retry only infrastructure failures around the structured query operation.
            failure = {
                "query_id": query_id,
                "question": getattr(lease.query, "question", None),
                "golden_answer": getattr(lease.query, "golden_answer", None),
                "answer": None,
                "score": None,
                "answer_result": None,
                "judge_result": None,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "infrastructure_error": True,
                "attempt": lease.attempt,
                "lease_token": lease.token,
                "worker_id": worker.worker_id,
                "container_id": getattr(worker.sandbox_case, "container_id", None),
            }
            query_attempt_root = _write_query_failure(workspace, context, query_id, lease.attempt, failure)
            if worker.sandbox_case is not None:
                await _best_effort_export(worker.sandbox_case, query_attempt_root / "full.tar.gz")
            await scheduler.fail(lease, failure)
            await _retire_worker_sandbox(worker)


async def _load_query_workspace(
    factory: DockerReMeSandboxFactory,
    worker: _WorkerState,
    plan: QueryCasePlan,
    context: _ConstructedCase,
) -> None:
    if worker.sandbox_case is not None and worker.loaded_case_id == plan.case_id:
        return
    if worker.sandbox_case is None:
        worker.sandbox_case = await factory.create_case(plan.case_id)
    else:
        await worker.sandbox_case.reset_case(plan.case_id)
    worker.loaded_case_id = None
    await worker.sandbox_case.upload_workspace(context.memory_workspace)
    worker.loaded_case_id = plan.case_id


async def _export_query_artifacts(
    workspace: Workspace,
    domain: DomainSpec,
    sandbox_case: Any,
    context: _ConstructedCase,
    query_id: str,
    lease: QueryLease,
) -> None:
    queries_root = context.attempt_root / "queries"
    queries_root.mkdir(parents=True, exist_ok=True)
    archive = queries_root / f".{query_id}.{lease.token}.tar.gz"
    try:
        await sandbox_case.export_query(query_id, archive)
        with tempfile.TemporaryDirectory(prefix=".query-export-", dir=queries_root) as temporary:
            temporary_root = Path(temporary)
            _extract_artifacts(
                archive,
                temporary_root,
                domain.sandbox.max_artifact_bytes,
                strip_prefix=PurePosixPath("queries"),
            )
            source = workspace.entity_path(temporary_root.relative_to(workspace.root), query_id)
            target = workspace.entity_path(queries_root.relative_to(workspace.root), query_id)
            if not source.is_dir():
                raise ValidationError(f"query archive is missing artifact directory: {query_id!r}")
            if target.exists():
                raise ValidationError(f"query artifact directory already exists: {query_id!r}")
            source.replace(target)
    finally:
        archive.unlink(missing_ok=True)


def _query_attempt_root(
    workspace: Workspace,
    context: _ConstructedCase,
    query_id: str,
    attempt: int,
) -> Path:
    attempts_root = context.attempt_root / "queries" / ".attempts"
    query_root = workspace.entity_path(attempts_root.relative_to(workspace.root), query_id)
    attempt_root = workspace.entity_path(query_root.relative_to(workspace.root), f"attempt-{attempt}")
    attempt_root.mkdir(parents=True, exist_ok=True)
    return attempt_root


def _write_query_failure(
    workspace: Workspace,
    context: _ConstructedCase,
    query_id: str,
    attempt: int,
    failure: dict[str, Any],
) -> Path:
    attempt_root = _query_attempt_root(workspace, context, query_id, attempt)
    workspace.atomic_write_json(attempt_root.relative_to(workspace.root) / "failure.json", failure)
    return attempt_root


def _finalize_constructed_case(
    workspace: Workspace,
    plan: QueryCasePlan,
    context: _ConstructedCase,
) -> dict[str, Any]:
    if any(result is None for result in plan.results):
        raise RuntimeError(f"case {plan.case_id!r} has unpublished query results")
    query_results = [result for result in plan.results if result is not None]
    scores = [float(result["score"]) for result in query_results if result.get("score") is not None]
    query_summary = {
        "schema_version": 1,
        "case_id": plan.case_id,
        "query_count": len(query_results),
        "scored_count": len(scores),
        "mean_score": sum(scores) / len(scores) if scores else None,
        "queries": [
            {
                "query_id": result.get("query_id"),
                "score": result.get("score"),
                **({"error": result["error"]} if result.get("error") is not None else {}),
            }
            for result in query_results
        ],
    }
    query_payload = {
        "success": all(result.get("error") is None for result in query_results),
        "summary": query_summary,
        "queries": query_results,
    }
    queries_root = context.attempt_root / "queries"
    queries_root.mkdir(parents=True, exist_ok=True)
    workspace.atomic_write_json(queries_root.relative_to(workspace.root) / "result.json", query_payload)

    has_infra_error = any(result.get("infrastructure_error") for result in query_results)
    result = {
        "case_id": plan.case_id,
        "attempt": context.attempt_number,
        "status": "infra_error" if has_infra_error else "completed",
        "completed_stages": ["prepare", "construct_memory", "test", *([] if has_infra_error else ["export"])],
        "build": context.build,
        "queries": query_results,
        "query_summary": query_summary,
        "error": "one or more query infrastructure attempts were exhausted" if has_infra_error else None,
        "artifact_sha256": {"memory_workspace": context.memory_sha256},
    }
    if has_infra_error:
        workspace.atomic_write_json(context.attempt_root.relative_to(workspace.root) / "failure.json", result)
    return _publish_case_result(workspace, context.attempt_root, result)


async def _retire_worker_sandbox(worker: _WorkerState) -> None:
    if worker.sandbox_case is not None:
        await _best_effort_close(worker.sandbox_case)
    worker.sandbox_case = None
    worker.loaded_case_id = None


async def _best_effort_export(sandbox_case: Any, destination: Path) -> None:
    try:
        await sandbox_case.export_full(destination)
    except Exception:
        return


async def _best_effort_close(sandbox_case: Any) -> None:
    try:
        await sandbox_case.close()
    except Exception:
        return


def _publish_case_result(workspace: Workspace, attempt_root: Path, result: dict[str, Any]) -> dict[str, Any]:
    workspace.atomic_write_json(attempt_root.relative_to(workspace.root) / "case_result.json", result)
    return result


def _summarize(
    validation_id: str,
    code_id: str,
    commit_sha: str,
    requested_case_ids: list[str],
    results: list[dict[str, Any]],
    fingerprints: dict[str, str],
) -> dict[str, Any]:
    query_results = [query for result in results for query in result.get("queries", [])]
    scores = [float(query["score"]) if query.get("score") is not None else 0.0 for query in query_results]
    infra_errors = sum(result["status"] == "infra_error" for result in results)
    candidate_failures = sum(result["status"] == "candidate_failure" for result in results)
    return {
        "schema_version": 1,
        "validation_id": validation_id,
        "code_id": code_id,
        "commit_sha": commit_sha,
        "status": "infra_error" if infra_errors else "completed",
        "case_ids": requested_case_ids,
        "case_count": len(results),
        "query_count": len(query_results),
        "mean_query_score": sum(scores) / len(scores) if scores else None,
        "infra_error_count": infra_errors,
        "candidate_failure_count": candidate_failures,
        "fingerprints": fingerprints,
        "cases": results,
        "completed_at": utc_now().isoformat(),
    }


def _load_domain(root: Path) -> DomainSpec:
    path = root / "domain_spec.yaml"
    try:
        return DomainSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise ValidationError(f"cannot load workspace domain spec: {exc}") from exc


def _load_cases(workspace: Workspace, requested_ids: list[str]) -> list[CaseSpec]:
    found: dict[str, CaseSpec] = {}
    for path in sorted(workspace.path("dataset/cases").glob("*.json")):
        case = CaseSpec.model_validate_json(path.read_text(encoding="utf-8"))
        if case.case_id in requested_ids:
            if case.case_id in found:
                raise ValidationError(f"duplicate prepared case_id: {case.case_id}")
            found[case.case_id] = case
    unknown = [case_id for case_id in requested_ids if case_id not in found]
    if unknown:
        raise ValidationError(f"unknown prepared case_ids: {', '.join(unknown)}")
    return [found[case_id] for case_id in requested_ids]


def _resolve_branch_commit(repository: Path, code_id: str) -> str:
    _validate_code_id(code_id)
    branch_check = subprocess.run(
        ["git", "check-ref-format", "--branch", code_id],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if branch_check.returncode:
        raise ValidationError(
            f"invalid Git branch code_id {code_id!r}: {(branch_check.stderr or branch_check.stdout).strip()}",
        )
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--end-of-options", f"refs/heads/{code_id}^{{commit}}"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise ValidationError(f"unknown local branch code_id {code_id!r}: {(result.stderr or result.stdout).strip()}")
    return result.stdout.strip()


def _validate_code_id(code_id: str) -> None:
    if (
        not code_id
        or code_id in {".", ".."}
        or Path(code_id).name != code_id
        or any(character in code_id for character in ("/", "\\", "\x00"))
    ):
        raise ValidationError(f"code_id must be a path-safe Git branch name: {code_id!r}")


@contextmanager
def _source_snapshot(repository: Path, commit_sha: str) -> Iterator[SourceSnapshot]:
    with tempfile.TemporaryDirectory(prefix="meta-reme-validation-code-") as temporary:
        destination = Path(temporary) / "source"
        destination.mkdir()
        archive = subprocess.run(
            ["git", "archive", "--format=tar", commit_sha],
            cwd=repository,
            capture_output=True,
            check=False,
        )
        if archive.returncode:
            raise ValidationError(f"cannot archive code commit {commit_sha}: {archive.stderr.decode(errors='replace')}")
        archive_path = Path(temporary) / "source.tar"
        archive_path.write_bytes(archive.stdout)
        _extract_source_archive(archive_path, destination)
        yield SourceSnapshot.from_directory(destination)


def _extract_source_archive(archive_path: Path, destination: Path) -> None:
    seen: set[str] = set()
    with tarfile.open(archive_path, mode="r:") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.name in seen:
                raise ValidationError(f"unsafe path in code archive: {member.name!r}")
            seen.add(member.name)
            target = destination.joinpath(*path.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValidationError(f"unsupported entry in code archive: {member.name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValidationError(f"cannot read code archive entry: {member.name!r}")
            with target.open("wb") as stream:
                shutil.copyfileobj(source, stream)
            target.chmod(member.mode & 0o777)


def _extract_artifacts(
    archive_path: Path,
    destination: Path,
    max_bytes: int,
    *,
    strip_prefix: PurePosixPath | None = None,
    excluded_paths: set[PurePosixPath] | None = None,
) -> None:
    _validate_archive_size(archive_path, max_bytes)
    destination.mkdir(parents=True, exist_ok=True)
    total = 0
    seen: set[PurePosixPath] = set()
    excluded = excluded_paths or set()
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValidationError(f"unsafe artifact path: {member.name!r}")
            if member.issym() or member.islnk() or member.isdev():
                raise ValidationError(f"unsupported artifact entry: {member.name!r}")
            total += member.size
            if total > max_bytes:
                raise ValidationError(f"expanded artifacts exceed {max_bytes} bytes")
            if strip_prefix is not None:
                if path == strip_prefix:
                    continue
                try:
                    path = path.relative_to(strip_prefix)
                except ValueError as exc:
                    raise ValidationError(f"artifact is outside {strip_prefix}: {member.name!r}") from exc
            if not path.parts or path in excluded:
                continue
            if path in seen:
                raise ValidationError(f"duplicate artifact path: {member.name!r}")
            seen.add(path)
            target = destination.joinpath(*path.parts)
            if target.exists():
                raise ValidationError(f"artifact path already exists: {member.name!r}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValidationError(f"cannot read artifact entry: {member.name!r}")
                with target.open("wb") as stream:
                    shutil.copyfileobj(source, stream)
                target.chmod(member.mode & (stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO))
            else:
                raise ValidationError(f"unsupported artifact entry: {member.name!r}")


def _validate_archive_size(archive_path: Path, max_bytes: int) -> None:
    if archive_path.stat().st_size > max_bytes:
        raise ValidationError(f"artifact archive exceeds {max_bytes} bytes")


def _validation_root(workspace: Workspace, code_id: str, validation_id: str) -> Path:
    if not validation_id or validation_id in {".", ".."} or Path(validation_id).name != validation_id:
        raise ValidationError(f"unsafe validation_id: {validation_id!r}")
    return workspace.entity_path(
        workspace.entity_path("evaluations", code_id).relative_to(workspace.root),
        validation_id,
    )


def _dataset_fingerprint(workspace: Workspace) -> str:
    manifest = json.loads(workspace.path("dataset/manifest.json").read_text(encoding="utf-8"))
    value = manifest.get("normalized_fingerprint")
    if not isinstance(value, str) or not value:
        raise ValidationError("dataset manifest has no normalized_fingerprint")
    return value


def _sandbox_environment() -> dict[str, str]:
    return {name: os.environ[name] for name in SANDBOX_ENV_NAMES if os.environ.get(name)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
