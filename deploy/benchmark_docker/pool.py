"""Bounded Docker workers for independent benchmark cases.

Each case completes construction and its queries before its worker is assigned
another case. A dependent case may reuse the workspace captured immediately
after its parent's construction, without sharing mutable container state.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ._cleanup import finish_cleanup, format_error
from .docker_sandbox import DockerReMeSandbox, DockerReMeSandboxFactory
from .models import EvaluationQuery, JobRequest


@dataclass(frozen=True, slots=True)
class CasePlan:
    """One case, optionally seeded by a host workspace or another case."""

    case_id: str
    build_jobs: tuple[JobRequest, ...] = ()
    queries: tuple[EvaluationQuery, ...] = ()
    workspace_source: str | Path | None = None
    reuse_workspace_from: str | None = None
    export_workspace_to: str | Path | None = None
    export_build_log_to: str | Path | None = None
    export_evaluation_to: str | Path | None = None


@dataclass(slots=True)
class CaseResult:
    """Structured build and answer results; judging stays on the host."""

    case_id: str
    build: dict[str, Any] | None = None
    queries: dict[str, Any] | None = None
    workspace_snapshot: Path | None = None
    build_log: str | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        """Whether all requested phases and owned-worker cleanup succeeded."""
        return (
            self.error is None
            and (self.build is None or self.build.get("success") is True)
            and (self.queries is None or self.queries.get("success") is True)
        )


class BenchmarkDockerPool:
    """Reuse at most ``concurrency`` containers across case plans.

    ``run`` owns and closes its containers. It never stages all construction
    phases ahead of queries: each available worker runs build then queries for
    one case. Dependencies await a completed parent without holding a worker.
    """

    def __init__(self, factory: DockerReMeSandboxFactory, *, concurrency: int = 4) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self.factory = factory
        self.concurrency = concurrency

    async def run(self, plans: list[CasePlan]) -> list[CaseResult]:
        """Return results in input order, closing every container on exit."""
        if not plans:
            return []
        by_id = {plan.case_id: plan for plan in plans}
        if len(by_id) != len(plans):
            raise ValueError("case IDs must be unique")
        for plan in plans:
            if plan.workspace_source is not None and plan.reuse_workspace_from is not None:
                raise ValueError(f"case {plan.case_id!r} has two workspace sources")
            if not plan.build_jobs and plan.workspace_source is None and plan.reuse_workspace_from is None:
                raise ValueError(f"case {plan.case_id!r} needs build jobs or a workspace source")
            if plan.reuse_workspace_from is not None and plan.reuse_workspace_from not in by_id:
                raise ValueError(f"unknown workspace source: {plan.reuse_workspace_from!r}")
            if plan.export_build_log_to is not None and not plan.build_jobs:
                raise ValueError("build log export requires build jobs")
            if plan.export_evaluation_to is not None and not plan.queries:
                raise ValueError("evaluation export requires queries")
        self._check_acyclic(by_id)
        reused_case_ids = {plan.reuse_workspace_from for plan in plans if plan.reuse_workspace_from is not None}

        workers = await self.factory.create_cases(
            [f"pool-{index}" for index in range(min(self.concurrency, len(plans)))],
            concurrency=self.concurrency,
        )
        available: asyncio.Queue[DockerReMeSandbox | None] = asyncio.Queue()
        for worker in workers:
            available.put_nowait(worker)
        unclaimed = set(by_id)
        live_slots = len(workers)
        next_worker_index = len(workers)
        exhaustion_error = "no available workers"
        worker_results: dict[int, list[CaseResult]] = {}
        results: list[CaseResult] = []
        run_error: BaseException | None = None

        try:
            with TemporaryDirectory(prefix="reme-benchmark-pool-") as temporary:
                snapshot_root = Path(temporary)
                # Tasks are started together; awaiting a parent never consumes
                # a container. The graph check above rules out deadlocks.
                tasks: dict[str, asyncio.Task[CaseResult]] = {}

                async def execute(plan: CasePlan) -> CaseResult:
                    nonlocal live_slots, next_worker_index, exhaustion_error
                    result = CaseResult(plan.case_id)
                    source = plan.workspace_source
                    if plan.reuse_workspace_from is not None:
                        parent = await tasks[plan.reuse_workspace_from]
                        if parent.workspace_snapshot is None:
                            unclaimed.discard(plan.case_id)
                            return CaseResult(plan.case_id, error=f"source case {parent.case_id!r} failed")
                        source = parent.workspace_snapshot

                    worker = await available.get()
                    unclaimed.discard(plan.case_id)
                    if worker is None:
                        return CaseResult(plan.case_id, error=exhaustion_error)
                    worker_results.setdefault(id(worker), []).append(result)

                    def record_lifecycle_error(phase: str, exc: Exception) -> str:
                        detail = f"{phase} failed: {format_error(exc)}"
                        result.error = f"{result.error}; {detail}" if result.error else detail
                        return detail

                    try:
                        await worker.reset_case(plan.case_id)
                        if source is not None:
                            await worker.upload_workspace(source)
                        if plan.build_jobs:
                            try:
                                result.build = await worker.run_build(list(plan.build_jobs))
                            except Exception as exc:
                                result.error = format_error(exc)
                                await self._retain_failed_build_log(worker, plan, result)
                                return result
                            if not result.build.get("success"):
                                result.error = "build failed"
                                await self._retain_failed_build_log(worker, plan, result)
                                return result
                            if plan.export_build_log_to is not None:
                                await worker.export_build_log(plan.export_build_log_to)
                        # Capture before queries, since answering may change the
                        # workspace. Dependents receive construction state.
                        if plan.export_workspace_to is not None or plan.case_id in reused_case_ids:
                            destination = plan.export_workspace_to or snapshot_root / f"{plan.case_id}.tar.gz"
                            result.workspace_snapshot = await worker.export_workspace(destination)
                        if plan.queries:
                            result.queries = await worker.run_queries(list(plan.queries))
                            if not result.queries.get("success"):
                                result.error = "query failed"
                        if plan.export_evaluation_to is not None:
                            await worker.export_evaluation(plan.export_evaluation_to)
                        return result
                    except Exception as exc:  # Keep other cases running and release this worker.
                        result.error = format_error(exc)
                        return result
                    finally:
                        if worker.reusable:
                            available.put_nowait(worker)
                        else:
                            # Confirm cleanup (or retry a failed removal) before
                            # allowing another case into this slot.
                            closed = False
                            try:
                                await worker.close()
                            except Exception as exc:
                                exhaustion_error = (
                                    f"no available workers: {record_lifecycle_error('worker close', exc)}"
                                )
                            else:
                                closed = True

                            replaced = False
                            if closed and unclaimed and not asyncio.current_task().cancelling():
                                worker_index = next_worker_index
                                next_worker_index += 1
                                try:
                                    replacement = await self.factory.create_case(f"pool-{worker_index}")
                                except Exception as exc:
                                    exhaustion_error = (
                                        f"no available workers: {record_lifecycle_error('replacement', exc)}"
                                    )
                                else:
                                    workers.append(replacement)
                                    available.put_nowait(replacement)
                                    replaced = True
                            if not replaced:
                                live_slots -= 1
                                if live_slots == 0:
                                    # Every future case, including one waiting on a
                                    # parent, must be able to leave available.get().
                                    for _ in unclaimed:
                                        available.put_nowait(None)

                for plan in plans:
                    tasks[plan.case_id] = asyncio.create_task(execute(plan))
                try:
                    results = await asyncio.gather(*(tasks[plan.case_id] for plan in plans))
                finally:
                    for task in tasks.values():
                        if not task.done() and not task.cancelling():
                            task.cancel()
                    await asyncio.gather(*tasks.values(), return_exceptions=True)
                for plan, result in zip(plans, results):
                    if plan.export_workspace_to is None:
                        result.workspace_snapshot = None  # The internal temporary archive is about to be deleted.
                return results
        except BaseException as exc:
            run_error = exc
            raise
        finally:
            await self._close_workers(workers, worker_results, results, run_error)

    @staticmethod
    async def _retain_failed_build_log(worker: DockerReMeSandbox, plan: CasePlan, result: CaseResult) -> None:
        """Keep diagnostics without replacing the original build failure."""
        try:
            result.build_log = await worker.read_build_log()
        except Exception as exc:
            result.error += f"; build log unavailable: {type(exc).__name__}: {exc}"
        if plan.export_build_log_to is not None:
            try:
                await worker.export_build_log(plan.export_build_log_to)
            except Exception as exc:
                result.error += f"; build log export failed: {type(exc).__name__}: {exc}"

    @staticmethod
    async def _close_workers(
        workers: list[DockerReMeSandbox],
        worker_results: dict[int, list[CaseResult]],
        results: list[CaseResult],
        run_error: BaseException | None,
    ) -> None:
        """Finish all teardown and attach failures to results or the run error."""
        outcomes, cleanup_cancel = await finish_cleanup(worker.close() for worker in workers)
        for worker, outcome in zip(workers, outcomes):
            if isinstance(outcome, BaseException):
                detail = f"worker {worker.container_id!r} final close failed: {format_error(outcome)}"
                error = run_error if run_error is not None else cleanup_cancel
                if error is not None:
                    error.add_note(detail)
                # An unused worker is still owned by this run; report its
                # failure on all results rather than silently dropping it.
                for result in worker_results.get(id(worker)) or results:
                    result.error = f"{result.error}; {detail}" if result.error else detail
        if cleanup_cancel is not None and run_error is None:
            raise cleanup_cancel

    @staticmethod
    def _check_acyclic(plans: dict[str, CasePlan]) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(case_id: str) -> None:
            if case_id in visiting:
                raise ValueError(f"workspace reuse cycle at {case_id!r}")
            if case_id in visited:
                return
            visiting.add(case_id)
            parent = plans[case_id].reuse_workspace_from
            if parent is not None:
                visit(parent)
            visiting.remove(case_id)
            visited.add(case_id)

        for case_id in plans:
            visit(case_id)
