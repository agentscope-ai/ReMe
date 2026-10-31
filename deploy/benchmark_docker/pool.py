"""Bounded Docker workers for independent benchmark cases.

Each case completes construction and its queries before its worker is assigned
another case. A dependent case may reuse the workspace captured immediately
after its parent's construction, without sharing mutable container state.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from collections.abc import Awaitable, Callable
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
    export_diagnostics_to: str | Path | None = None


@dataclass(slots=True)
class CaseResult:
    """Structured build and answer results; judging stays on the host."""

    case_id: str
    build: dict[str, Any] | None = None
    queries: dict[str, Any] | None = None
    workspace_snapshot: Path | None = None
    build_log: str | None = None
    error: str | None = None
    failure_stage: str | None = None
    exception: dict[str, Any] | None = None

    @property
    def success(self) -> bool:
        """Whether all requested phases and owned-worker cleanup succeeded."""
        return (
            self.error is None
            and (self.build is None or self.build.get("success") is True)
            and (self.queries is None or self.queries.get("success") is True)
        )


@dataclass(slots=True)
class _WorkerSlots:
    """Mutable worker availability owned by one pool run."""

    workers: list[DockerReMeSandbox]
    available: asyncio.Queue[DockerReMeSandbox | None]
    unclaimed: set[str]
    live_slots: int
    next_worker_index: int
    exhaustion_error: str = "no available workers"


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

    async def run(
        self,
        plans: list[CasePlan],
        *,
        on_case_finished: Callable[[CasePlan, CaseResult], Awaitable[None]] | None = None,
        accept_build: Callable[[dict[str, Any]], bool] | None = None,
        fail_fast: bool = False,
    ) -> list[CaseResult]:
        """Return results in input order, closing every container on exit."""
        if not plans:
            return []
        self._validate_plans(plans)
        reused_case_ids = {plan.reuse_workspace_from for plan in plans if plan.reuse_workspace_from is not None}

        workers = await self.factory.create_cases(
            [f"pool-{index}" for index in range(min(self.concurrency, len(plans)))],
            concurrency=self.concurrency,
        )
        available: asyncio.Queue[DockerReMeSandbox | None] = asyncio.Queue()
        for worker in workers:
            available.put_nowait(worker)
        slots = _WorkerSlots(workers, available, {plan.case_id for plan in plans}, len(workers), len(workers))
        worker_results: dict[int, list[CaseResult]] = {}
        results: list[CaseResult] = []
        run_error: BaseException | None = None

        try:
            with TemporaryDirectory(prefix="reme-benchmark-pool-") as temporary:
                snapshot_root = Path(temporary)
                # Tasks are started together; awaiting a parent never consumes
                # a container. The graph check above rules out deadlocks.
                tasks: dict[str, asyncio.Task[CaseResult]] = {}

                async def execute(plan: CasePlan, result: CaseResult) -> CaseResult:
                    source = plan.workspace_source
                    if plan.reuse_workspace_from is not None:
                        parent = await tasks[plan.reuse_workspace_from]
                        if parent.workspace_snapshot is None:
                            slots.unclaimed.discard(plan.case_id)
                            return CaseResult(plan.case_id, error=f"source case {parent.case_id!r} failed")
                        source = parent.workspace_snapshot

                    worker = await available.get()
                    slots.unclaimed.discard(plan.case_id)
                    if worker is None:
                        return CaseResult(plan.case_id, error=slots.exhaustion_error)
                    worker_results.setdefault(id(worker), []).append(result)

                    operation = "prepare"
                    try:
                        await worker.reset_case(plan.case_id)
                        if source is not None:
                            operation = "restore"
                            await worker.upload_workspace(source)
                        if plan.build_jobs:
                            operation = "construction"
                            try:
                                result.build = await worker.run_build(list(plan.build_jobs))
                            except Exception as exc:
                                result.error = format_error(exc)
                                result.failure_stage = operation
                                result.exception = {"type": type(exc).__name__, "module": type(exc).__module__}
                                await self._retain_failed_build_log(worker, plan, result)
                                return result
                            accepted = accept_build(result.build) if accept_build else result.build.get("success")
                            if not accepted:
                                result.error = "build failed"
                                result.failure_stage = operation
                                await self._retain_failed_build_log(worker, plan, result)
                                return result
                            if plan.export_build_log_to is not None:
                                operation = "artifacts"
                                await worker.export_build_log(plan.export_build_log_to)
                        # Capture before queries, since answering may change the
                        # workspace. Dependents receive construction state.
                        if plan.export_workspace_to is not None or plan.case_id in reused_case_ids:
                            operation = "artifacts"
                            destination = plan.export_workspace_to or snapshot_root / f"{plan.case_id}.tar.gz"
                            result.workspace_snapshot = await worker.export_workspace(destination)
                        if plan.queries:
                            operation = "answer"
                            result.queries = await worker.run_queries(list(plan.queries))
                            if not result.queries.get("success"):
                                result.error = "query failed"
                                result.failure_stage = operation
                        if plan.export_evaluation_to is not None:
                            operation = "artifacts"
                            await worker.export_evaluation(plan.export_evaluation_to)
                        return result
                    except Exception as exc:  # Keep other cases running and release this worker.
                        result.error = format_error(exc)
                        result.failure_stage = operation
                        result.exception = {"type": type(exc).__name__, "module": type(exc).__module__}
                        return result
                    finally:
                        if result.error and plan.export_diagnostics_to is not None and worker.reusable:
                            try:
                                await worker.export_full(plan.export_diagnostics_to)
                            except Exception as exc:
                                result.error += f"; diagnostic export failed: {format_error(exc)}"
                        await self._release_worker(worker, result, slots)

                async def run_plan(plan: CasePlan) -> CaseResult:
                    result = CaseResult(plan.case_id)
                    try:
                        result = await execute(plan, result)
                    except asyncio.CancelledError:
                        result.error = result.error or "interrupted"
                        raise
                    finally:
                        if on_case_finished is not None:
                            await on_case_finished(plan, result)
                    failed = not result.success
                    if accept_build is not None and result.build is not None:
                        failed = (
                            result.error is not None
                            or not accept_build(result.build)
                            or (result.queries is not None and not result.queries.get("success"))
                        )
                    if fail_fast and failed:
                        raise RuntimeError(f"case {plan.case_id!r} failed: {result.error or 'execution failed'}")
                    return result

                for plan in plans:
                    tasks[plan.case_id] = asyncio.create_task(run_plan(plan))
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

    async def _release_worker(self, worker: DockerReMeSandbox, result: CaseResult, slots: _WorkerSlots) -> None:
        """Return a reusable worker or retire its slot after confirmed cleanup."""
        if worker.reusable:
            slots.available.put_nowait(worker)
            return

        def record_lifecycle_error(phase: str, exc: Exception) -> None:
            detail = f"{phase} failed: {format_error(exc)}"
            result.error = f"{result.error}; {detail}" if result.error else detail
            slots.exhaustion_error = f"no available workers: {detail}"

        # Do not reuse the slot while its old container may still be running.
        try:
            await worker.close()
        except Exception as exc:
            record_lifecycle_error("worker close", exc)
        else:
            if slots.unclaimed and not asyncio.current_task().cancelling():
                worker_index = slots.next_worker_index
                slots.next_worker_index += 1
                try:
                    replacement = await self.factory.create_case(f"pool-{worker_index}")
                except Exception as exc:
                    record_lifecycle_error("replacement", exc)
                else:
                    slots.workers.append(replacement)
                    slots.available.put_nowait(replacement)
                    return

        slots.live_slots -= 1
        if slots.live_slots == 0:
            # Also wake cases that are still waiting for a parent to complete.
            for _ in slots.unclaimed:
                slots.available.put_nowait(None)

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
    def _validate_plans(plans: list[CasePlan]) -> None:
        """Reject invalid plans and dependencies before creating containers."""
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
        BenchmarkDockerPool._check_acyclic(by_id)

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
