"""Application context: shared state container for components, jobs, and service."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
from typing import TYPE_CHECKING, Any, get_args

from ..schema import ApplicationConfig, JobExecution, JobExecutionStatus
from .component_registry import ComponentRegistry, create_application_registry

if TYPE_CHECKING:
    from .base_component import BaseComponent
    from .job import BaseJob
    from .service import BaseService


class ApplicationContext:
    """Shared state container holding parsed config and wired components.

    Application performs the wiring; this class stores the results and provides
    synchronized access to in-memory job execution snapshots.
    """

    def __init__(self, *, registry: ComponentRegistry | None = None, **kwargs):
        # Parse raw kwargs into a typed, validated config object.
        self.app_config: ApplicationConfig = ApplicationConfig(**kwargs)
        self.registry = registry or create_application_registry()

        # Populated by Application during initialization.
        self.service: "BaseService | None" = None
        self.components: dict[str, dict[str, "BaseComponent"]] = {}
        self.jobs: dict[str, "BaseJob"] = {}
        self.thread_pool: ThreadPoolExecutor | None = None
        self.job_history: dict[str, list[JobExecution]] = {}
        self._job_counts: dict[str, int] = {}
        self._job_lock = Lock()
        self._job_execution_id = 0

        # Application-lifetime shared state. Values remain available across Job and Step
        # invocations while this Application is running, so Jobs may keep cross-call state here.
        # This is in-memory state, not durable storage; use workspace files or a store when state
        # must survive an Application restart.
        self.metadata: dict[str, Any] = {}

    def start_job_execution(self, name: str, *, track_execution: bool = True) -> JobExecution | None:
        """Count every invocation and optionally append a running execution."""
        with self._job_lock:
            self._job_counts[name] = self._job_counts.get(name, 0) + 1
            if not track_execution:
                return None
            self._job_execution_id += 1
            execution = JobExecution(id=self._job_execution_id, started_at=datetime.now(timezone.utc))
            self.job_history.setdefault(name, []).append(execution)
            return execution

    def get_job_count(self, name: str) -> int:
        """Return cumulative starts, independently of tracking and history retention."""
        with self._job_lock:
            if name not in self.jobs:
                raise KeyError(f"Job '{name}' not found")
            return self._job_counts.get(name, 0)

    def finish_job_execution(
        self,
        name: str,
        execution: JobExecution,
        status: JobExecutionStatus,
        error: str | None = None,
    ) -> None:
        """Finish an execution and prune only completed records, oldest first."""
        with self._job_lock:
            execution.status = status
            execution.finished_at = datetime.now(timezone.utc)
            execution.error = error[:2000] if error else None
            if status != "failed":
                execution.failed_step = None
            execution.step_path = []
            execution.step_started_at = None
            history = self.job_history[name]
            completed = [record for record in history if record.status != "running"]
            limit = self.app_config.job_history_limit
            keep = {record.id for record in completed[-limit:]} if limit else set()
            self.job_history[name] = [record for record in history if record.status == "running" or record.id in keep]

    def enter_job_step(self, execution: JobExecution, name: str) -> tuple[list[str], datetime | None] | None:
        """Push a step and return the parent path and its original start time."""
        with self._job_lock:
            if execution.status != "running":
                return None
            previous = execution.step_path, execution.step_started_at
            execution.step_path = [*execution.step_path, name]
            execution.step_started_at = datetime.now(timezone.utc)
            return previous

    def leave_job_step(
        self,
        execution: JobExecution,
        previous: tuple[list[str], datetime | None],
        failed: bool,
        response_failed: bool | None = None,
    ) -> None:
        """Restore the parent; None preserves an exception path, False clears a recovered failure."""
        with self._job_lock:
            if execution.status != "running":
                return
            if failed or (response_failed and execution.failed_step is None):
                execution.failed_step = list(execution.step_path)
            elif response_failed is False:
                execution.failed_step = None
            execution.step_path, execution.step_started_at = previous

    def get_job_status(
        self,
        name: str | None = None,
        status: JobExecutionStatus | None = None,
        limit: int = 20,
    ) -> dict[str, list[dict]]:
        """Return detached JSON snapshots, newest first, including untriggered tracked jobs."""
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if status is not None and status not in get_args(JobExecutionStatus):
            raise ValueError(f"Invalid execution status: {status}")
        with self._job_lock:
            if name is not None:
                if name not in self.jobs and name not in self.job_history:
                    raise KeyError(f"Job '{name}' not found")
                names = [name]
            else:
                names = sorted(
                    self.job_history.keys() | {key for key, job in self.jobs.items() if job.track_execution},
                )
            result = {}
            for key in names:
                records = [
                    record
                    for record in reversed(self.job_history.get(key, []))
                    if status is None or record.status == status
                ]
                result[key] = [record.model_dump(mode="json") for record in records[:limit]]
            return result
