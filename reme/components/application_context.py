"""Application context: shared state container for components, jobs, and service."""

import asyncio
from collections.abc import Iterator, MutableMapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
from typing import TYPE_CHECKING, Any, get_args

from ..schema import ApplicationConfig, JobExecution, JobExecutionStatus, WorkspaceContext
from .component_registry import ComponentRegistry, create_application_registry

if TYPE_CHECKING:
    from .base_component import BaseComponent
    from .job import BaseJob
    from .service import BaseService


class _LegacyComponentView(MutableMapping):
    """Compatibility mapping combining shared components with default-workspace instances."""

    def __init__(self, context: "ApplicationContext") -> None:
        self._context = context

    def _default_components(self) -> dict:
        return next(iter(self._context.workspaces.values())).components

    def __getitem__(self, key: str) -> dict:
        if key in self._context.shared_components:
            return self._context.shared_components[key]
        return self._default_components()[key]

    def __setitem__(self, key: str, value: dict) -> None:
        if value and all(getattr(component, "workspace_scoped", False) is True for component in value.values()):
            self._default_components()[key] = value
            self._context.shared_components.pop(key, None)
        else:
            self._context.shared_components[key] = value
            self._default_components().pop(key, None)

    def __delitem__(self, key: str) -> None:
        if key in self._context.shared_components:
            del self._context.shared_components[key]
        else:
            del self._default_components()[key]

    def __iter__(self) -> Iterator[str]:
        yield from self._context.shared_components
        yield from (key for key in self._default_components() if key not in self._context.shared_components)

    def __len__(self) -> int:
        return len(set(self._context.shared_components) | set(self._default_components()))


class ApplicationContext:
    """Application-wide configuration, shared resources, and workspace registry."""

    def __init__(self, *, registry: ComponentRegistry | None = None, **kwargs):
        self.app_config: ApplicationConfig = ApplicationConfig(**kwargs)
        self.registry = registry or create_application_registry()
        self.workspaces: dict[str, WorkspaceContext] = {
            alias: WorkspaceContext.from_config(alias, path, self.app_config)
            for alias, path in self.app_config.workspaces.items()
        }

        self.workspace: WorkspaceContext = next(iter(self.workspaces.values()))
        self.service: "BaseService | None" = None
        self.shared_components: dict[str, dict[str, "BaseComponent"]] = {}
        self._components = _LegacyComponentView(self)
        self.jobs: dict[str, "BaseJob"] = {}
        self.thread_pool: ThreadPoolExecutor | None = None
        self.codex_thread_ownership_lock = asyncio.Lock()
        self.metadata: dict[str, Any] = {}
        self.job_history: dict[str, list[JobExecution]] = {}
        self._job_counts: dict[str, int] = {}
        self._job_lock = Lock()
        self._job_execution_id = 0

    @property
    def components(self) -> MutableMapping:
        """Legacy component lookup; internal execution uses explicit shared or workspace ownership."""
        return self._components

    @components.setter
    def components(self, value: dict) -> None:
        self.shared_components = {}
        for ctype, group in value.items():
            self._components[ctype] = group

    def components_for(self, workspace: WorkspaceContext) -> dict[str, dict[str, "BaseComponent"]]:
        """Snapshot shared and bound-workspace groups without consulting the legacy view."""
        if self.workspaces.get(workspace.alias) is not workspace:
            raise ValueError(f"Workspace {workspace.alias!r} is not bound to this Application")
        groups = {ctype: dict(group) for ctype, group in self.shared_components.items()}
        for ctype, group in workspace.components.items():
            groups.setdefault(ctype, {}).update(group)
        return groups

    def get_component(self, ctype: str, name: str, alias: str | None = None) -> "BaseComponent | None":
        """Compatibility resolver for external callers that select a workspace by alias."""
        alias = alias or next(iter(self.workspaces))
        workspace = self.workspaces.get(alias)
        if workspace is None:
            raise KeyError(f"Unknown workspace alias {alias!r}")
        if name in workspace.components.get(ctype, {}):
            return workspace.get_component(ctype, name)
        if any(name in ws.components.get(ctype, {}) for ws in self.workspaces.values()):
            raise KeyError(f"Component '{name}' in {ctype} has no instance for workspace alias {alias!r}")
        return self.shared_components.get(ctype, {}).get(name)

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
