"""Shared application state and per-invocation execution context."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from datetime import datetime, timezone
from threading import Lock
from typing import TYPE_CHECKING, Any, get_args

from .registry import ComponentRegistry, create_application_registry
from ..enumeration import ChunkEnum
from ..schema import ApplicationConfig, JobExecution, JobExecutionStatus, Response, StreamChunk

if TYPE_CHECKING:
    from ..components.base import BaseComponent
    from ..components.job import BaseJob
    from ..components.service import BaseService


_current_job_execution: ContextVar[JobExecution | None] = ContextVar("reme_job_execution", default=None)


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


class RuntimeContext:
    """Scratch space for a single execution.

    Holds the response object, an optional stream queue, a stop event, and a
    free-form data dict accessible via mapping-style operators (``ctx[key]``).
    """

    def __init__(
        self,
        response: Response | None = None,
        stream_queue: asyncio.Queue | None = None,
        stop_event: asyncio.Event | None = None,
        **kwargs,
    ):
        self.response: Response = response or Response()
        self.stream_queue: asyncio.Queue | None = stream_queue
        self.stop_event: asyncio.Event | None = stop_event
        self.data: dict = kwargs
        self.job_execution: JobExecution | None = _current_job_execution.get()
        self.step_error: Exception | None = None
        # Providers can emit terminal errors without raising or changing the Response.
        self.stream_error: str | None = None

    # ----- Data dict access ----------------------------------------------

    def get(self, key: str, default=None):
        """Get a value from the data dict with an optional default."""
        return self.data.get(key, default)

    def update(self, data: dict) -> "RuntimeContext":
        """Merge `data` into the context and return self for chaining."""
        self.data.update(data)
        return self

    def __getitem__(self, key: str):
        return self.data[key]

    def __setitem__(self, key: str, value):
        self.data[key] = value

    def __delitem__(self, key: str):
        del self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    @classmethod
    def from_context(cls, context: "RuntimeContext | None" = None, **kwargs) -> "RuntimeContext":
        """Reuse `context` (merging kwargs into its data) or create a fresh one."""
        if context is None:
            return cls(**kwargs)
        return context.update(kwargs)

    # ----- Streaming -----------------------------------------------------

    @property
    def stream(self) -> bool:
        """Whether a stream queue is attached (i.e., streaming is enabled)."""
        return self.stream_queue is not None

    async def _enqueue(self, chunk: StreamChunk) -> None:
        """Put a chunk on the stream queue; raises when streaming is disabled."""
        if self.stream_queue is None:
            raise RuntimeError("Stream queue not initialized")
        if chunk.chunk_type == ChunkEnum.ERROR and chunk.metadata.get("will_retry") is not True:
            self.stream_error = str(chunk.chunk)
        if chunk.chunk_type == ChunkEnum.REPLY_END and chunk.status == "failed":
            self.stream_error = self.stream_error or str(
                chunk.metadata.get("error") or chunk.metadata.get("finished_reason") or "Agent reply failed",
            )
        await self.stream_queue.put(chunk)

    async def add_stream_string(self, chunk: str, chunk_type: ChunkEnum) -> "RuntimeContext":
        """Emit a text chunk to the stream queue."""
        await self._enqueue(StreamChunk(chunk_type=chunk_type, chunk=chunk))
        return self

    async def add_stream_chunk(self, chunk: StreamChunk) -> "RuntimeContext":
        """Emit a complete chunk while preserving session and tool metadata."""
        await self._enqueue(chunk)
        return self

    async def add_stream_done(self) -> "RuntimeContext":
        """Emit the terminal DONE marker to close the stream."""
        await self._enqueue(StreamChunk(chunk_type=ChunkEnum.DONE, chunk="", done=True))
        return self

    # ----- Misc ----------------------------------------------------------

    def apply_mapping(self, mapping: dict[str, str]) -> "RuntimeContext":
        """Copy ``data[source]`` to ``data[target]`` for each ``source: target`` pair."""
        if not mapping:
            return self
        for source, target in mapping.items():
            if source in self.data:
                self.data[target] = self.data[source]
        return self
