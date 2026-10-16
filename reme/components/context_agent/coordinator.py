"""Bounded host-facing observation, delivery leases and memory hint coordination."""

import asyncio
import copy
from dataclasses import dataclass, field
import time

from pydantic import BaseModel, ConfigDict, Field

from ..as_classifier import BaseAsClassifier
from ..base_component import BaseComponent
from ..component_registry import R
from ..file_store import BaseFileStore
from ...enumeration import ComponentEnum
from ...schema.context_agent import ContextAgentOptions, ContextEvaluationResult, ContextEvent, ContextHint


class _Limits(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    enabled: bool = False
    mode: str = Field(default="shadow", pattern="^(shadow|hint)$")
    max_sessions: int = Field(default=128, ge=1, le=4096)
    max_concurrency: int = Field(default=4, ge=1, le=64)
    max_evaluations: int = Field(default=10, ge=1, le=100)
    max_hints: int = Field(default=3, ge=1, le=20)
    cooldown_seconds: float = Field(default=30, ge=0, le=3600)
    session_ttl_seconds: float = Field(default=300, gt=0, le=86400)
    lease_seconds: float = Field(default=5, gt=0, le=60)


@dataclass
class _Session:
    event: ContextEvent
    allowed_paths: object = None
    pending: bool = True
    worker: asyncio.Task | None = None
    result: ContextEvaluationResult | None = None
    evaluations: int = 0
    delivered: int = 0
    seen: set[str] = field(default_factory=set)
    next_hint_at: float = 0
    touched_at: float = field(default_factory=time.monotonic)
    lease_until: float = 0
    leased_hint_id: str | None = None


@R.register("jev")
class ContextAgent(BaseComponent):
    """Observe host events and offer ready evidence without awaiting model inference.

    Hosts call submit at tool boundaries, then take_ready before a model request.
    After successful insertion, ack the hint. All calls belong to one event loop.
    The component shares the single-evaluation engine with evaluate_context Job;
    it has no dependency on Job startup or shutdown.
    """

    component_type = ComponentEnum.CONTEXT_AGENT

    def __init__(
        self,
        as_classifier: str = "context_agent",
        file_store: str = "default",
        options: dict | None = None,
        limits: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.as_classifier = self.bind(as_classifier, BaseAsClassifier, optional=False)
        self.file_store = self.bind(file_store, BaseFileStore, optional=False)
        self.options = ContextAgentOptions.model_validate(options or {})
        self.limits = _Limits.model_validate(limits or {})
        self._state_lock = asyncio.Lock()
        self._sessions: dict[str, _Session] = {}
        self._tasks: set[asyncio.Task] = set()
        self._slots = asyncio.Semaphore(self.limits.max_concurrency)
        self._accepting = False

    async def _start(self):
        self._accepting = self.limits.enabled

    async def _close(self):
        async with self._state_lock:
            self._accepting = False
            tasks = list(self._tasks)
            for task in tasks:
                task.cancel()
            self._sessions.clear()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _expire_sessions(self):
        now = time.monotonic()
        expired = [
            key for key, state in self._sessions.items() if now - state.touched_at > self.limits.session_ttl_seconds
        ]
        for key in expired:
            state = self._sessions.pop(key)
            if state.worker:
                state.worker.cancel()

    async def submit(self, event: ContextEvent | dict, *, allowed_paths=None) -> bool:
        """Publish a copied snapshot; false means disabled, old, or resource-limited."""
        event = ContextEvent.model_validate(event).model_copy(deep=True)
        async with self._state_lock:
            if not self._accepting:
                return False
            self._expire_sessions()
            state = self._sessions.get(event.session_id)
            if state is not None and event.revision <= state.event.revision:
                return False
            if state is None:
                if len(self._sessions) >= self.limits.max_sessions or len(self._tasks) >= self.limits.max_sessions:
                    return False
                state = _Session(event=event)
                self._sessions[event.session_id] = state
            elif event.task_id != state.event.task_id:
                # Keep the same worker to ensure task switches cannot accumulate tasks.
                state.evaluations = state.delivered = 0
                state.seen.clear()
                state.next_hint_at = 0
            state.event = event
            state.allowed_paths = copy.deepcopy(allowed_paths)
            state.pending = True
            state.result = None
            state.leased_hint_id = None
            state.lease_until = 0
            state.touched_at = time.monotonic()
            if state.worker is None:
                state.worker = asyncio.create_task(self._drain(event.session_id, state))
                self._tasks.add(state.worker)
                state.worker.add_done_callback(self._tasks.discard)
            return True

    async def _drain(self, session_id, state):
        # Lazy import avoids component/Step registration cycles.
        from ...steps.context.evaluate_context import evaluate_context

        try:
            while True:
                async with self._state_lock:
                    if not state.pending or self._sessions.get(session_id) is not state:
                        state.worker = None
                        return
                    state.pending = False
                    if state.evaluations >= self.limits.max_evaluations or state.delivered >= self.limits.max_hints:
                        state.worker = None
                        return
                    if time.monotonic() < state.next_hint_at:
                        state.worker = None
                        return
                    event = state.event.model_copy(deep=True)
                    event.task_snapshot.seen_evidence = list(
                        dict.fromkeys([*sorted(state.seen), *event.task_snapshot.seen_evidence]),
                    )[:128]
                    allowed_paths = copy.deepcopy(state.allowed_paths)
                    state.evaluations += 1
                try:
                    # Includes queueing behind other sessions in the total deadline.
                    async with asyncio.timeout(self.options.total_timeout_ms / 1000):
                        async with self._slots:
                            result = await evaluate_context(
                                event,
                                self.options,
                                store=self.file_store,
                                classifier=self.as_classifier,
                                workspace=self.workspace_path,
                                allowed_paths=allowed_paths,
                            )
                except Exception as exc:
                    # Never include event text, provider bodies or credentials in logs.
                    result = ContextEvaluationResult(
                        event_id=event.event_id,
                        task_id=event.task_id,
                        based_on_revision=event.revision,
                        reason_code="timeout" if isinstance(exc, TimeoutError) else "evaluation_failed",
                    )
                async with self._state_lock:
                    if self._sessions.get(session_id) is not state:
                        return
                    if state.event.revision == event.revision:
                        state.result = result
        finally:
            async with self._state_lock:
                if state.worker is asyncio.current_task():
                    state.worker = None

    async def wait_idle(self, session_id: str):
        """Wait for offline replay/debugging only; production checkpoints never wait."""
        async with self._state_lock:
            state = self._sessions.get(session_id)
            task = state.worker if state else None
        if task:
            await asyncio.shield(task)

    async def last_result(self, session_id: str) -> ContextEvaluationResult | None:
        """Read a copied shadow result without marking any evidence as delivered."""
        async with self._state_lock:
            state = self._sessions.get(session_id)
            return state.result.model_copy(deep=True) if state and state.result else None

    async def take_ready(
        self,
        session_id: str,
        task_id: str,
        revision: int,
        *,
        allowed_paths=None,
    ) -> ContextHint | None:
        """Lease a ready hint after checking current host permissions and source versions."""
        from ...steps.context.candidates import evidence_is_current

        async with self._state_lock:
            state = self._sessions.get(session_id)
            if not self._accepting or self.limits.mode != "hint" or state is None:
                return None
            result = state.result
            if state.event.task_id != task_id or state.event.revision != revision or result is None:
                return None
            if (
                result.hint is None
                or result.based_on_revision != revision
                or state.delivered >= self.limits.max_hints
                or time.monotonic() < state.lease_until
            ):
                return None
            hint = result.hint.model_copy(deep=True)
            original_scope = copy.deepcopy(state.allowed_paths)
        for scope in (original_scope, allowed_paths):
            if not await evidence_is_current(self.workspace_path, hint.evidence, scope, self.options.max_file_bytes):
                return None
        async with self._state_lock:
            if (
                not self._accepting
                or self._sessions.get(session_id) is not state
                or state.result is not result
                or time.time() >= hint.expires_at
                or time.monotonic() < state.lease_until
            ):
                return None
            state.leased_hint_id = hint.hint_id
            state.lease_until = time.monotonic() + self.limits.lease_seconds
            state.touched_at = time.monotonic()
            return hint

    async def ack(self, session_id: str, hint_id: str) -> bool:
        """Commit successful insertion; hosts must also deduplicate by hint_id."""
        async with self._state_lock:
            state = self._sessions.get(session_id)
            if state is None or state.leased_hint_id != hint_id or not state.result or not state.result.hint:
                return False
            state.seen.update(item.candidate_id for item in state.result.hint.evidence)
            state.delivered += 1
            state.next_hint_at = time.monotonic() + self.limits.cooldown_seconds
            state.result = None
            state.leased_hint_id = None
            state.lease_until = 0
            return True

    async def dispose_session(self, session_id: str):
        """Stop observing a completed session and reclaim its in-flight evaluation."""
        async with self._state_lock:
            state = self._sessions.pop(session_id, None)
            task = state.worker if state else None
            if task:
                task.cancel()
        if task:
            await asyncio.gather(task, return_exceptions=True)
