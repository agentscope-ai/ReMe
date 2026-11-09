"""Codex agent sessions backed by an application-owned app-server."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
import json
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .base_agent_wrapper import BaseAgentWrapper
from .codex_session_history import item_chunks
from ..codex_app_server import CodexAppServer
from ..component_registry import R
from ..job import BackgroundJob, StreamJob
from ...enumeration import ChunkEnum
from ...utils.agent_activity import ActivityTracker
from ...schema import AgentActivity, AgentReply, SessionMessage, SessionSnapshot, SessionStatus, StreamChunk, TokenUsage
from ...schema.stream_chunk import stream_status

if TYPE_CHECKING:
    from openai_codex import AsyncThread, RunInput
    from openai_codex.types import Notification


@dataclass
class _TurnUsage:
    """Invocation-local usage derived from monotonic thread counters."""

    baseline: TokenUsage
    usage: TokenUsage | None = None

    def update(self, total: TokenUsage) -> None:
        """Keep reported spending separate from synthetic context estimates."""
        current = TokenUsage(
            input_tokens=max(0, total.input_tokens - self.baseline.input_tokens),
            output_tokens=max(0, total.output_tokens - self.baseline.output_tokens),
        )
        if current.total_tokens or self.usage is not None:
            self.usage = current


@R.register("codex")
class CodexAgentWrapper(BaseAgentWrapper):
    """Configure agents while sharing transport, authentication and session locks."""

    SDK_PACKAGE = "openai-codex"
    _SERVER_OPTIONS = frozenset(
        {
            "auth_mode",
            "api_key",
            "base_url",
            "supports_websockets",
            "codex_bin",
            "codex_home",
            "config_overrides",
            "model_catalog",
            "experimental_api",
            "launch_args_override",
            "mcp_config",
            "release_threads",
        },
    )

    def __init__(self, app_server: str = "default", **kwargs) -> None:
        self._reject_server_options(kwargs)
        super().__init__(**kwargs)
        self.app_server = self.bind(app_server, CodexAppServer, default_factory=CodexAppServer, optional=False)
        self._active = 0
        self._idle = asyncio.Event()
        self._idle.set()

    @classmethod
    def _reject_server_options(cls, kwargs: dict[str, Any]) -> None:
        invalid = sorted(cls._SERVER_OPTIONS.intersection(kwargs))
        if invalid:
            raise TypeError(f"Configure Codex process options on codex_app_server: {', '.join(invalid)}")

    def _reply_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        self._reject_server_options(kwargs)
        invalid = sorted({"app_server", "cwd"}.intersection(kwargs))
        if invalid:
            raise TypeError(f"Configure Codex wrapper options on the component: {', '.join(invalid)}")
        # A fork inherits its source, rather than this wrapper's current defaults.
        return dict(kwargs) if kwargs.get("fork_session") else self._merged_kwargs(kwargs)

    def _ensure_skills(self, skills: list[str] | str | None) -> None:
        """Expose selected project skills through Codex's repo-level directory."""
        sources = self._resolve_project_skills(skills)
        if not sources:
            return

        target_root = self.project_path / ".agents" / "skills"
        target_root.mkdir(parents=True, exist_ok=True)
        for name, source in sources.items():
            target = target_root / name
            if target.is_symlink():
                if target.resolve() == source.resolve():
                    continue
                raise FileExistsError(f"Codex skill conflict: {target} points to {target.resolve(strict=False)}")
            if target.exists():
                raise FileExistsError(f"Codex skill conflict: {target} already exists and was preserved")
            relative_source = os.path.relpath(source, target.parent)
            target.symlink_to(relative_source, target_is_directory=True)

    @staticmethod
    def _enum(enum_cls: Any, value: Any, default: Any = None) -> Any:
        if value is None:
            return default
        return value if isinstance(value, enum_cls) else enum_cls(value)

    def _thread_options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        from openai_codex import ApprovalMode, Sandbox
        from openai_codex.types import Personality, ThreadSource, ThreadStartSource

        config = dict(kwargs.get("config") or {})
        if proxy_environment := self.command_proxy_environment:
            policy = dict(config.get("shell_environment_policy") or {})
            policy["set"] = {**policy.get("set", {}), **proxy_environment}
            config["shell_environment_policy"] = policy
        options = {
            "approval_mode": self._enum(ApprovalMode, kwargs.get("approval_mode"), ApprovalMode.deny_all),
            "base_instructions": kwargs.get("base_instructions"),
            "config": config,
            "cwd": str(self.cwd),
            "developer_instructions": kwargs.get("system_prompt"),
            "model": kwargs.get("model"),
            "model_provider": kwargs.get("model_provider"),
            "sandbox": self._enum(Sandbox, kwargs.get("sandbox"), Sandbox.full_access),
            "service_tier": kwargs.get("service_tier"),
        }
        if kwargs.get("fork_session"):
            options.update(
                ephemeral=kwargs.get("ephemeral"),
                thread_source=self._enum(ThreadSource, kwargs.get("thread_source")),
            )
            names = {"developer_instructions": "system_prompt"}
            return {key: value for key, value in options.items() if names.get(key, key) in kwargs}
        else:
            options["personality"] = self._enum(Personality, kwargs.get("personality"))
            if not (kwargs.get("resume") or kwargs.get("session_id")):
                options.update(
                    ephemeral=kwargs.get("ephemeral"),
                    service_name=kwargs.get("service_name"),
                    session_start_source=self._enum(ThreadStartSource, kwargs.get("session_start_source")),
                    thread_source=self._enum(ThreadSource, kwargs.get("thread_source")),
                )
        return options

    @asynccontextmanager
    async def _session(
        self,
        kwargs: dict[str, Any],
        *,
        compact: bool = False,
        keep_empty: bool = False,
    ) -> AsyncIterator[AsyncThread]:
        resume, session_id = kwargs.get("resume") or "", kwargs.get("session_id") or ""
        if resume and session_id and resume != session_id:
            raise ValueError("resume and session_id must identify the same Codex thread")
        thread_id = resume or session_id
        fork = bool(kwargs.get("fork_session"))
        if fork and not thread_id:
            raise ValueError("fork_session=True requires resume or session_id")
        inherit_jobs = fork and "job_tools" not in kwargs
        jobs = (
            None
            if compact or inherit_jobs
            else self._resolve_job_tools(list(dict.fromkeys(kwargs.get("job_tools") or [])))
        )
        if any(isinstance(job, (BackgroundJob, StreamJob)) for job in jobs or []):
            raise TypeError("Codex job_tools must be non-stream request jobs")
        if any(not getattr(job, "enabled", True) for job in jobs or []):
            raise ValueError("Codex job_tools must be enabled")
        self._ensure_skills(kwargs.get("skills"))
        await self.start()
        self._active += 1
        self._idle.clear()
        try:
            async with self.app_server.thread(
                thread_id,
                fork=fork,
                options={} if compact else self._thread_options(kwargs),
                jobs=jobs,
                injected=self._resolve_injected_job_kwargs(kwargs),
                mcp_startup_timeout=kwargs.get("mcp_startup_timeout", 30),
                mcp_tool_timeout=kwargs.get("mcp_tool_timeout", 300),
                keep_empty=keep_empty,
            ) as thread:
                yield thread
        finally:
            self._active -= 1
            if not self._active:
                self._idle.set()

    async def _close(self) -> None:
        await self._idle.wait()

    async def _interrupt(self, turn) -> None:
        try:
            await turn.interrupt()
        except Exception as exc:
            self.logger.warning(f"Failed to interrupt Codex turn {turn.id}: {exc}")

    async def _turn_stream(self, turn: Any, usage: _TurnUsage) -> AsyncIterator[Notification]:
        """Observe only this turn's reported spending, including compaction calls."""
        stream = turn.stream()
        try:
            async for event in stream:
                if event.method == "thread/tokenUsage/updated":
                    if event.payload.turn_id != turn.id:
                        continue
                    total = self.app_server.update_token_usage(
                        turn.thread_id,
                        TokenUsage.from_provider(event.payload.token_usage.total),
                    )
                    usage.update(total)
                yield event
        finally:
            await stream.aclose()

    def _finish_reply(self, session_id, completed, items, native_usage, usage, kwargs) -> AgentReply:
        """Build the native TurnResult and portable outcome from this turn's events."""
        from openai_codex import TurnResult

        by_id = {item.root.id: item for item in items}
        by_id.update({item.root.id: item for item in completed.items})
        items = list(by_id.values())
        final_response = None
        fallback = None
        for wrapped in reversed(items):
            item = wrapped.root
            if item.type != "agentMessage":
                continue
            if item.phase is not None and item.phase.value == "final_answer":
                final_response = item.text
                break
            if item.phase is None and fallback is None:
                fallback = item.text
        if final_response is None:
            final_response = fallback
        result = TurnResult(
            id=completed.id,
            status=completed.status,
            error=completed.error,
            started_at=completed.started_at,
            completed_at=completed.completed_at,
            duration_ms=completed.duration_ms,
            final_response=final_response,
            items=items,
            usage=native_usage,
        )
        status = completed.status.value
        error = completed.error.message if completed.error else None
        structured_output = None
        if status == "completed" and kwargs.get("output_schema") is not None:
            try:
                structured_output = json.loads(final_response or "")
            except json.JSONDecodeError:
                status = "failed"
                error = "Codex returned invalid JSON for the requested output_schema"
        return self._reply_result(
            session_id=session_id,
            result=final_response,
            structured_output=structured_output,
            usage=usage,
            status=status,
            error=error,
            raw=self._serialize(result),
        )

    async def create_session(self, **kwargs) -> str:
        """Start an empty thread with the same instructions and tools as its future turns."""
        async with self._session(self._reply_kwargs(kwargs), keep_empty=True) as thread:
            return thread.id

    async def release_session(self, session_id: str) -> None:
        """Release a superseded empty thread; regular turns already release their leases."""
        await self.app_server.release_session(session_id)

    async def compact_session(self, session_id: str) -> None:
        """Hold the conversation lock until native compaction has finished."""
        async with self._session({"session_id": session_id}, compact=True) as thread:
            await self.app_server.compact(thread)

    async def get_session_path(self, session_id: str) -> Path | None:
        """Locate the native session file without loading its history or waiting for a turn."""
        await self.start()
        return await self.app_server.get_session_path(session_id)

    async def read_session(self, session_id: str) -> SessionSnapshot:
        """Read history without acquiring the generation lease."""
        async with self._session_request():
            return await self.app_server.read_session(session_id)

    async def get_session_status(self, session_id: str) -> SessionStatus:
        """Report native thread activity, including approval/input waits."""
        async with self._session_request():
            return await self.app_server.get_session_status(session_id)

    @asynccontextmanager
    async def _session_request(self):
        """Normalize recoverable SDK failures at the session-owner boundary."""
        from openai_codex.errors import TransportClosedError, is_retryable_error

        try:
            await self.start()
            yield
        except Exception as exc:
            if isinstance(exc, TransportClosedError) or is_retryable_error(exc):
                raise ConnectionError(str(exc)) from exc
            raise

    async def append_session(
        self,
        session_id: str,
        content: str,
        *,
        message_id: str,
        timeout: float = 30,
        expected_turn_id: str | None = None,
        delivery: str = "idle",
    ) -> SessionMessage:
        """Append a proactive message through the shared session owner."""
        async with self._session_request():
            return await self.app_server.append_session(
                session_id,
                content,
                message_id=message_id,
                timeout=timeout,
                expected_turn_id=expected_turn_id,
                delivery=delivery,
            )

    def _turn_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        from openai_codex import ApprovalMode, Sandbox
        from openai_codex.types import Personality, ReasoningEffort, ReasoningSummary

        options = {
            "approval_mode": self._enum(ApprovalMode, kwargs.get("approval_mode")),
            "cwd": str(self.cwd),
            "effort": self._enum(ReasoningEffort, kwargs.get("effort")),
            "model": kwargs.get("model"),
            "output_schema": kwargs.get("output_schema"),
            "personality": self._enum(Personality, kwargs.get("personality")),
            "sandbox": self._enum(Sandbox, kwargs.get("sandbox")),
            "service_tier": kwargs.get("service_tier"),
            "summary": self._enum(ReasoningSummary, kwargs.get("summary")),
        }
        return (
            {key: value for key, value in options.items() if key in kwargs} if kwargs.get("fork_session") else options
        )

    @classmethod
    # pylint: disable=too-many-return-statements
    def _event_to_chunks(
        cls,
        event: Notification,
        session_id: str,
        *,
        usage: TokenUsage | None = None,
    ) -> list[StreamChunk]:
        """Map notifications; usage replaces the cumulative counts for its turn block."""
        method, payload = event.method, event.payload
        make_chunk = partial(cls._chunk, session_id=session_id, turn_id=getattr(payload, "turn_id", None))
        if method == "turn/started":
            return [
                cls._chunk(
                    ChunkEnum.REPLY_START,
                    event="start",
                    session_id=session_id,
                    turn_id=payload.turn.id,
                    status="running",
                    metadata={"turn_id": payload.turn.id},
                ),
            ]
        if method == "item/agentMessage/delta":
            return [
                make_chunk(
                    ChunkEnum.CONTENT,
                    block_id=payload.item_id,
                    message_id=payload.item_id,
                    chunk=payload.delta,
                ),
            ]
        if method in {"item/reasoning/summaryTextDelta", "item/reasoning/textDelta", "item/plan/delta"}:
            kind = {"item/reasoning/summaryTextDelta": "summary", "item/reasoning/textDelta": "content"}.get(method)
            index = getattr(payload, "summary_index" if kind == "summary" else "content_index", 0)
            return [
                make_chunk(
                    ChunkEnum.THINK,
                    block_id=f"{payload.item_id}:{kind}:{index}" if kind else payload.item_id,
                    message_id=payload.item_id,
                    chunk=payload.delta,
                    metadata={"kind": f"reasoning_{kind}" if kind else "plan", "index": index},
                ),
            ]
        if method in {"item/commandExecution/outputDelta", "item/fileChange/outputDelta"}:
            return [
                make_chunk(
                    ChunkEnum.TOOL_RESULT,
                    block_id=payload.item_id,
                    tool_call_id=payload.item_id,
                    chunk=payload.delta,
                    status="running",
                ),
            ]
        if method == "item/mcpToolCall/progress":
            return [
                make_chunk(
                    ChunkEnum.TOOL_RESULT,
                    block_id=payload.item_id,
                    tool_call_id=payload.item_id,
                    chunk=payload.message,
                    status="running",
                ),
            ]
        if method in {"item/autoApprovalReview/started", "item/autoApprovalReview/completed"}:
            action = cls._serialize(payload.action)
            review = cls._serialize(payload.review)
            review_id = payload.review_id
            target_item_id = getattr(payload, "target_item_id", None)
            status = "started" if method.endswith("/started") else "completed"
            decision_source = cls._serialize(getattr(payload, "decision_source", None))
            return [
                make_chunk(
                    ChunkEnum.APPROVAL,
                    block_id=target_item_id or review_id,
                    tool_call_id=target_item_id,
                    chunk=action,
                    event="start" if status == "started" else "end",
                    status="running" if status == "started" else stream_status((review or {}).get("status")),
                    metadata={
                        "review_id": review_id,
                        "status": status,
                        "review": review,
                        "decision_source": decision_source,
                        "turn_id": payload.turn_id,
                    },
                ),
            ]
        if method in {"item/started", "item/completed"}:
            return item_chunks(
                payload.item.root,
                event="start" if method == "item/started" else "end",
                session_id=session_id,
                turn_id=payload.turn_id,
            )
        if method == "thread/tokenUsage/updated":
            if usage is None:
                return []
            return [
                make_chunk(
                    ChunkEnum.USAGE,
                    block_id=f"usage:{payload.turn_id}",
                    chunk=usage.model_dump(),
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    metadata={
                        "usage": usage.model_dump(),
                        "native": cls._serialize(getattr(payload, "token_usage", None)),
                    },
                    event="end",
                ),
            ]
        if method == "error":
            return [
                make_chunk(
                    ChunkEnum.ERROR,
                    chunk=payload.error.message,
                    metadata={"will_retry": payload.will_retry, "native": cls._serialize(payload.error)},
                ),
            ]
        if method == "turn/completed":
            turn = payload.turn
            chunks = [
                chunk
                for item in getattr(turn, "items", [])
                for chunk in item_chunks(getattr(item, "root", item), session_id=session_id, turn_id=turn.id)
            ]
            if turn.error:
                chunks.append(make_chunk(ChunkEnum.ERROR, chunk=turn.error.message))
            chunks.append(
                cls._chunk(
                    ChunkEnum.REPLY_END,
                    event="end",
                    session_id=session_id,
                    turn_id=turn.id,
                    status=stream_status(turn.status),
                    metadata={
                        "turn_id": turn.id,
                        "status": turn.status.value,
                        "duration_ms": turn.duration_ms,
                    },
                ),
            )
            return chunks
        diagnostic = method.startswith("hook/") or not getattr(payload, "turn_id", None)
        return [
            make_chunk(
                ChunkEnum.DATA,
                block_id=getattr(payload, "item_id", None),
                chunk=cls._serialize(payload),
                metadata={"codex_method": method, **({"visibility": "diagnostic"} if diagnostic else {})},
            ),
        ]

    async def reply_stream(self, inputs: RunInput, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """Execute once, retaining every notification and a complete terminal outcome."""
        kwargs = self._reply_kwargs(kwargs)
        emitted_usage = None
        source = kwargs.pop("activity_source", "chat")
        target_session = kwargs.pop("activity_session_id", None)
        run_id = kwargs.pop("activity_run_id", None)
        task_id = kwargs.pop("activity_task_id", None)
        activity = None
        try:
            async with self._session(kwargs) as thread:
                usage = _TurnUsage(await self.app_server.token_usage(thread.id))
                turn = await self.app_server.turn(thread, inputs, **self._turn_kwargs(kwargs))
                activity = AgentActivity(
                    session_id=target_session or thread.id,
                    turn_id=turn.id,
                    source=source,
                    phase="thinking",
                    run_id=run_id,
                    task_id=task_id,
                )
                self.app_server.publish_activity(activity)
                tracker = ActivityTracker(activity)
                stream = self._turn_stream(turn, usage)
                items, native_usage, completed, terminal = [], None, None, None
                try:
                    async for event in stream:
                        payload = event.payload
                        if event.method == "item/completed" and payload.turn_id == turn.id:
                            items.append(payload.item)
                        elif event.method == "thread/tokenUsage/updated" and payload.turn_id == turn.id:
                            native_usage = payload.token_usage
                        elif event.method == "turn/completed" and payload.turn.id == turn.id:
                            completed = payload.turn
                        update = usage.usage if usage.usage != emitted_usage else None
                        chunks = self._event_to_chunks(event, thread.id, usage=update)
                        if event.method == "thread/tokenUsage/updated":
                            emitted_usage = usage.usage
                        if not chunks:
                            chunks = [self._chunk(ChunkEnum.DATA, chunk=None, metadata={"visibility": "diagnostic"})]
                        for chunk in chunks:
                            chunk.metadata.pop("native", None)
                        chunks[0].metadata["native"] = {"method": event.method, "payload": self._serialize(payload)}
                        for chunk in chunks:
                            chunk.session_id = thread.id
                            chunk.turn_id = chunk.turn_id or turn.id
                            if chunk.chunk_type == ChunkEnum.REPLY_END:
                                terminal = chunk
                            else:
                                updated = tracker.update(chunk)
                                if updated != activity:
                                    activity = updated
                                    self.app_server.publish_activity(activity)
                                yield chunk
                    if completed is None or terminal is None:
                        raise RuntimeError("Codex turn completed event not received")
                    result = self._finish_reply(thread.id, completed, items, native_usage, usage.usage, kwargs)
                finally:
                    if usage.usage is not None:
                        self._record_token_usage(usage.usage)
                    if completed is None:
                        await self._interrupt(turn)
                    await stream.aclose()
            terminal.chunk = {key: value for key, value in result.items() if key != "events"}
            terminal.status = result["status"]
            terminal.metadata["error"] = result["error"]
            activity = tracker.update(terminal)
            self.app_server.publish_activity(activity)
            yield terminal
        except BaseException as exc:
            if activity is not None and activity.phase not in {"completed", "failed", "interrupted"}:
                phase = "interrupted" if isinstance(exc, (asyncio.CancelledError, GeneratorExit)) else "failed"
                self.app_server.publish_activity(activity.model_copy(update={"phase": phase, "tool": None}))
            raise
