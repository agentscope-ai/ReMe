"""Codex agent sessions backed by an application-owned app-server."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
import json
import os
from typing import Any, TYPE_CHECKING

from .base_agent_wrapper import BaseAgentWrapper
from ..codex_app_server import CodexAppServer
from ..component_registry import R
from ..job import BackgroundJob, StreamJob
from ...enumeration import ChunkEnum
from ...schema import StreamChunk, TokenUsage

if TYPE_CHECKING:
    from openai_codex import AsyncThread, RunInput
    from openai_codex.types import Notification


@R.register("codex")
class CodexAgentWrapper(BaseAgentWrapper):
    """Configure agents while sharing transport, authentication and session locks."""

    SDK_PACKAGE = "openai-codex"
    _SERVER_OPTIONS = frozenset(
        {
            "auth_mode",
            "api_key",
            "base_url",
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

    def _reply_kwargs(self, kwargs: dict[str, Any], *, stream: bool = False) -> dict[str, Any]:
        self._reject_server_options(kwargs)
        invalid = sorted({"app_server", "cwd"}.intersection(kwargs))
        if invalid:
            raise TypeError(f"Configure Codex wrapper options on the component: {', '.join(invalid)}")
        return self._merged_stream_kwargs(kwargs) if stream else self._merged_kwargs(kwargs)

    @staticmethod
    def _codex_usage(usage: Any) -> TokenUsage:
        return TokenUsage.from_provider(usage)

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
            "approval_mode": self._enum(ApprovalMode, kwargs.get("approval_mode"), ApprovalMode.auto_review),
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
    async def _session(self, kwargs: dict[str, Any], *, compact: bool = False) -> AsyncIterator[AsyncThread]:
        resume, session_id = kwargs.get("resume") or "", kwargs.get("session_id") or ""
        if resume and session_id and resume != session_id:
            raise ValueError("resume and session_id must identify the same Codex thread")
        thread_id = resume or session_id
        fork = bool(kwargs.get("fork_session"))
        if fork and not thread_id:
            raise ValueError("fork_session=True requires resume or session_id")
        jobs = None if compact else self._resolve_job_tools(list(dict.fromkeys(kwargs.get("job_tools") or [])))
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

    @staticmethod
    def _serialize(value: Any) -> Any:
        from pydantic_core import to_jsonable_python

        return to_jsonable_python(value, by_alias=True)

    async def reply(self, inputs: RunInput, **kwargs) -> dict:
        """Run an isolated or explicitly resumed conversation on the shared server."""
        kwargs = self._reply_kwargs(kwargs)
        async with self._session(kwargs) as thread:
            turn = await self.app_server.turn(thread, inputs, **self._turn_kwargs(kwargs))
            try:
                result = await turn.run()
            except BaseException:
                await self._interrupt(turn)
                raise
        final_response = result.final_response or ""
        response = {
            "session_id": thread.id,
            "last_message": final_response,
            "result": final_response,
            "turn": self._serialize(result),
            "usage": None,
        }
        if (raw_usage := getattr(result, "usage", None)) is not None:
            usage = self._codex_usage(raw_usage.last)
            response["usage"] = usage.model_dump()
            self._record_token_usage(usage)
        if kwargs.get("output_schema") is not None:
            try:
                response["structured_output"] = json.loads(final_response)
            except json.JSONDecodeError as exc:
                raise ValueError("Codex returned invalid JSON for the requested output_schema") from exc
        return response

    async def compact_session(self, session_id: str) -> None:
        """Hold the conversation lock until native compaction has finished."""
        async with self._session({"session_id": session_id}, compact=True) as thread:
            await self.app_server.compact(thread)

    def _turn_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        from openai_codex import ApprovalMode, Sandbox
        from openai_codex.types import Personality, ReasoningEffort, ReasoningSummary

        return {
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

    @classmethod
    # pylint: disable=too-many-return-statements
    def _event_to_chunks(cls, event: Notification, session_id: str) -> list[StreamChunk]:
        """Convert one Codex app-server notification to unified stream chunks."""
        method, payload = event.method, event.payload
        make_chunk = partial(cls._chunk, session_id=session_id)
        if method == "turn/started":
            return [make_chunk(ChunkEnum.REPLY_START, metadata={"turn_id": payload.turn.id})]
        if method == "item/agentMessage/delta":
            return [make_chunk(ChunkEnum.CONTENT, block_id=payload.item_id, chunk=payload.delta)]
        if method in {"item/reasoning/summaryTextDelta", "item/reasoning/textDelta", "item/plan/delta"}:
            return [make_chunk(ChunkEnum.THINK, block_id=payload.item_id, chunk=payload.delta)]
        if method in {"item/commandExecution/outputDelta", "item/fileChange/outputDelta"}:
            return [
                make_chunk(
                    ChunkEnum.TOOL_RESULT,
                    block_id=payload.item_id,
                    tool_call_id=payload.item_id,
                    chunk=payload.delta,
                ),
            ]
        if method == "item/mcpToolCall/progress":
            return [
                make_chunk(
                    ChunkEnum.TOOL_RESULT,
                    block_id=payload.item_id,
                    tool_call_id=payload.item_id,
                    chunk=payload.message,
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
            item = payload.item.root
            item_type, item_id = item.type, item.id
            tool_types = {"commandExecution", "fileChange", "mcpToolCall", "dynamicToolCall", "collabAgentToolCall"}
            if item_type not in tool_types:
                return []
            name = getattr(item, "tool", None) or item_type
            chunk_type = ChunkEnum.TOOL_CALL if method == "item/started" else ChunkEnum.TOOL_RESULT
            return [
                make_chunk(
                    chunk_type,
                    block_id=item_id,
                    tool_call_id=item_id,
                    tool_call_name=name,
                    chunk=cls._serialize(item),
                ),
            ]
        if method == "thread/tokenUsage/updated":
            usage = payload.token_usage.last
            normalized = cls._codex_usage(usage)
            return [
                make_chunk(
                    ChunkEnum.USAGE,
                    chunk=normalized.model_dump(),
                    input_tokens=normalized.input_tokens,
                    output_tokens=normalized.output_tokens,
                    metadata={"usage": normalized.model_dump()},
                ),
            ]
        if method == "error":
            return [
                make_chunk(
                    ChunkEnum.ERROR,
                    chunk=payload.error.message,
                    metadata={"will_retry": payload.will_retry},
                ),
            ]
        if method == "turn/completed":
            turn = payload.turn
            chunks = []
            if turn.error:
                chunks.append(make_chunk(ChunkEnum.ERROR, chunk=turn.error.message))
            chunks.append(
                make_chunk(
                    ChunkEnum.REPLY_END,
                    metadata={
                        "turn_id": turn.id,
                        "status": turn.status.value,
                        "duration_ms": turn.duration_ms,
                    },
                ),
            )
            return chunks
        if getattr(payload, "turn_id", None):
            return [
                make_chunk(
                    ChunkEnum.DATA,
                    block_id=getattr(payload, "item_id", None),
                    chunk=cls._serialize(payload),
                    metadata={"codex_method": method},
                ),
            ]
        return []

    async def reply_stream(self, inputs: RunInput, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """Stream notifications routed by the SDK to this conversation's turn."""
        kwargs = self._reply_kwargs(kwargs, stream=True)
        async with self._session(kwargs) as thread:
            turn = await self.app_server.turn(thread, inputs, **self._turn_kwargs(kwargs))
            stream = turn.stream()
            completed = False
            try:
                async for event in stream:
                    if event.method == "turn/completed":
                        completed = True
                    for chunk in self._event_to_chunks(event, thread.id):
                        yield chunk
            finally:
                if not completed:
                    await self._interrupt(turn)
                await stream.aclose()
