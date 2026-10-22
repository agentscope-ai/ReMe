"""Codex Python SDK backend for the unified agent wrapper."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, TYPE_CHECKING

from .base_agent_wrapper import BaseAgentWrapper
from .loopback_job_server import LoopbackJobServer
from ..base_component import _context_workspaces
from ..component_registry import R
from ...enumeration import ChunkEnum
from ...schema import StreamChunk, TokenUsage

if TYPE_CHECKING:
    from openai_codex import AsyncCodex, AsyncThread, CodexConfig, RunInput
    from openai_codex.types import Notification
else:
    # Keep the optional Codex SDK out of ReMe's package import path. Tests may
    # also replace this value before the SDK is loaded.
    AsyncCodex: Any = None


def _get_async_codex_class():
    """Return the optional Codex client class, importing its SDK on first use."""
    global AsyncCodex  # pylint: disable=global-statement
    if AsyncCodex is None:
        from openai_codex import AsyncCodex as AsyncCodexClass

        AsyncCodex = AsyncCodexClass
    return AsyncCodex


@dataclass(frozen=True)
class _CodexAuthConfig:
    """Resolved authentication settings for one Codex app-server."""

    mode: str
    api_key: str = ""
    base_url: str = ""


@dataclass(frozen=True)
class _CodexLaunchConfig:
    """Options fixed for the lifetime of one Codex app-server."""

    auth_mode: str
    api_key: str
    base_url: str
    codex_bin: str | None
    config_overrides: tuple[str, ...]
    experimental_api: bool


@R.register("codex")
class CodexAgentWrapper(BaseAgentWrapper):
    """Agent wrapper backed by the Codex Python SDK."""

    SDK_PACKAGE = "openai-codex"
    _CLIENT_OPTION_NAMES = frozenset(
        {
            "api_key",
            "auth_mode",
            "base_url",
            "codex_bin",
            "codex_home",
            "config_overrides",
            "cwd",
            "experimental_api",
            "launch_args_override",
        },
    )

    @staticmethod
    def _codex_usage(usage: Any) -> TokenUsage:
        """Normalize Codex's full-turn input/output usage snapshot."""
        return TokenUsage.from_provider(usage)

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        mcp_config: str | None = None,
        codex_home: str | Path | None = None,
        *,
        auth_mode: str = "auto",
        api_key: str = "",
        base_url: str = "",
        codex_bin: str | None = None,
        config_overrides: list[str] | tuple[str, ...] | None = None,
        experimental_api: bool = True,
        release_threads: bool = True,
        **kwargs,
    ) -> None:
        if "launch_args_override" in kwargs:
            raise TypeError("launch_args_override is not supported; configure codex_bin instead")
        super().__init__(**kwargs)
        self.mcp_config = mcp_config
        self._codex_home = codex_home
        self._launch_config = _CodexLaunchConfig(
            auth_mode=auth_mode,
            api_key=api_key,
            base_url=base_url,
            codex_bin=codex_bin,
            config_overrides=tuple(config_overrides or ()),
            experimental_api=experimental_api,
        )
        self._codex: AsyncCodex | None = None
        self._release_threads = release_threads
        self._turn_lock = asyncio.Lock()
        self._thread_tool_contexts: dict[str, str] = {}
        self._thread_owners: dict[str, set[str]] = {}
        self._job_server: LoopbackJobServer | None = None
        self._mcp_config_warned = False

    @property
    def session_path(self) -> Path:
        """Directory used for Codex state and persisted threads."""
        if self._codex_home:
            path = Path(self._codex_home).expanduser()
            return path if path.is_absolute() else self.workspace_path / path
        if self.app_context is None:
            return self.workspace_path / "mem_session" / "codex"
        return self.workspace_path / self.app_context.app_config.mem_session_dir / "codex"

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

    def _mcp_config_source_unused(self, kwargs: dict[str, Any]) -> None:
        """Warn once that job_tools now execute in the parent via the loopback endpoint."""
        value = kwargs.get("mcp_config") if "mcp_config" in kwargs else self.mcp_config
        if value is not None and not self._mcp_config_warned:
            self._mcp_config_warned = True
            self.logger.warning(
                "mcp_config is ignored for job_tools: Codex job tools execute in the parent "
                "process via the wrapper-owned loopback endpoint",
            )

    @staticmethod
    def _resolve_auth_config(auth_mode: str, api_key: str = "", base_url: str = "") -> _CodexAuthConfig:
        """Resolve login from wrapper options.

        The Codex child process still inherits the parent process environment.
        """
        requested_mode = str(auth_mode or "auto").lower()
        if requested_mode not in {"auto", "api_key", "oauth"}:
            raise ValueError("auth_mode must be one of: auto, api_key, oauth")
        if requested_mode == "oauth":
            return _CodexAuthConfig(mode="oauth")

        api_key = api_key if isinstance(api_key, str) else ""
        if requested_mode == "api_key" and not api_key:
            raise ValueError("auth_mode='api_key' requires a non-empty API key")
        if not api_key:
            return _CodexAuthConfig(mode="oauth")

        base_url = base_url if isinstance(base_url, str) else ""
        return _CodexAuthConfig(mode="api_key", api_key=api_key, base_url=base_url)

    def _build_client_config(self, auth: _CodexAuthConfig) -> CodexConfig:
        from openai_codex import CodexConfig

        env = dict(self.subprocess_environment)
        self.session_path.mkdir(parents=True, exist_ok=True)
        env["CODEX_HOME"] = str(self.session_path)

        overrides = list(self._launch_config.config_overrides)
        if auth.base_url:
            overrides.append(f"openai_base_url={json.dumps(auth.base_url)}")
        login_method = "api" if auth.mode == "api_key" else "chatgpt"
        overrides.append(f"forced_login_method={json.dumps(login_method)}")
        return CodexConfig(
            codex_bin=self._launch_config.codex_bin,
            config_overrides=tuple(overrides),
            cwd=str(self.cwd),
            env=env,
            client_name="reme",
            client_title="ReMe",
            experimental_api=self._launch_config.experimental_api,
        )

    @classmethod
    def _reject_client_options(cls, kwargs: dict[str, Any]) -> None:
        invalid = sorted(cls._CLIENT_OPTION_NAMES.intersection(kwargs))
        if invalid:
            names = ", ".join(invalid)
            raise TypeError(f"Codex client options must be configured on the wrapper: {names}")

    async def _ensure_job_server(self) -> LoopbackJobServer:
        """Start the wrapper-owned loopback job endpoint on first bridge use."""
        if self._job_server is None:
            if self.app_context is None:
                raise RuntimeError("Codex job_tools require an app_context")
            self._job_server = LoopbackJobServer(self.app_context)
            await self._job_server.start()
        return self._job_server

    def _mcp_server_config(self, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        from ..job import BackgroundJob, StreamJob

        self._mcp_config_source_unused(kwargs)
        job_names = list(dict.fromkeys(kwargs.get("job_tools") or []))
        if not job_names:
            return None
        jobs = self._resolve_job_tools(job_names)
        unsupported = [job.name for job in jobs if isinstance(job, (BackgroundJob, StreamJob))]
        if unsupported:
            raise TypeError(f"Codex job_tools must be non-stream request jobs: {', '.join(unsupported)}")
        if self._job_server is None:
            raise RuntimeError("Codex job endpoint is not started; call _ensure_job_server first")

        alias = str(kwargs.get("workspace") or "")
        args = [
            "-m",
            "reme.components.agent_wrapper.codex_mcp_server",
            "--endpoint",
            self._job_server.url,
            "--token",
            self._job_server.token,
        ]
        if alias:
            args.extend(["--workspace-alias", alias])
        for name in job_names:
            args.extend(["--job", name])
        args.extend(["--tool-context-id", str(kwargs.get("tool_context_id") or "")])
        injected_job_kwargs = dict(kwargs.get("injected_job_kwargs") or {})
        if workspace := kwargs.get("workspace"):
            # The bridge injects the binding into nested job calls; models cannot override it.
            injected_job_kwargs["workspace"] = workspace
        if injected_job_kwargs:
            args.extend(["--injected-job-kwargs", json.dumps(injected_job_kwargs, sort_keys=True)])
        return {
            "command": sys.executable,
            "args": args,
            "cwd": str(self.project_path),
            "required": True,
            "enabled_tools": job_names,
            "startup_timeout_sec": kwargs.get("mcp_startup_timeout", 30),
            "tool_timeout_sec": kwargs.get("mcp_tool_timeout", 300),
        }

    def _thread_config(self, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        config = dict(kwargs.get("config") or {})
        if proxy_environment := self.command_proxy_environment:
            shell_environment_policy = dict(config.get("shell_environment_policy") or {})
            environment = dict(shell_environment_policy.get("set") or {})
            environment.update(proxy_environment)
            shell_environment_policy["set"] = environment
            config["shell_environment_policy"] = shell_environment_policy
        if server := self._mcp_server_config(kwargs):
            servers = dict(config.get("mcp_servers") or {})
            server_key = hashlib.sha256(json.dumps(server, sort_keys=True).encode()).hexdigest()[:12]
            servers[f"reme_jobs_{server_key}"] = server
            config["mcp_servers"] = servers
        return config or None

    @staticmethod
    def _enum(enum_cls: Any, value: Any, default: Any = None) -> Any:
        if value is None:
            return default
        return value if isinstance(value, enum_cls) else enum_cls(value)

    async def _open_thread(self, codex: AsyncCodex, kwargs: dict[str, Any]) -> AsyncThread:
        from openai_codex import ApprovalMode, Sandbox
        from openai_codex.types import Personality, ThreadSource, ThreadStartSource

        alias = str(kwargs.get("workspace") or "")
        resume = kwargs.get("resume") or ""
        session_id = kwargs.get("session_id") or ""
        if resume and session_id and resume != session_id:
            raise ValueError("resume and session_id must identify the same Codex thread")
        thread_id = resume or session_id
        fork_session = bool(kwargs.get("fork_session", False))
        if fork_session and not thread_id:
            raise ValueError("fork_session=True requires resume or session_id")
        if thread_id:
            # Cross-workspace thread access is rejected before touching the SDK.
            await self._check_thread_ownership(thread_id, alias)
        if kwargs.get("job_tools"):
            await self._ensure_job_server()
        if self._release_threads and self._thread_tool_contexts:
            await self._prune_thread_tool_contexts(codex)
        requested_tool_context = json.dumps(
            {
                "tool_context_id": str(kwargs.get("tool_context_id") or ""),
                "injected_job_kwargs": dict(kwargs.get("injected_job_kwargs") or {}),
            },
            sort_keys=True,
            default=str,
        )
        guard_key = f"{alias}/{thread_id}" if alias else thread_id
        if not fork_session and guard_key in self._thread_tool_contexts:
            if requested_tool_context != self._thread_tool_contexts[guard_key]:
                raise ValueError("tool_context_id and injected_job_kwargs cannot change when resuming a Codex thread")

        common = {
            "approval_mode": self._enum(ApprovalMode, kwargs.get("approval_mode"), ApprovalMode.auto_review),
            "base_instructions": kwargs.get("base_instructions"),
            "config": self._thread_config(kwargs),
            "cwd": str(self._call_cwd(kwargs)),
            "developer_instructions": kwargs.get("system_prompt"),
            "model": kwargs.get("model"),
            "model_provider": kwargs.get("model_provider"),
            "sandbox": self._enum(Sandbox, kwargs.get("sandbox"), Sandbox.full_access),
            "service_tier": kwargs.get("service_tier"),
        }
        personality = self._enum(Personality, kwargs.get("personality"))
        thread_source = self._enum(ThreadSource, kwargs.get("thread_source"))
        if fork_session:
            thread = await codex.thread_fork(
                thread_id,
                ephemeral=kwargs.get("ephemeral"),
                thread_source=thread_source,
                **common,
            )
        elif thread_id:
            thread = await codex.thread_resume(thread_id, personality=personality, **common)
        else:
            thread = await codex.thread_start(
                ephemeral=kwargs.get("ephemeral"),
                personality=personality,
                service_name=kwargs.get("service_name"),
                session_start_source=self._enum(ThreadStartSource, kwargs.get("session_start_source")),
                thread_source=thread_source,
                **common,
            )
        if fork_session or not thread_id:
            # Threads created under this binding are owned by this workspace.
            self._record_thread_ownership(alias, thread.id)
        self._thread_tool_contexts[f"{alias}/{thread.id}" if alias else thread.id] = requested_tool_context
        return thread

    # ----- Thread ownership (per-workspace, persisted) ---------------------

    def _normalize_thread_alias(self, alias: str) -> str:
        """Map a call alias to a concrete workspace alias ("" means the default)."""
        workspaces = _context_workspaces(self.app_context)
        if not workspaces:
            return alias
        return alias if alias in workspaces else next(iter(workspaces))

    def _thread_ownership_path(self, alias: str) -> Path | None:
        workspaces = _context_workspaces(self.app_context)
        if not workspaces:
            return None
        workspace = workspaces.get(alias)
        if workspace is None or workspace.mem_session_path is None:
            return None
        return workspace.mem_session_path / "codex" / "owned_threads.json"

    def _owned_thread_ids(self, alias: str) -> set[str]:
        """Load (once) the persisted ids of threads this workspace created."""
        if alias in self._thread_owners:
            return self._thread_owners[alias]
        ids: set[str] = set()
        path = self._thread_ownership_path(alias)
        if path is not None and path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                ids = {str(thread_id) for thread_id in raw.get("thread_ids", [])}
            except (OSError, ValueError, AttributeError) as exc:
                self.logger.warning(f"Ignoring unreadable Codex thread ownership file {path}: {exc}")
        self._thread_owners[alias] = ids
        return ids

    def _record_thread_ownership(self, alias: str, thread_id: str) -> None:
        alias = self._normalize_thread_alias(alias)
        path = self._thread_ownership_path(alias)
        ids = self._owned_thread_ids(alias)
        if path is None or thread_id in ids:
            return
        ids.add(thread_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"thread_ids": sorted(ids)}, indent=1), encoding="utf-8")

    async def _check_thread_ownership(self, thread_id: str, alias: str) -> None:
        """Reject resume/fork/compact of a thread owned by another workspace.

        Threads recorded nowhere predate ownership tracking or came from other
        tools; they pass through and let the SDK decide.
        """
        workspaces = _context_workspaces(self.app_context)
        if not workspaces:
            return
        alias = self._normalize_thread_alias(alias)
        for owner_alias in workspaces:
            if owner_alias != alias and thread_id in self._owned_thread_ids(owner_alias):
                raise ValueError(
                    f"Codex thread {thread_id} belongs to workspace {owner_alias!r}; "
                    f"it cannot be used from workspace {alias!r}",
                )

    def _turn_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        from openai_codex import ApprovalMode, Sandbox
        from openai_codex.types import Personality, ReasoningEffort, ReasoningSummary

        return {
            "approval_mode": self._enum(ApprovalMode, kwargs.get("approval_mode")),
            "cwd": str(self._call_cwd(kwargs)),
            "effort": self._enum(ReasoningEffort, kwargs.get("effort")),
            "model": kwargs.get("model"),
            "output_schema": kwargs.get("output_schema"),
            "personality": self._enum(Personality, kwargs.get("personality")),
            "sandbox": self._enum(Sandbox, kwargs.get("sandbox")),
            "service_tier": kwargs.get("service_tier"),
            "summary": self._enum(ReasoningSummary, kwargs.get("summary")),
        }

    async def _get_codex(self) -> AsyncCodex:
        """Lazily start the component-owned client from its fixed launch configuration."""
        if self._codex is not None:
            return self._codex
        auth = self._resolve_auth_config(
            self._launch_config.auth_mode,
            self._launch_config.api_key,
            self._launch_config.base_url,
        )
        codex = _get_async_codex_class()(self._build_client_config(auth))
        try:
            if auth.mode == "api_key":
                await codex.login_api_key(auth.api_key)
            else:
                account = await codex.account()
                if account.account is None:
                    raise RuntimeError(f"No ChatGPT OAuth login found in CODEX_HOME: {self.session_path}")
        except BaseException:
            await codex.close()
            raise
        self._codex = codex
        return codex

    async def _close(self) -> None:
        """Close the persistent app-server and the wrapper-owned job endpoint."""
        async with self._turn_lock:
            codex, self._codex = self._codex, None
            self._thread_tool_contexts.clear()
            try:
                if codex is not None:
                    await codex.close()
            finally:
                if self._job_server is not None:
                    job_server, self._job_server = self._job_server, None
                    await job_server.close()

    async def _prune_thread_tool_contexts(self, codex: AsyncCodex) -> None:
        """Forget tool constraints only after the server confirms a thread is unloaded."""
        try:
            from openai_codex.generated.v2_all import ThreadLoadedListResponse

            loaded_ids: set[str] = set()
            cursor = None
            while True:
                page = await codex._client.request(  # pylint: disable=protected-access
                    "thread/loaded/list",
                    {"cursor": cursor},
                    response_model=ThreadLoadedListResponse,
                )
                loaded_ids.update(page.data)
                cursor = page.next_cursor
                if not cursor:
                    break
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.warning(f"Failed to list loaded Codex threads: {exc}")
            return
        self._thread_tool_contexts = {
            key: context for key, context in self._thread_tool_contexts.items() if key.rsplit("/", 1)[-1] in loaded_ids
        }

    async def _release_thread(self, codex: AsyncCodex, thread_id: str) -> None:
        """Unsubscribe so the app-server can unload the thread once it is idle."""
        try:
            from openai_codex.generated.v2_all import ThreadUnsubscribeResponse

            # The SDK exposes no public unsubscribe; request() is public but
            # reached through the private client attribute on SDK 0.162.x.
            await codex._client.request(  # pylint: disable=protected-access
                "thread/unsubscribe",
                {"threadId": thread_id},
                response_model=ThreadUnsubscribeResponse,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.warning(f"Failed to unsubscribe Codex thread {thread_id}: {exc}")
        # Unsubscribe is not an unload acknowledgement. In particular, a
        # cancelled turn can still be running with its original MCP constraints.
        # Keep the guard until thread/loaded/list confirms the thread is gone.

    @staticmethod
    def _serialize(value: Any) -> Any:
        from pydantic_core import to_jsonable_python

        return to_jsonable_python(value, by_alias=True)

    async def reply(self, inputs: RunInput, **kwargs) -> dict:
        """Run one Codex turn and return its final response."""
        self._reject_client_options(kwargs)
        kwargs = self._merged_kwargs(kwargs)
        self._ensure_skills(kwargs.get("skills"))
        await self.start()
        async with self._turn_lock:
            codex = await self._get_codex()
            thread = await self._open_thread(codex, kwargs)
            try:
                result = await thread.run(inputs, **self._turn_kwargs(kwargs))
            finally:
                if self._release_threads:
                    await self._release_thread(codex, thread.id)

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

    async def compact_session(self, session_id: str, workspace: str | None = None) -> None:
        """Start native compaction of a Codex thread owned by the bound workspace."""
        await self.start()
        async with self._turn_lock:
            await self._check_thread_ownership(session_id, str(workspace or ""))
            codex = await self._get_codex()
            thread = await codex.thread_resume(session_id)
            try:
                await thread.compact()
            finally:
                if self._release_threads:
                    await self._release_thread(codex, thread.id)

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
        """Stream Codex app-server notifications as unified chunks."""
        self._reject_client_options(kwargs)
        kwargs = self._merged_stream_kwargs(kwargs)
        self._ensure_skills(kwargs.get("skills"))
        await self.start()
        async with self._turn_lock:
            codex = await self._get_codex()
            thread = await self._open_thread(codex, kwargs)
            try:
                turn = await thread.turn(inputs, **self._turn_kwargs(kwargs))
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
                        try:
                            await turn.interrupt()
                        except Exception as exc:  # pylint: disable=broad-exception-caught
                            self.logger.warning(f"Failed to interrupt Codex turn {turn.id}: {exc}")
                    await stream.aclose()
            finally:
                if self._release_threads:
                    await self._release_thread(codex, thread.id)
