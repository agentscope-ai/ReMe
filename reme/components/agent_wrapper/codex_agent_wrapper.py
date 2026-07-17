"""Codex Python SDK backend for the unified agent wrapper."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import suppress
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from .base_agent_wrapper import BaseAgentWrapper
from ..component_registry import R
from ...enumeration import ChunkEnum
from ...schema import StreamChunk
from ...utils.env_utils import load_env


@R.register("codex")
class CodexAgentWrapper(BaseAgentWrapper):
    """Agent wrapper backed by the Codex Python SDK."""

    def __init__(self, mcp_config: str | None = None, codex_home: str | Path | None = None, **kwargs):
        super().__init__(**kwargs)
        self.mcp_config = mcp_config
        self._codex_home = codex_home
        self._codex: Any | None = None
        self._codex_config: Any | None = None
        self._client_lock = asyncio.Lock()
        self._turn_lock = asyncio.Lock()
        self._mcp_snapshot_path: Path | None = None
        self._thread_tool_contexts: dict[str, str] = {}

    @staticmethod
    def _first_non_empty(*values: Any) -> str:
        for value in values:
            if isinstance(value, str) and value:
                return value
        return ""

    def _default_llm_credential(self) -> dict[str, Any]:
        if self.app_context is None:
            return {}
        from ...enumeration import ComponentEnum

        llm_configs = self.app_context.app_config.components.get(ComponentEnum.AS_LLM)
        if not isinstance(llm_configs, dict):
            return {}
        default_llm = llm_configs.get("default")
        credential = getattr(default_llm, "credential", None)
        return credential if isinstance(credential, dict) else {}

    @property
    def session_path(self) -> Path:
        """Directory used for Codex state and persisted threads."""
        if self._codex_home:
            path = Path(self._codex_home).expanduser()
            return path if path.is_absolute() else self.workspace_path / path
        if self.app_context is None:
            return self.workspace_path / "mem_session" / "codex"
        return self.workspace_path / self.app_context.app_config.mem_session_dir / "codex"

    @property
    def wrapper_session_path(self) -> Path:
        """ReMe-owned session data, kept separate from a shared OAuth CODEX_HOME."""
        if self.app_context is None:
            return self.workspace_path / "mem_session" / "codex"
        return self.workspace_path / self.app_context.app_config.mem_session_dir / "codex"

    def _ensure_skills(self, skills: list[str] | str | None) -> None:
        """Expose selected project skills through Codex's repo-level directory."""
        if skills is None:
            return
        if skills == "all":
            if not self.project_skills_root.is_dir():
                raise FileNotFoundError(f"Project skills directory not found: {self.project_skills_root}")
            names = sorted(
                path.name
                for path in self.project_skills_root.iterdir()
                if path.is_dir() and (path / "SKILL.md").is_file()
            )
        else:
            names = [skills] if isinstance(skills, str) else list(skills)
            names = list(dict.fromkeys(names))

        target_root = self.workspace_path / ".agents" / "skills"
        target_root.mkdir(parents=True, exist_ok=True)
        for name in names:
            if not name or Path(name).name != name or name in {".", ".."}:
                raise ValueError(f"Invalid skill name: {name!r}")
            source = self.project_skills_root / name
            if not source.is_dir():
                raise FileNotFoundError(f"Skill directory not found: {source}")
            if not (source / "SKILL.md").is_file():
                raise FileNotFoundError(f"Skill '{name}' is missing SKILL.md: {source}")

            target = target_root / name
            if target.is_symlink():
                if target.resolve() == source.resolve():
                    continue
                raise FileExistsError(f"Codex skill conflict: {target} points to {target.resolve(strict=False)}")
            if target.exists():
                raise FileExistsError(f"Codex skill conflict: {target} already exists and was preserved")
            relative_source = os.path.relpath(source, target.parent)
            target.symlink_to(relative_source, target_is_directory=True)

    def _explicit_mcp_config(self, kwargs: dict[str, Any]) -> str | None:
        value = kwargs.get("mcp_config") if "mcp_config" in kwargs else self.mcp_config
        if value is None:
            return None
        source = Path(str(value)).expanduser()
        if source.suffix in {".yaml", ".yml", ".json"}:
            if not source.is_absolute():
                source = self.workspace_path / source
            return str(source.absolute())
        return str(value)

    def _effective_config_snapshot(self) -> Path:
        """Create one private snapshot that remains valid for the client lifetime."""
        if self._mcp_snapshot_path is not None:
            return self._mcp_snapshot_path
        if self.app_context is None:
            raise RuntimeError("Cannot snapshot MCP config without an app_context")
        snapshot_dir = self.wrapper_session_path / "reme-mcp"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        fd, raw_path = tempfile.mkstemp(prefix="config-", suffix=".json", dir=snapshot_dir)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self.app_context.app_config.model_dump(mode="json"), stream)
        except BaseException:
            with suppress(OSError):
                os.close(fd)
            Path(raw_path).unlink(missing_ok=True)
            raise
        self._mcp_snapshot_path = Path(raw_path)
        return self._mcp_snapshot_path

    def _mcp_config_source(self, kwargs: dict[str, Any]) -> str:
        return self._explicit_mcp_config(kwargs) or str(self._effective_config_snapshot())

    def _build_client_config(self, kwargs: dict[str, Any]):
        from openai_codex import CodexConfig

        credential = kwargs.get("credential") if isinstance(kwargs.get("credential"), dict) else {}
        default_credential = self._default_llm_credential()
        api_key = self._first_non_empty(
            kwargs.get("api_key"),
            credential.get("api_key"),
            os.getenv("CODEX_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
            os.getenv("LLM_API_KEY"),
            default_credential.get("api_key"),
        )
        base_url = self._first_non_empty(
            kwargs.get("base_url"),
            credential.get("base_url"),
            os.getenv("CODEX_BASE_URL"),
            os.getenv("OPENAI_BASE_URL"),
            os.getenv("LLM_BASE_URL"),
            default_credential.get("base_url"),
        )

        project_env = self.project_path / ".env"
        env = load_env(project_env) if project_env.exists() else load_env()
        self.session_path.mkdir(parents=True, exist_ok=True)
        env["CODEX_HOME"] = str(self.session_path)
        if api_key:
            env["OPENAI_API_KEY"] = api_key

        overrides = list(kwargs.get("config_overrides") or [])
        if base_url:
            overrides.append(f"openai_base_url={json.dumps(base_url)}")
        return CodexConfig(
            codex_bin=kwargs.get("codex_bin"),
            config_overrides=tuple(overrides),
            cwd=str(self.cwd),
            env=env,
            client_name="reme",
            client_title="ReMe",
        )

    def _mcp_server_config(self, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        from ..job import BackgroundJob, StreamJob

        job_names = list(kwargs.get("job_tools") or [])
        if not job_names:
            return None
        jobs = self._resolve_job_tools(job_names)
        unsupported = [job.name for job in jobs if isinstance(job, (BackgroundJob, StreamJob))]
        if unsupported:
            raise TypeError(f"Codex job_tools must be non-stream request jobs: {', '.join(unsupported)}")

        config_source = self._mcp_config_source(kwargs)
        return {
            "command": sys.executable,
            "args": [
                "-m",
                "reme.components.agent_wrapper.codex_mcp_server",
                "--config",
                config_source,
                "--workspace",
                str(self.workspace_path),
                "--jobs",
                json.dumps(job_names),
                "--tool-context-id",
                str(kwargs.get("tool_context_id") or ""),
            ],
            "cwd": str(self.project_path),
            "required": True,
            "enabled_tools": job_names,
            "startup_timeout_sec": kwargs.get("mcp_startup_timeout", 30),
            "tool_timeout_sec": kwargs.get("mcp_tool_timeout", 300),
        }

    def _thread_config(self, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        config = dict(kwargs.get("config") or {})
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

    async def _open_thread(self, codex: Any, kwargs: dict[str, Any]):
        from openai_codex import ApprovalMode, Sandbox

        resume = kwargs.get("resume") or ""
        session_id = kwargs.get("session_id") or ""
        if resume and session_id and resume != session_id:
            raise ValueError("resume and session_id must identify the same Codex thread")
        thread_id = resume or session_id
        fork_session = bool(kwargs.get("fork_session", False))
        if fork_session and not thread_id:
            raise ValueError("fork_session=True requires resume or session_id")
        requested_tool_context = str(kwargs.get("tool_context_id") or "")
        if not fork_session and thread_id in self._thread_tool_contexts:
            if requested_tool_context != self._thread_tool_contexts[thread_id]:
                raise ValueError("tool_context_id cannot change when resuming a Codex thread")

        common = {
            "approval_mode": self._enum(ApprovalMode, kwargs.get("approval_mode"), ApprovalMode.auto_review),
            "base_instructions": kwargs.get("base_instructions"),
            "config": self._thread_config(kwargs),
            "cwd": str(self.cwd),
            "developer_instructions": kwargs.get("system_prompt"),
            "model": kwargs.get("model"),
            "model_provider": kwargs.get("model_provider"),
            "sandbox": self._enum(Sandbox, kwargs.get("sandbox"), Sandbox.full_access),
            "service_tier": kwargs.get("service_tier"),
        }
        if fork_session:
            thread = await codex.thread_fork(thread_id, ephemeral=kwargs.get("ephemeral"), **common)
        elif thread_id:
            thread = await codex.thread_resume(thread_id, **common)
        else:
            thread = await codex.thread_start(ephemeral=kwargs.get("ephemeral"), **common)
        self._thread_tool_contexts[thread.id] = requested_tool_context
        return thread

    def _turn_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        from openai_codex import ApprovalMode, Sandbox
        from openai_codex.generated.v2_all import Personality, ReasoningEffort, ReasoningSummary

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

    async def _get_codex(self, kwargs: dict[str, Any]) -> Any:
        """Lazily start one app-server and reject launch-config changes while it is live."""
        from openai_codex import AsyncCodex

        config = self._build_client_config(kwargs)
        async with self._client_lock:
            if self._codex is not None:
                if config != self._codex_config:
                    raise RuntimeError("Codex client configuration changed; close the wrapper before reconfiguring it")
                return self._codex
            codex = AsyncCodex(config)
            self._codex = codex
            self._codex_config = config
            return codex

    async def _close(self) -> None:
        """Close the persistent app-server and remove its private config snapshot."""
        async with self._client_lock:
            codex, self._codex = self._codex, None
            self._codex_config = None
            self._thread_tool_contexts.clear()
            try:
                if codex is not None:
                    await codex.close()
            finally:
                if self._mcp_snapshot_path is not None:
                    self._mcp_snapshot_path.unlink(missing_ok=True)
                    self._mcp_snapshot_path = None

    @classmethod
    def _serialize(cls, value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json", by_alias=True)
        if isinstance(value, Enum):
            return value.value
        if is_dataclass(value):
            return {field.name: cls._serialize(getattr(value, field.name)) for field in fields(value)}
        if isinstance(value, dict):
            return {key: cls._serialize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._serialize(item) for item in value]
        return value

    async def reply(self, inputs: Any, **kwargs) -> dict:
        """Run one Codex turn and return its final response."""
        if not isinstance(inputs, str):
            raise NotImplementedError("Only string input is supported for Codex.")
        kwargs = self._merged_kwargs(kwargs)
        self._ensure_skills(kwargs.get("skills"))
        async with self._turn_lock:
            codex = await self._get_codex(kwargs)
            thread = await self._open_thread(codex, kwargs)
            result = await thread.run(inputs, **self._turn_kwargs(kwargs))

        final_response = result.final_response or ""
        response = {
            "session_id": thread.id,
            "last_message": final_response,
            "result": final_response,
            "turn": self._serialize(result),
        }
        if kwargs.get("output_schema") is not None:
            try:
                response["structured_output"] = json.loads(final_response)
            except json.JSONDecodeError as exc:
                raise ValueError("Codex returned invalid JSON for the requested output_schema") from exc
        return response

    @classmethod
    def _item_data(cls, item: Any) -> tuple[Any, str, str]:
        item = item.root if hasattr(item, "root") else item
        return item, getattr(item, "type", ""), getattr(item, "id", "")

    @classmethod
    # pylint: disable=too-many-return-statements
    def _event_to_chunks(cls, event: Any, session_id: str) -> list[StreamChunk]:
        """Convert one Codex app-server notification to unified stream chunks."""
        method, payload = event.method, event.payload
        if method == "turn/started":
            return [cls._chunk(ChunkEnum.REPLY_START, session_id=session_id, metadata={"turn_id": payload.turn.id})]
        if method == "item/agentMessage/delta":
            return [cls._chunk(ChunkEnum.CONTENT, session_id=session_id, block_id=payload.item_id, chunk=payload.delta)]
        if method in {"item/reasoning/summaryTextDelta", "item/reasoning/textDelta", "item/plan/delta"}:
            return [cls._chunk(ChunkEnum.THINK, session_id=session_id, block_id=payload.item_id, chunk=payload.delta)]
        if method in {"item/commandExecution/outputDelta", "item/fileChange/outputDelta"}:
            return [
                cls._chunk(
                    ChunkEnum.TOOL_RESULT,
                    session_id=session_id,
                    block_id=payload.item_id,
                    tool_call_id=payload.item_id,
                    chunk=payload.delta,
                ),
            ]
        if method == "item/mcpToolCall/progress":
            return [
                cls._chunk(
                    ChunkEnum.TOOL_RESULT,
                    session_id=session_id,
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
                cls._chunk(
                    ChunkEnum.APPROVAL,
                    session_id=session_id,
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
            item, item_type, item_id = cls._item_data(payload.item)
            tool_types = {"commandExecution", "fileChange", "mcpToolCall", "dynamicToolCall", "collabAgentToolCall"}
            if item_type not in tool_types:
                return []
            name = getattr(item, "tool", None) or item_type
            chunk_type = ChunkEnum.TOOL_CALL if method == "item/started" else ChunkEnum.TOOL_RESULT
            return [
                cls._chunk(
                    chunk_type,
                    session_id=session_id,
                    block_id=item_id,
                    tool_call_id=item_id,
                    tool_call_name=name,
                    chunk=cls._serialize(item),
                ),
            ]
        if method == "thread/tokenUsage/updated":
            usage = payload.token_usage.last
            data = cls._serialize(usage)
            return [
                cls._chunk(
                    ChunkEnum.USAGE,
                    session_id=session_id,
                    chunk=data,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                ),
            ]
        if method == "error":
            return [
                cls._chunk(
                    ChunkEnum.ERROR,
                    session_id=session_id,
                    chunk=payload.error.message,
                    metadata={"will_retry": payload.will_retry},
                ),
            ]
        if method == "turn/completed":
            turn = payload.turn
            chunks = []
            if getattr(turn, "error", None):
                chunks.append(cls._chunk(ChunkEnum.ERROR, session_id=session_id, chunk=turn.error.message))
            chunks.append(
                cls._chunk(
                    ChunkEnum.REPLY_END,
                    session_id=session_id,
                    metadata={
                        "turn_id": turn.id,
                        "status": getattr(turn.status, "value", str(turn.status)),
                        "duration_ms": turn.duration_ms,
                    },
                ),
            )
            return chunks
        return []

    async def reply_stream(self, inputs: Any, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """Stream Codex app-server notifications as unified chunks."""
        if not isinstance(inputs, str):
            raise NotImplementedError("Only string input is supported for Codex.")
        kwargs = self._merged_stream_kwargs(kwargs)
        self._ensure_skills(kwargs.get("skills"))
        async with self._turn_lock:
            codex = await self._get_codex(kwargs)
            thread = await self._open_thread(codex, kwargs)
            turn = await thread.turn(inputs, **self._turn_kwargs(kwargs))
            async for event in turn.stream():
                for chunk in self._event_to_chunks(event, thread.id):
                    yield chunk
