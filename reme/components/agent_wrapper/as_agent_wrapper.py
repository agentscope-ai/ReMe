"""AgentScope backend for the unified agent wrapper."""

import json
import re
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from agentscope.agent import Agent, ContextConfig, ReActConfig
from agentscope.agent._config import ModelConfig
from agentscope.event import (
    DataBlockDeltaEvent,
    DataBlockEndEvent,
    DataBlockStartEvent,
    ExceedMaxItersEvent,
    ModelCallEndEvent,
    ModelCallStartEvent,
    ReplyEndEvent,
    ReplyStartEvent,
    ThinkingBlockDeltaEvent,
    ThinkingBlockEndEvent,
    ThinkingBlockStartEvent,
    TextBlockDeltaEvent,
    TextBlockEndEvent,
    TextBlockStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultDataDeltaEvent,
    ToolResultEndEvent,
    ToolResultStartEvent,
    ToolResultTextDeltaEvent,
)
from agentscope.message import TextBlock, ToolResultState, UserMsg
from agentscope.permission import PermissionBehavior, PermissionContext, PermissionDecision, PermissionMode
from agentscope.state import AgentState
from agentscope.tool import (
    Bash,
    Edit,
    FunctionTool,
    Glob,
    Grep,
    Read,
    ToolBase,
    ToolChunk,
    Toolkit,
    Write,
)

from .base_agent_wrapper import BaseAgentWrapper
from ..as_llm import BaseAsLLM
from ..component_registry import R
from ...enumeration import ChunkEnum
from ...schema import StreamChunk
from ...utils import AsStateHandler
from ...utils.env_utils import load_env

if TYPE_CHECKING:
    from ..job.base_job import BaseJob

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


class BypassAnalysisBash(Bash):
    """Bash variant that delegates permission decisions to PermissionEngine.

    AgentScope's built-in Bash performs bypass-immune static analysis before
    the engine can apply `permission_mode: bypass`. For this app we want the
    configured permission mode to be authoritative.
    """

    async def check_permissions(
        self,
        _tool_input: dict[str, Any],
        _context: PermissionContext,
    ) -> PermissionDecision:
        """Bypass Bash static analysis and let the permission engine decide."""
        return PermissionDecision(
            behavior=PermissionBehavior.PASSTHROUGH,
            message="Bash static analysis skipped; delegating to permission engine.",
        )


@R.register("agentscope")
class AsAgentWrapper(BaseAgentWrapper):
    """Agent wrapper backed by AgentScope framework."""

    def __init__(self, as_llm: str = "default", session_retention_days: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.as_llm = self.bind(as_llm, BaseAsLLM, optional=False)
        self.session_retention_days = int(session_retention_days)
        self._session_cleanup_done = False

    @staticmethod
    def _make_tool(job: "BaseJob") -> FunctionTool:
        async def run_job(**kwargs) -> ToolChunk:
            response = await job(**kwargs)
            state = ToolResultState.SUCCESS if response.success else ToolResultState.ERROR
            return ToolChunk(content=[TextBlock(text=str(response.answer))], state=state)

        tool = FunctionTool(func=run_job, name=job.name, description=job.description)
        if job.parameters:
            tool.input_schema = job.parameters
        return tool

    @classmethod
    def _builtin_tools(cls) -> list[ToolBase]:
        """Return built-in tools expected by local skills."""
        return [BypassAnalysisBash(), Edit(), Glob(), Grep(), Read(), Write()]

    @property
    def session_path(self) -> Path:
        """Directory used for persisted AgentScope sessions."""
        if self.app_context is None:
            return self.workspace_path / "session" / "agentscope"
        return self.workspace_path / self.app_context.app_config.session_dir / "agentscope"

    @staticmethod
    def _validate_session_id(session_id: str, field: str = "session_id") -> str:
        if not _UUID_RE.match(session_id):
            raise ValueError(f"{field} must be a valid UUID: {session_id!r}")
        return session_id.lower()

    def _cleanup_expired_sessions(self) -> None:
        """Delete persisted session files older than ``session_retention_days``."""
        if self._session_cleanup_done or self.session_retention_days <= 0:
            self._session_cleanup_done = True
            return

        session_path = self.session_path
        if not session_path.is_dir():
            self._session_cleanup_done = True
            return

        cutoff = time.time() - self.session_retention_days * 24 * 60 * 60
        removed = 0
        for path in session_path.glob("*.jsonl"):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError as exc:
                self.logger.warning(f"Failed to clean expired AgentScope session {path}: {exc}")

        if removed:
            self.logger.info(
                f"Cleaned {removed} AgentScope session(s) older than {self.session_retention_days} day(s)",
            )
        self._session_cleanup_done = True

    async def _load_state(self, kwargs: dict[str, Any], perm_mode: PermissionMode) -> AgentState:
        resume = kwargs.get("resume") or ""
        session_id = kwargs.get("session_id") or ""
        fork_session = bool(kwargs.get("fork_session", False))
        if resume:
            resume = self._validate_session_id(resume, "resume")
        if session_id:
            session_id = self._validate_session_id(session_id)

        if session_id and resume and not fork_session:
            raise ValueError("session_id cannot be used with resume unless fork_session=True")

        if resume:
            handler = AsStateHandler.for_session(self.session_path, resume)
            state = await handler.load_or_none()
            if state is None:
                raise FileNotFoundError(f"AgentScope session not found: {resume}")
            state.permission_context = PermissionContext(mode=perm_mode)
            state.session_id = resume
            if fork_session:
                forked = AgentState(
                    session_id=session_id or str(uuid4()),
                    summary=state.summary,
                    context=list(state.context),
                    permission_context=PermissionContext(mode=perm_mode),
                )
                return forked
            return state

        return AgentState(session_id=session_id or str(uuid4()), permission_context=PermissionContext(mode=perm_mode))

    async def _dump_state(self, state: AgentState) -> None:
        await AsStateHandler.for_session(self.session_path, state.session_id).dump(state)

    def _resolve_skills(self, skills: list[str] | str | None) -> list[str]:
        """Resolve configured skill names to AgentScope local skill directories."""
        if skills is None:
            return []
        if skills == "all":
            return [str(self.project_skills_root)]
        if isinstance(skills, str):
            skills = [skills]
        return [str(self.project_skills_root / skill) for skill in skills]

    def _load_tool_env(self) -> dict[str, str]:
        """Load project environment variables for tools spawned by AgentScope."""
        project_env = self.project_path / ".env"
        return load_env(project_env) if project_env.exists() else load_env()

    async def _build_agent(self, inputs: Any, **kwargs) -> tuple[Agent, Any]:
        """Build an Agent instance from kwargs. Returns (agent, processed_inputs)."""
        model = self.as_llm.model if self.as_llm else None
        if model is None:
            raise ValueError("AsAgentWrapper requires a bound as_llm component with a valid model.")

        kwargs = self._merged_kwargs(kwargs)
        self._cleanup_expired_sessions()
        self._load_tool_env()

        system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")
        job_tools: list[str] = kwargs.get("job_tools", [])
        resolved_jobs = self._resolve_job_tools(job_tools)
        skills = self._resolve_skills(kwargs.get("skills"))
        toolkit = kwargs.get("toolkit") or Toolkit(
            tools=[*self._builtin_tools(), *(self._make_tool(job) for job in resolved_jobs)],
            skills_or_loaders=skills,
        )

        perm_mode = PermissionMode(kwargs.get("permission_mode", "bypass"))
        state = await self._load_state(kwargs, perm_mode)

        agent = Agent(
            name=self.name,
            system_prompt=system_prompt,
            model=model,
            toolkit=toolkit,
            state=state,
            model_config=ModelConfig(**(kwargs.get("model_config") or {})),
            context_config=ContextConfig(**(kwargs.get("context_config") or {})),
            react_config=ReActConfig(**(kwargs.get("react_config") or {})),
        )

        if isinstance(inputs, str):
            inputs = UserMsg(name="user", content=inputs)

        return agent, inputs

    async def reply(self, inputs: Any, **kwargs) -> dict:
        kwargs = self._merged_kwargs(kwargs)
        agent, inputs = await self._build_agent(inputs, **kwargs)

        await agent.observe(inputs)
        await agent.reply()
        await self._dump_state(agent.state)
        last_msg = agent.state.context[-1]

        result = {
            "session_id": agent.state.session_id,
            "last_message": last_msg.model_dump(),
            "result": last_msg.get_text_content(),
        }

        output_schema: dict | None = kwargs.get("output_schema")
        if output_schema is not None:
            assert self.as_llm is not None, "AsAgentWrapper requires a bound as_llm component with a valid model."
            model = self.as_llm.model
            assert model is not None, "AsAgentWrapper requires a bound as_llm component with a valid model."
            res = await model.generate_structured_output(
                messages=agent.state.context,
                structured_model=output_schema,
            )
            result["structured_output"] = res.content

        return result

    # ----- StreamChunk conversion -------------------------------------------

    @classmethod
    # pylint: disable=too-many-return-statements
    def _event_to_chunk(cls, event: Any) -> StreamChunk | None:
        """Convert an AgentScope event to a unified StreamChunk.

        Returns ``None`` for events that should be silently skipped
        (e.g. ``RequireUserConfirmEvent``).
        """
        if isinstance(event, ReplyStartEvent):
            meta = {"reply_id": event.reply_id, "name": event.name, "role": event.role}
            return cls._chunk(ChunkEnum.REPLY_START, session_id=event.session_id, chunk="", metadata=meta)
        if isinstance(event, ReplyEndEvent):
            return cls._chunk(
                ChunkEnum.REPLY_END,
                session_id=event.session_id,
                chunk="",
                metadata={"reply_id": event.reply_id},
            )

        for event_cls, chunk_type, attr in (
            (TextBlockStartEvent, ChunkEnum.CONTENT, None),
            (TextBlockDeltaEvent, ChunkEnum.CONTENT, "delta"),
            (TextBlockEndEvent, ChunkEnum.CONTENT, None),
            (ThinkingBlockStartEvent, ChunkEnum.THINK, None),
            (ThinkingBlockDeltaEvent, ChunkEnum.THINK, "delta"),
            (ThinkingBlockEndEvent, ChunkEnum.THINK, None),
            (DataBlockStartEvent, ChunkEnum.DATA, None),
            (DataBlockDeltaEvent, ChunkEnum.DATA, "data"),
            (DataBlockEndEvent, ChunkEnum.DATA, None),
        ):
            if isinstance(event, event_cls):
                kwargs = {"block_id": event.block_id, "chunk": getattr(event, attr) if attr else ""}
                if isinstance(event, (DataBlockStartEvent, DataBlockDeltaEvent)):
                    kwargs["media_type"] = event.media_type
                return cls._chunk(chunk_type, **kwargs)

        if isinstance(event, ToolCallStartEvent):
            payload = {"name": event.tool_call_name, "id": event.tool_call_id}
            return cls._chunk(
                ChunkEnum.TOOL_CALL,
                tool_call_id=event.tool_call_id,
                tool_call_name=event.tool_call_name,
                chunk=json.dumps(payload),
            )
        if isinstance(event, ToolCallDeltaEvent):
            return cls._chunk(ChunkEnum.TOOL_CALL, tool_call_id=event.tool_call_id, chunk=event.delta)
        if isinstance(event, ToolCallEndEvent):
            return cls._chunk(ChunkEnum.TOOL_CALL, tool_call_id=event.tool_call_id, chunk="")
        if isinstance(event, ToolResultStartEvent):
            return cls._chunk(
                ChunkEnum.TOOL_RESULT,
                tool_call_id=event.tool_call_id,
                tool_call_name=event.tool_call_name,
                chunk="",
            )
        if isinstance(event, ToolResultTextDeltaEvent):
            return cls._chunk(ChunkEnum.TOOL_RESULT, tool_call_id=event.tool_call_id, chunk=event.delta)
        if isinstance(event, ToolResultDataDeltaEvent):
            return cls._chunk(
                ChunkEnum.TOOL_RESULT,
                tool_call_id=event.tool_call_id,
                chunk=event.data,
                media_type=event.media_type,
                metadata={"url": event.url} if event.url else {},
            )
        if isinstance(event, ToolResultEndEvent):
            return cls._chunk(
                ChunkEnum.TOOL_RESULT,
                tool_call_id=event.tool_call_id,
                chunk="",
                metadata={"state": str(event.state)},
            )
        if isinstance(event, ModelCallStartEvent):
            return cls._chunk(ChunkEnum.USAGE, chunk="", metadata={"model_name": getattr(event, "model_name", None)})
        if isinstance(event, ModelCallEndEvent):
            usage = {"input_tokens": event.input_tokens, "output_tokens": event.output_tokens}
            return cls._chunk(
                ChunkEnum.USAGE,
                chunk=json.dumps(usage),
                input_tokens=event.input_tokens,
                output_tokens=event.output_tokens,
                metadata={"model_name": getattr(event, "model_name", None)},
            )
        if isinstance(event, ExceedMaxItersEvent):
            return cls._chunk(ChunkEnum.ERROR, chunk="Exceeded max iterations")
        return None

    async def reply_stream(self, inputs: Any, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """Stream agent events as unified StreamChunk objects."""
        agent, inputs = await self._build_agent(inputs, **kwargs)

        async for event in agent.reply_stream(inputs):
            chunk = self._event_to_chunk(event)
            if chunk is not None:
                chunk.session_id = chunk.session_id or agent.state.session_id
                yield chunk

        await self._dump_state(agent.state)
