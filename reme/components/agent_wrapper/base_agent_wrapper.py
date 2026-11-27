"""Base agent wrapper component."""

from collections.abc import AsyncGenerator
from contextlib import aclosing, asynccontextmanager
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, ClassVar, Literal, TYPE_CHECKING

from pydantic import BaseModel

from .skills import SkillScope, skill_specs
from ..base import BaseComponent
from ..outbound_proxy import BaseOutboundProxy
from ...enumeration import ChunkEnum, ComponentEnum
from ...schema import AgentReply, SessionMessage, SessionSnapshot, SessionStatus, SkillSpec, StreamChunk, TokenUsage
from ...schema.agent_reply import AgentReplyStatus, collect_reply
from ...utils.counter import global_counter_add_many
from ...utils.files.workspace_files import WorkspaceFiles
from ...utils.prompt import PromptHandler
from ...utils.timezone_utils import resolve_timezone

if TYPE_CHECKING:
    from ..job.base_job import BaseJob


class BaseAgentWrapper(BaseComponent):
    """One session contract for Codex, Claude Code and AgentScope.

    Every backend forks current context independently, including during a turn.
    Immediate append submits without the generation lock; idle append waits.
    Both return only after durable persistence, with the same retry identity.

    Codex owns its native active-input queue. Claude uses the invocation's live
    SDK client, since another resumed CLI cannot update its in-memory context.
    Claude's no-query batches are serialized to prevent queued user frames
    being merged into another message's durable ID.
    AgentScope queues observations until model-input/reply boundaries, because
    concurrent observe() during SDK compression can be overwritten.

    Native ephemeral threads are a Codex feature: they inherit model context,
    but cannot expose hydrated turns or provide durable append receipts.
    Claude/AgentScope forks remain file-backed; releasing runtime resources
    never deletes history.
    The shared read-only hint restricts builtins in Claude/AgentScope; it is
    not their OS command sandbox. Native tool/network settings remain specific
    to each SDK.
    """

    component_type = ComponentEnum.AGENT_WRAPPER
    SDK_PACKAGE: ClassVar[str | None] = None
    TOKEN_COUNTER_PREFIX: ClassVar[str] = "__token_counter"

    def __init__(
        self,
        cwd: str | Path | None = None,
        project_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._cwd = cwd
        self._project_path = project_path
        language = self.app_context.app_config.language if self.app_context else ""
        self.prompt = PromptHandler(language=language)
        for cls in reversed(type(self).__mro__):
            self.prompt.load_prompt_by_class(cls)
        self.outbound_proxy = self.bind(
            "default",
            BaseOutboundProxy,
            optional=True,
        )
        if self.SDK_PACKAGE:
            try:
                sdk_version = metadata.version(self.SDK_PACKAGE)
            except metadata.PackageNotFoundError:
                sdk_version = "unknown"
            self.logger.info(f"Agent SDK name={self.name} package={self.SDK_PACKAGE} version={sdk_version}")

    @property
    def cwd(self) -> Path:
        """Working directory shared by the agent's shell and file tools.

        Defaults to the project root. Override via the ``cwd`` init argument;
        a relative value resolves against the workspace root.
        """
        if not self._cwd:
            return self.project_path
        cwd = Path(self._cwd)
        return cwd if cwd.is_absolute() else (self.workspace_path / cwd)

    def set_system_prompt(self, prompt: str) -> "BaseAgentWrapper":
        """Set the agent's system prompt. Returns self for chaining."""
        self.kwargs["system_prompt"] = prompt
        return self

    def get_session_system_prompt(self, prompt: str) -> str | dict[str, Any]:
        """Adapt a Runtime's role instructions to the backend's native prompt format."""
        return prompt

    def get_base_instructions(self) -> str:
        """Render shared defaults and backend guidance separately from role instructions."""
        timezone = self.app_context.app_config.timezone if self.app_context else None
        now = datetime.now(resolve_timezone(timezone))
        return "\n\n".join(
            self.prompt.prompt_format(
                key,
                workspace_dir=self.workspace_path,
                app_timezone=timezone or str(now.tzinfo),
                now=now.isoformat(),
            )
            for key in ("base_instructions", "backend_instructions")
            if self.prompt.has_prompt(key)
        )

    def add_job_tools(self, job_tools: list[str]) -> "BaseAgentWrapper":
        """Append job names as tools to the agent. Returns self for chaining."""
        self.kwargs.setdefault("job_tools", []).extend(job_tools)
        return self

    def set_skills(self, skills: list[SkillSpec | dict]) -> "BaseAgentWrapper":
        """Set file references or generated skill definitions."""
        self.kwargs["skills"] = [spec.model_dump(exclude_none=True) for spec in skill_specs(skills)]
        return self

    @property
    def project_path(self) -> Path:
        """Project root containing shared assets such as skills.

        A relative configured path resolves from the workspace so applications
        can keep runtime data in a subdirectory such as ``.reme`` while loading
        project assets from its parent. Defaults to the workspace.
        """
        if not self._project_path:
            return self.workspace_path
        project_path = Path(self._project_path).expanduser()
        if not project_path.is_absolute():
            project_path = self.workspace_path / project_path
        return project_path.resolve(strict=False)

    @property
    def project_skills_root(self) -> Path:
        """Project-level skills directory shared by agent backends."""
        return self.project_path / "skills"

    @asynccontextmanager
    async def skill_scope(self, **kwargs):
        """Prepare skills once for session creation, execution and follow-up calls."""
        options = self._merged_kwargs(kwargs)
        specs = skill_specs(options.get("skills", []))
        prepared = {**kwargs, "skills": [spec.model_dump(exclude_none=True) for spec in specs]}
        if (scope := options.get("_skills")) is not None:
            if not isinstance(scope, SkillScope) or not scope.active or scope.specs != specs:
                raise ValueError("Prepared skills must belong to the current run and match its definitions")
            prepared["_skills"] = scope
            yield prepared
            return
        metadata_dir = self.app_context.app_config.metadata_dir if self.app_context is not None else "metadata"
        scope = SkillScope(
            specs,
            self.project_skills_root,
            WorkspaceFiles(self.workspace_path).path(metadata_dir) / "agent_skills",
        )
        prepared["_skills"] = scope
        try:
            scope.prepare()
            yield prepared
        finally:
            scope.close()

    @property
    def subprocess_environment(self) -> dict[str, str]:
        """Configured environment variables for child agent processes."""
        if self.app_context is None:
            return {}
        return self.app_context.app_config.environment

    @property
    def command_proxy_environment(self) -> dict[str, str]:
        """Managed proxy variables for agent command tools only."""
        if not isinstance(self.outbound_proxy, BaseOutboundProxy):
            return {}
        return self.outbound_proxy.merge_environment()

    @property
    def bash_environment(self) -> dict[str, str]:
        """Configured command environment with the managed proxy applied last."""
        if not isinstance(self.outbound_proxy, BaseOutboundProxy):
            return dict(self.subprocess_environment)
        return self.outbound_proxy.merge_environment(self.subprocess_environment)

    def set_output_schema(self, schema: dict | type[BaseModel]) -> "BaseAgentWrapper":
        """Set a JSON schema for structured output. Accepts dict or BaseModel class. Returns self for chaining."""
        self.kwargs["output_schema"] = self._normalize_output_schema(schema)
        return self

    @staticmethod
    def _normalize_output_schema(schema: Any) -> dict | None:
        """Return a JSON-serializable output schema shared by every backend."""
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.model_json_schema()
        if schema is None or isinstance(schema, dict):
            return schema
        raise TypeError("output_schema must be a JSON schema dict or BaseModel class")

    @staticmethod
    def _resolve_injected_job_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Collect server-owned kwargs merged into every job tool call.

        ``injected_job_kwargs`` values are enforced constraints added by the
        wrapper after receiving the model's tool arguments; the model can
        neither see nor override them. ``tool_context_id`` keeps its dedicated
        option but is carried through the same mechanism.
        """
        injected = dict(kwargs.get("injected_job_kwargs") or {})
        if tool_context_id := kwargs.get("tool_context_id"):
            injected["tool_context_id"] = tool_context_id
        return injected

    @staticmethod
    def _merge_injected_job_kwargs(model_kwargs: dict[str, Any], injected: dict[str, Any]) -> dict[str, Any]:
        """Merge server-owned kwargs over model tool arguments, rejecting conflicts.

        Silently letting model values win would make injected constraints
        bypassable, so any overlap is an explicit error.
        """
        if conflicts := sorted(injected.keys() & model_kwargs.keys()):
            names = ", ".join(conflicts)
            raise ValueError(f"injected tool arguments cannot be provided by the model: {names}")
        return {**model_kwargs, **injected}

    @staticmethod
    def _strip_injected_parameters(parameters: dict | None, injected: dict[str, Any]) -> dict | None:
        """Hide injected keys from the tool parameter schema exposed to the model."""
        if not parameters or not injected:
            return parameters
        parameters = dict(parameters)
        if "properties" in parameters:
            parameters["properties"] = {
                name: schema for name, schema in parameters["properties"].items() if name not in injected
            }
        if "required" in parameters:
            parameters["required"] = [name for name in parameters["required"] if name not in injected]
        return parameters

    def _resolve_job_tools(self, job_tools: list[str]) -> list["BaseJob"]:
        """Resolve job name strings to BaseJob instances via app_context."""
        if not job_tools:
            return []
        if self.app_context is None:
            raise RuntimeError("Cannot resolve job_tools without an app_context")
        resolved: list["BaseJob"] = []
        for name in job_tools:
            if (job := self.app_context.jobs.get(name)) is None:
                raise KeyError(f"Job '{name}' not found in app_context.jobs")
            resolved.append(job)
        return resolved

    def _merged_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Merge component defaults with call-time kwargs; call-time values win."""
        merged = {**self.kwargs, **kwargs}
        if "output_schema" in merged:
            merged["output_schema"] = self._normalize_output_schema(merged["output_schema"])
        return merged

    @staticmethod
    def _chunk(chunk_type: ChunkEnum = ChunkEnum.CONTENT, **kwargs: Any) -> StreamChunk:
        """Create a StreamChunk with a short backend-friendly call site."""
        return StreamChunk(chunk_type=chunk_type, **kwargs)

    def _record_token_usage(self, usage: TokenUsage) -> None:
        """Add one completed invocation to the application token tree."""
        if self.app_context is None:
            return
        counters = getattr(self.app_context, "metadata", None)
        if not isinstance(counters, dict):
            return
        prefix = (self.TOKEN_COUNTER_PREFIX, self.name)
        global_counter_add_many(
            counters,
            {
                (*prefix, "input_tokens"): usage.input_tokens,
                (*prefix, "output_tokens"): usage.output_tokens,
                (*prefix, "total_tokens"): usage.total_tokens,
            },
        )

    @staticmethod
    def _serialize(value: Any) -> Any:
        """Keep native model/dataclass fields in a JSON-serializable form."""
        from pydantic_core import to_jsonable_python

        return to_jsonable_python(value, by_alias=True)

    @staticmethod
    def _reply_result(
        *,
        session_id: str,
        result: str | None,
        raw: dict[str, Any],
        status: AgentReplyStatus = "completed",
        error: str | None = None,
        structured_output: Any = None,
        usage: TokenUsage | None = None,
    ) -> AgentReply:
        """Build the same portable fields for every backend."""
        return AgentReply(
            session_id=session_id,
            result=result or "",
            structured_output=structured_output,
            usage=usage.model_dump() if usage is not None else None,
            status=status,
            error=error,
            raw=raw,
            events=[],
        )

    async def reply(self, inputs: Any, **kwargs) -> AgentReply:
        """Collect the same execution used by streaming; errors and cancellation propagate."""
        return await collect_reply(self.reply_stream(inputs, **kwargs))

    async def compact_session(self, session_id: str) -> None:
        """Request compaction of one persisted agent session."""
        raise NotImplementedError(f"{type(self).__name__} does not support session compaction")

    async def create_session(self, **kwargs) -> str:
        """Create a session while keeping its configured skills available."""
        async with self.skill_scope(**kwargs) as options:
            return await self._create_session(**options)

    async def _create_session(self, **kwargs) -> str:
        """Create a resumable session without generating a reply.

        With resume and fork_session=True, capture source context in an independent
        session without waiting for or interrupting a running source. Running
        tools are not cloned; unfinished work is closed only in the branch.
        Subsequent source messages are not automatically copied into the fork.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support session creation")

    async def release_session(self, session_id: str) -> None:
        """Release idle runtime resources; persisted session history remains available."""

    @asynccontextmanager
    async def session_guard(self, session_id: str, *, expected_revision: str):
        """Hold generation and append writers while committing a content-dependent decision."""
        raise NotImplementedError("This agent backend does not support guarded decisions")
        yield  # pylint: disable=unreachable

    async def session_exists(self, session_id: str) -> bool:
        """Check whether a session can be resumed without generating a reply."""
        raise NotImplementedError(f"{type(self).__name__} does not support session existence checks")

    async def get_session_path(self, session_id: str) -> Path | None:
        """Return the native session file path, or None when no path is available.

        The returned path may precede file creation. Lookup errors propagate.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support session path lookup")

    async def get_tool_calls(self, session_id: str, turn_id: str | None = None) -> list[dict[str, Any]] | None:
        """Read saved calls for a turn, or None when only live events are available."""

    async def read_session(self, session_id: str) -> SessionSnapshot:
        """Read display history without waiting for the session to become idle."""
        raise NotImplementedError(f"{type(self).__name__} does not support session reading")

    async def get_session_status(self, session_id: str) -> SessionStatus:
        """Read activity without resuming or waiting for the session."""
        raise NotImplementedError(f"{type(self).__name__} does not support session status")

    async def append_session(
        self,
        session_id: str,
        content: str,
        *,
        message_id: str,
        role: Literal["user", "assistant"] = "assistant",
        timeout: float = 30,
        expected_turn_id: str | None = None,
        expected_revision: str | None = None,
        delivery: str = "idle",
    ) -> SessionMessage:
        """Append a user or assistant message once without generating a reply.

        The optional revision guard rejects any changed display content; the turn guard checks only its anchor.
        Idle delivery waits for generation. Immediate delivery submits during
        generation and may wait until a native processing/SDK event boundary.

        Successful return confirms native persistence, not merely input-queue acceptance.
        Acceptance need not interrupt the current model request or start another turn.
        A timeout or cancellation after submission does not prove rejection: retry with
        the same message_id, role and content. An already accepted request keeps its
        original identity and is recovered before applying guards for a new append.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support session appending")

    async def reply_stream(self, inputs: Any, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """Keep generated skills alive until the backend stream has closed."""
        async with self.skill_scope(**kwargs) as options:
            async with aclosing(self._reply_stream(inputs, **options)) as stream:
                async for chunk in stream:
                    yield chunk

    async def _reply_stream(self, inputs: Any, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """Implement one backend execution using the prepared skill scope."""
        raise NotImplementedError
        yield  # pylint: disable=unreachable
