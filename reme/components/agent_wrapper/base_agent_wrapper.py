"""Base agent wrapper component."""

from abc import abstractmethod
from collections.abc import AsyncGenerator
from importlib import metadata
from pathlib import Path
from typing import Any, ClassVar, TYPE_CHECKING

from pydantic import BaseModel

from ..base_component import BaseComponent
from ...enumeration import ChunkEnum, ComponentEnum
from ...schema import StreamChunk

if TYPE_CHECKING:
    from ..job.base_job import BaseJob


class BaseAgentWrapper(BaseComponent):
    """Abstract base for agent wrapper components with swappable backends."""

    component_type = ComponentEnum.AGENT_WRAPPER
    SDK_PACKAGE: ClassVar[str | None] = None

    def __init__(
        self,
        cwd: str | Path | None = None,
        project_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._cwd = cwd
        self._project_path = project_path
        if self.SDK_PACKAGE:
            try:
                sdk_version = metadata.version(self.SDK_PACKAGE)
            except metadata.PackageNotFoundError:
                sdk_version = "unknown"
            self.logger.info(f"Agent SDK package={self.SDK_PACKAGE} version={sdk_version}")

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

    def add_job_tools(self, job_tools: list[str]) -> "BaseAgentWrapper":
        """Append job names as tools to the agent. Returns self for chaining."""
        self.kwargs.setdefault("job_tools", []).extend(job_tools)
        return self

    def add_skills(self, skills: list[str] | str) -> "BaseAgentWrapper":
        """Set agent skill names. Returns self for chaining."""
        self.kwargs["skills"] = skills
        return self

    @property
    def project_path(self) -> Path:
        """Project root containing shared assets such as skills.

        A relative configured path resolves from the workspace so applications
        can keep runtime data in a subdirectory such as ``.reme`` while loading
        project assets from its parent. The workspace remains the default for
        backward compatibility.
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

    def _resolve_project_skills(self, skills: list[str] | str | None) -> dict[str, Path]:
        """Resolve selected skill names to validated project directories."""
        if skills is None:
            return {}
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

        sources: dict[str, Path] = {}
        for name in names:
            if not name or Path(name).name != name or name in {".", ".."}:
                raise ValueError(f"Invalid skill name: {name!r}")
            source = self.project_skills_root / name
            if not source.is_dir():
                raise FileNotFoundError(f"Skill directory not found: {source}")
            if not (source / "SKILL.md").is_file():
                raise FileNotFoundError(f"Skill '{name}' is missing SKILL.md: {source}")
            sources[name] = source
        return sources

    @property
    def subprocess_environment(self) -> dict[str, str]:
        """Configured environment variables for child agent processes."""
        if self.app_context is None:
            return {}
        return self.app_context.app_config.environment

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

    def _merged_stream_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Merge stream options and reject unsupported structured output."""
        merged = self._merged_kwargs(kwargs)
        if merged.get("output_schema") is not None:
            raise NotImplementedError("Structured output is not supported by reply_stream()")
        return merged

    @staticmethod
    def _chunk(chunk_type: ChunkEnum = ChunkEnum.CONTENT, **kwargs: Any) -> StreamChunk:
        """Create a StreamChunk with a short backend-friendly call site."""
        return StreamChunk(chunk_type=chunk_type, **kwargs)

    @abstractmethod
    async def reply(self, inputs: Any, **kwargs) -> dict:
        """Send inputs to the agent and return a dict with session_id and last_message."""

    async def reply_stream(self, inputs: Any, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """Stream agent events as unified StreamChunk objects."""
