"""Base agent wrapper component."""

from abc import abstractmethod
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel

from ..base_component import BaseComponent
from ...enumeration import ChunkEnum, ComponentEnum
from ...schema import StreamChunk

if TYPE_CHECKING:
    from ..job.base_job import BaseJob


class BaseAgentWrapper(BaseComponent):
    """Abstract base for agent wrapper components with swappable backends."""

    component_type = ComponentEnum.AGENT_WRAPPER

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
        """Project root that contains shared assets such as skills."""
        return self.workspace_path

    @property
    def project_skills_root(self) -> Path:
        """Project-level skills directory shared by agent backends."""
        return self.project_path / "skills"

    def set_output_schema(self, schema: dict | type[BaseModel]) -> "BaseAgentWrapper":
        """Set a JSON schema for structured output. Accepts dict or BaseModel class. Returns self for chaining."""
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema = schema.model_json_schema()
        self.kwargs["output_schema"] = schema
        return self

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
        return {**self.kwargs, **kwargs}

    @staticmethod
    def _chunk(chunk_type: ChunkEnum = ChunkEnum.CONTENT, **kwargs: Any) -> StreamChunk:
        """Create a StreamChunk with a short backend-friendly call site."""
        return StreamChunk(chunk_type=chunk_type, **kwargs)

    @abstractmethod
    async def reply(self, inputs: Any, **kwargs) -> dict:
        """Send inputs to the agent and return a dict with session_id and last_message."""

    async def reply_stream(self, inputs: Any, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """Stream agent events as unified StreamChunk objects."""
