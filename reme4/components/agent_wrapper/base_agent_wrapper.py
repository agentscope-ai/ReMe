"""Base agent wrapper component."""

from abc import abstractmethod
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum

if TYPE_CHECKING:
    from ..job.base_job import BaseJob


class BaseAgentWrapper(BaseComponent):
    """Abstract base for agent wrapper components with swappable backends."""

    component_type = ComponentEnum.AGENT_WRAPPER

    def set_system_prompt(self, prompt: str) -> "BaseAgentWrapper":
        """Set the agent's system prompt. Returns self for chaining."""
        self.kwargs["system_prompt"] = prompt
        return self

    def add_tools(self, tools: list["BaseJob"]) -> "BaseAgentWrapper":
        """Append callable tools to the agent. Returns self for chaining."""
        self.kwargs.setdefault("tools", []).extend(tools)
        return self

    def set_output_schema(self, schema: dict | type[BaseModel]) -> "BaseAgentWrapper":
        """Set a JSON schema for structured output. Accepts dict or BaseModel class. Returns self for chaining."""
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema = schema.model_json_schema()
        self.kwargs["output_schema"] = schema
        return self

    @abstractmethod
    async def reply(self, inputs: Any, session_id: str | None = None, **kwargs) -> tuple[str, Any]:
        """Send inputs to the agent and return (session_id, last_message)."""
