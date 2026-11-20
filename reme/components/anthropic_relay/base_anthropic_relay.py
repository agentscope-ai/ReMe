"""Base component for Anthropic-compatible subprocess endpoints."""

from abc import abstractmethod

from ..base import BaseComponent
from ...enumeration import ComponentEnum


class BaseAnthropicRelay(BaseComponent):
    """Provide a local Anthropic API endpoint to agent wrappers."""

    component_type = ComponentEnum.ANTHROPIC_RELAY

    @property
    @abstractmethod
    def endpoint(self) -> str:
        """Return the base URL for the running relay."""
