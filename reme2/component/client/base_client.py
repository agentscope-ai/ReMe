"""Abstract base class for client implementations."""

from abc import abstractmethod

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum


class BaseClient(BaseComponent):
    """Abstract base class for clients that communicate with ReMe services."""

    component_type = ComponentEnum.CLIENT

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.client = None

    async def _start(self, app_context=None) -> None:
        """Initialize the client."""

    async def _close(self) -> None:
        """Close the client."""

    @abstractmethod
    async def __call__(self, action: str, **kwargs) -> dict:
        """Invoke an action with the given configuration."""
