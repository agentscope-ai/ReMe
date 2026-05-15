"""Base client abstraction."""

from abc import abstractmethod

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum


class BaseClient(BaseComponent):
    """Abstract base for clients that communicate with ReMe services."""

    component_type = ComponentEnum.CLIENT

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.client = None

    async def _start(self) -> None:
        """Initialize the client."""

    async def _close(self) -> None:
        """Close the client and release resources."""

    @abstractmethod
    async def __call__(self) -> dict:
        """Execute the configured action and return the response."""
