"""Abstract base class for client implementations."""

from abc import ABC, abstractmethod
from typing import Any

from ...schema import Response


class BaseClient(ABC):
    """Abstract base class for clients that communicate with ReMe services.

    Clients provide a unified interface for invoking jobs/actions on a
    ReMe application, whether locally or remotely via HTTP, MCP, etc.
    """

    @abstractmethod
    async def invoke(self, action: str, **config: Any) -> Response:
        """Invoke an action (job) with the given configuration.

        Args:
            action: The name of the action/job endpoint to invoke.
            **config: Configuration parameters passed as POST body.

        Returns:
            Response from the invoked action.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the client and release resources."""
        pass

    async def __aenter__(self) -> "BaseClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()