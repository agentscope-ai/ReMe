"""ReMe client for easy interaction with ReMe HTTP service."""

import os
from typing import Any

from .http_client import HttpClient
from .base_client import BaseClient
from ...schema import Response

# Import the environment variable name from http_service
REME_PORT_ENV = "REME_PORT"
DEFAULT_PORT = 8000
DEFAULT_HOST = "127.0.0.1"


class ReMeClient(BaseClient):
    """High-level client for ReMe HTTP service.

    Automatically reads port from environment variable REME_PORT if set.
    Provides a simple interface to call actions with config parameters.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int | None = None,
        timeout: float = 30.0,
        **kwargs
    ):
        """Initialize the ReMe client.

        Args:
            host: Host address of the ReMe service.
            port: Port number. If None, reads from REME_PORT env var,
                  or falls back to 8000.
            timeout: Request timeout in seconds.
            **kwargs: Additional arguments passed to HttpClient.
        """
        if port is None:
            port_str = os.environ.get(REME_PORT_ENV)
            port = int(port_str) if port_str else DEFAULT_PORT

        self._http_client = HttpClient(
            base_url=f"http://{host}:{port}",
            timeout=timeout,
            **kwargs
        )

    @property
    def http_client(self) -> HttpClient:
        """Get the underlying HTTP client."""
        return self._http_client

    async def invoke(self, action: str, **config: Any) -> Response:
        """Invoke an action with the given configuration.

        Args:
            action: The action/job endpoint name.
            **config: Configuration parameters passed as POST body.

        Returns:
            Response from the invoked action.
        """
        return await self._http_client.invoke(action, **config)

    async def call(self, action: str, **config: Any) -> Response:
        """Alias for invoke method for more natural usage.

        Args:
            action: The action/job endpoint name.
            **config: Configuration parameters passed as POST body.

        Returns:
            Response from the invoked action.
        """
        return await self.invoke(action, **config)

    async def health_check(self) -> bool:
        """Check if the ReMe service is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        return await self._http_client.health_check()

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http_client.close()


def get_client(host: str = DEFAULT_HOST, port: int | None = None, **kwargs) -> ReMeClient:
    """Factory function to create a ReMeClient.

    Args:
        host: Host address of the ReMe service.
        port: Port number. If None, reads from REME_PORT env var.
        **kwargs: Additional arguments passed to ReMeClient.

    Returns:
        A configured ReMeClient instance.
    """
    return ReMeClient(host=host, port=port, **kwargs)