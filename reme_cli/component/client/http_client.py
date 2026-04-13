"""HTTP client implementation using httpx."""

from typing import Any

import httpx

from .base_client import BaseClient
from ...schema import Response


class HttpClient(BaseClient):
    """HTTP client for communicating with ReMe HTTP service.

    Provides a simple interface to invoke jobs/actions via HTTP POST requests.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 30.0, **kwargs):
        """Initialize the HTTP client.

        Args:
            base_url: Base URL of the ReMe HTTP service.
            timeout: Request timeout in seconds.
            **kwargs: Additional arguments passed to httpx.AsyncClient.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._kwargs = kwargs

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create the underlying httpx client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                **self._kwargs
            )
        return self._client

    async def invoke(self, action: str, **config: Any) -> Response:
        """Invoke an action via HTTP POST request.

        Args:
            action: The action/job endpoint name (appended to base_url).
            **config: Configuration parameters sent as JSON body.

        Returns:
            Response from the service.
        """
        response = await self.client.post(f"/{action}", json=config)
        response.raise_for_status()
        return Response.model_validate(response.json())

    async def invoke_raw(self, action: str, **config: Any) -> dict:
        """Invoke an action and return raw dict response.

        Args:
            action: The action/job endpoint name.
            **config: Configuration parameters sent as JSON body.

        Returns:
            Raw response dict.
        """
        response = await self.client.post(f"/{action}", json=config)
        response.raise_for_status()
        return response.json()

    async def health_check(self) -> bool:
        """Check if the service is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            response = await self.client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None