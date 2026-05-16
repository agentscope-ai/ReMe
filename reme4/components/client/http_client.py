"""HTTP client for ReMe services."""

import json
import os

import httpx

from .base_client import BaseClient
from ..component_registry import R
from ...constants import REME_SERVICE_INFO, REME_DEFAULT_HOST, REME_DEFAULT_PORT


@R.register("http")
class HttpClient(BaseClient):
    """HTTP client that communicates with ReMe service via REST API."""

    def __init__(
        self,
        action: str,
        host: str | None = None,
        port: int | None = None,
        timeout: float = 30.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Resolve host/port: explicit args > env var > defaults
        if not (host and port):
            if service_info := os.environ.get(REME_SERVICE_INFO):
                try:
                    data = json.loads(service_info)
                    host = data["host"]
                    port = data["port"]
                except Exception:
                    self.logger.warning(f"Invalid service info: {service_info}")
                    host, port = REME_DEFAULT_HOST, REME_DEFAULT_PORT
            else:
                host, port = REME_DEFAULT_HOST, REME_DEFAULT_PORT

        self.action = action
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

    async def _start(self) -> None:
        """Initialize the HTTP client."""
        if self.client is None:
            self.client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def __call__(self) -> str:
        """Send POST request to the configured action endpoint."""
        if self.client is None:
            raise RuntimeError("Client not initialized. Call _start() first.")

        response = await self.client.post(f"/{self.action}", json=self.kwargs)
        response.raise_for_status()
        result = response.json()
        return json.dumps(result, indent=2, ensure_ascii=False)

    async def _close(self) -> None:
        """Close the HTTP client."""
        if self.client is not None:
            await self.client.aclose()
            self.client = None
