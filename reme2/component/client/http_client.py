"""HTTP client for ReMe services."""

import json
import os

import httpx

from .base_client import BaseClient
from ..component_registry import R
from ...constants import REME_SERVICE_INFO, REME_DEFAULT_HOST, REME_DEFAULT_PORT


@R.register("http")
class HttpClient(BaseClient):
    """HTTP client for ReMe service."""

    def __init__(
        self,
        action: str,
        host: str | None = None,
        port: int | None = None,
        timeout: float = 30.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if host and port:
            pass
        elif service_info := os.environ.get(REME_SERVICE_INFO):
            try:
                data = json.loads(service_info)
                host = data.get("host", host)
                port = data.get("port", port)
            except Exception:
                pass
        else:
            host = REME_DEFAULT_HOST
            port = REME_DEFAULT_PORT

        self.action = action
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

    async def _start(self) -> None:
        """Initialize the HTTP client."""
        if self.client is None:
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )

    async def __call__(self, **_kwargs) -> dict:
        response = await self.client.post(f"/{self.action}", json=self.kwargs)
        response.raise_for_status()
        return response.json()

    async def _close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None
