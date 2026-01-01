"""Asynchronous HTTP client for executing flows with built-in retry logic."""

import json
from collections.abc import AsyncIterator
from typing import Optional

import httpx
from loguru import logger

from ..schema import Response


class HttpClient:
    """Async client for flow endpoints with automated retries and error handling."""

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout: float = 3600.0,
        max_retries: int = 3,
        raise_exception: bool = True,
    ):
        """Initialize the client with base configuration."""
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.raise_exception = raise_exception
        self.client = httpx.AsyncClient(timeout=timeout)

    async def __aenter__(self):
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager and close connection."""
        await self.close()

    async def close(self):
        """Close the underlying HTTP client."""
        await self.client.aclose()

    async def health_check(self) -> dict[str, str]:
        """Check the health status of the flow service."""
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    async def execute_flow(self, flow_name: str, **kwargs) -> Optional[Response]:
        """Execute a flow with automated retry logic."""
        endpoint = f"{self.base_url}/{flow_name}"

        for attempt in range(self.max_retries):
            try:
                response = await self.client.post(endpoint, json=kwargs)
                response.raise_for_status()
                return Response(**response.json())

            except (httpx.HTTPError, Exception) as e:
                logger.error(f"Flow {flow_name} failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1 and self.raise_exception:
                    raise e
        return None

    async def list_endpoints(self) -> dict:
        """Retrieve available endpoints from OpenAPI specification."""
        response = await self.client.get(f"{self.base_url}/openapi.json")
        response.raise_for_status()
        return response.json()

    async def execute_stream_flow(self, flow_name: str, **kwargs) -> AsyncIterator[dict[str, str]]:
        """Execute a flow and yield parsed SSE stream chunks."""
        endpoint = f"{self.base_url}/{flow_name}"

        async with self.client.stream("POST", endpoint, json=kwargs) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue

                content = line.removeprefix("data:").strip()
                if content == "[DONE]":
                    break

                try:
                    data = json.loads(content)
                    yield {
                        "type": data.get("chunk_type", "answer"),
                        "content": data.get("chunk", ""),
                    }
                except json.JSONDecodeError:
                    continue
