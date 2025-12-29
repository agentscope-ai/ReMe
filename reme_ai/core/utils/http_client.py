import asyncio
import json
from typing import Dict, AsyncIterator

import httpx
from httpx_sse import aconnect_sse
from loguru import logger

from ..schema import Response


class HttpClient:
    """
    Asynchronous HTTP client for executing flows with built-in retry logic, 
    exponential backoff, and specialized SSE support.
    """

    def __init__(
            self,
            base_url: str = "http://localhost:8001",
            timeout: float = 60.0,
            max_retries: int = 3,
            backoff_factor: float = 0.5,
            raise_exception: bool = True,
    ):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the service.
            timeout: Default timeout in seconds for standard requests.
            max_retries: Number of retry attempts for failed requests.
            backoff_factor: Multiplier for exponential backoff (delay = factor * 2^attempt).
            raise_exception: If True, raises errors; if False, returns None/empty on failure.
        """
        self.base_url = base_url.rstrip("/")
        # Fine-grained timeout control
        self.timeout = httpx.Timeout(timeout, connect=5.0, read=timeout)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.raise_exception = raise_exception
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy initialization of the httpx client to ensure it's bound to the correct event loop."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close the underlying transport."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def execute_flow(self, flow_name: str, **kwargs) -> Response | None:
        """
        Executes a standard POST flow with exponential backoff.
        """
        endpoint = f"{self.base_url}/{flow_name}"

        for attempt in range(self.max_retries):
            try:
                response = await self.client.post(endpoint, json=kwargs)
                response.raise_for_status()
                return Response(**response.json())

            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                wait_time = self.backoff_factor * (2 ** attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed for {flow_name}. "
                    f"Error: {e}. Retrying in {wait_time:.2f}s..."
                )

                if attempt == self.max_retries - 1:
                    if self.raise_exception:
                        raise e
                    return None

                await asyncio.sleep(wait_time)
        return None

    async def execute_stream_flow(self, flow_name: str, **kwargs) -> AsyncIterator[Dict[str, str]]:
        """
        Executes a flow and yields parsed SSE events using httpx-sse.
        
        Yields:
            A dictionary with 'type' and 'content'.
        """
        endpoint = f"{self.base_url}/{flow_name}"

        try:
            # We set timeout to None for streaming to prevent long generations from being cut off
            async with aconnect_sse(self.client, "POST", endpoint, json=kwargs, timeout=None) as event_source:
                async for event in event_source.aiter_sse():
                    # Check for explicit [DONE] signal or end of stream
                    if event.data == "[DONE]":
                        break

                    try:
                        payload = event.json()  # Automatically parses JSON data
                        yield {
                            "type": payload.get("chunk_type", "answer"),
                            "content": payload.get("chunk", "")
                        }
                    except json.JSONDecodeError:
                        # Fallback for non-JSON or plain text data
                        if event.data:
                            yield {"type": "message", "content": event.data}

        except Exception as e:
            logger.error(f"Streaming failed for {flow_name}: {e}")
            if self.raise_exception:
                raise e

    async def health_check(self) -> Dict[str, str]:
        """Verify service availability."""
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
