"""HTTP client for ReMe services."""

import json
import os
from collections.abc import AsyncGenerator

import httpx

from .base_client import BaseClient
from ..component_registry import R
from ...constants import REME_SERVICE_INFO, REME_DEFAULT_HOST, REME_DEFAULT_PORT
from ...enumeration import ChunkEnum
from ...schema import StreamChunk


@R.register("http")
class HttpClient(BaseClient):
    """HTTP client that auto-adapts to JSON or SSE endpoints via Content-Type."""

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

    async def _iter_stream_chunks(self) -> AsyncGenerator[StreamChunk, None]:
        """Send request and yield StreamChunks; auto-detects JSON vs SSE via Content-Type."""
        if self.client is None:
            raise RuntimeError("Client not initialized. Call _start() first.")

        async with self.client.stream("POST", f"/{self.action}", json=self.kwargs) as resp:
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")

            if ctype.startswith("text/event-stream"):
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :]
                    if payload.strip() == "[DONE]":
                        return
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    chunk = StreamChunk(**data)
                    if chunk.chunk_type == ChunkEnum.ERROR:
                        # Surface server-side errors as exceptions so callers don't
                        # mistake error chunks for valid content.
                        raise RuntimeError(str(chunk.chunk))
                    if chunk.done:
                        return
                    yield chunk
            else:
                body = await resp.aread()
                text = body.decode()
                try:
                    data = json.loads(text)
                    pretty = json.dumps(data, indent=2, ensure_ascii=False)
                except json.JSONDecodeError:
                    pretty = text
                yield StreamChunk(chunk_type=ChunkEnum.CONTENT, chunk=pretty)

    async def stream_chunks(self) -> AsyncGenerator[StreamChunk, None]:
        """HTTP-specific richer access: yield full StreamChunk objects with chunk_type/metadata."""
        async for chunk in self._iter_stream_chunks():
            yield chunk

    async def list_actions(self) -> list[dict]:
        """Return raw OpenAPI operations; each dict gets an `action` key (path without leading '/')."""
        if self.client is None:
            raise RuntimeError("Client not initialized. Call _start() first.")
        resp = await self.client.get("/openapi.json")
        resp.raise_for_status()
        spec = resp.json()
        actions: list[dict] = []
        for path, methods in spec.get("paths", {}).items():
            for method, op in methods.items():
                actions.append({"action": path.lstrip("/"), "method": method.upper(), **op})
        return actions

    # pylint: disable=invalid-overridden-method
    async def _execute(self) -> AsyncGenerator[str, None]:
        """Yield text chunks; one yield for JSON endpoints, many for SSE."""
        async for chunk in self._iter_stream_chunks():
            payload = chunk.chunk
            yield payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)

    async def _close(self) -> None:
        """Close the HTTP client."""
        if self.client is not None:
            await self.client.aclose()
            self.client = None
