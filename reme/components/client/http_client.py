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

        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

    async def _start(self) -> None:
        """Initialize the HTTP client."""
        if self.client is None:
            self.client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def _iter_stream_chunks(self, action: str, payload: dict) -> AsyncGenerator[StreamChunk, None]:
        """Send request and yield raw StreamChunks; auto-detects JSON vs SSE via Content-Type.

        For JSON responses: yields a single CONTENT chunk with the raw response body.
        For SSE responses: yields each streaming chunk as it arrives.
        """
        if self.client is None:
            raise RuntimeError("Client not initialized. Call _start() first.")

        async with self.client.stream("POST", f"/{action}", json=payload) as resp:
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")

            if ctype.startswith("text/event-stream"):
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:") :]
                    if data_str.strip() == "[DONE]":
                        return
                    try:
                        data = json.loads(data_str)
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
                yield StreamChunk(chunk_type=ChunkEnum.CONTENT, chunk=body.decode())

    async def stream_chunks(self, action: str, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """HTTP-specific richer access: yield raw StreamChunk objects (no display formatting)."""
        async for chunk in self._iter_stream_chunks(action, kwargs):
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

    @staticmethod
    def _format_for_display(text: str) -> str:
        """Render a JSON response as human-friendly CLI text; pass through unrecognized payloads."""
        try:
            data = json.loads(text)
        except (ValueError, json.JSONDecodeError):
            return text
        if not (isinstance(data, dict) and isinstance(data.get("answer"), str)):
            return json.dumps(data, indent=2, ensure_ascii=False) if isinstance(data, (dict, list)) else text
        d = dict(data)
        answer = d.pop("answer")
        success = d.pop("success", None)
        metadata = d.pop("metadata", None)
        parts = [answer]
        status_pieces = []
        if success is not None:
            status_pieces.append("✅" if success else "❌")
        if metadata:
            status_pieces.append(json.dumps(metadata, ensure_ascii=False))
        if status_pieces:
            parts.append(" ".join(status_pieces))
        if d:
            parts.append(json.dumps(d, indent=2, ensure_ascii=False))
        return "\n".join(parts)

    # pylint: disable=invalid-overridden-method
    async def _execute(self, action: str, payload: dict) -> AsyncGenerator[str, None]:
        """Yield text chunks for CLI display; JSON responses are pretty-formatted."""
        async for chunk in self._iter_stream_chunks(action, payload):
            chunk_payload = chunk.chunk
            text = chunk_payload if isinstance(chunk_payload, str) else json.dumps(chunk_payload, ensure_ascii=False)
            yield self._format_for_display(text)

    async def _close(self) -> None:
        """Close the HTTP client."""
        if self.client is not None:
            await self.client.aclose()
            self.client = None
