"""HTTP client for ReMe services."""

import json
from collections.abc import AsyncGenerator

import httpx

from .base_client import BaseClient
from ..component_registry import R
from ...enumeration import ChunkEnum
from ...schema import StreamChunk


@R.register("http")
class HttpClient(BaseClient):
    """HTTP client that auto-adapts to JSON or SSE endpoints via Content-Type."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        timeout: float = 3600.0,
        show_metadata: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        host, port = self._resolve_service_address(host, port)
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.show_metadata = show_metadata

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
                        yield StreamChunk(chunk_type=ChunkEnum.DONE, chunk="", done=True)
                        return
                    data = json.loads(data_str)
                    chunk = StreamChunk(**data)
                    yield chunk
                    if chunk.done or chunk.chunk_type == ChunkEnum.DONE:
                        return
                raise RuntimeError("Stream ended without DONE")
            body = await resp.aread()
            yield StreamChunk(chunk_type=ChunkEnum.CONTENT, chunk=body.decode())

    async def stream_chunks(self, action: str, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """Yield the complete event stream, including errors and DONE, without display formatting."""
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

    def _format_for_display(self, text: str) -> str:
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
        if self.show_metadata and metadata:
            status_pieces.append(json.dumps(metadata, ensure_ascii=False))
        if status_pieces:
            parts.append(" ".join(status_pieces))
        if d:
            parts.append(json.dumps(d, indent=2, ensure_ascii=False))
        return "\n".join(parts)

    # pylint: disable=invalid-overridden-method
    async def _execute(self, action: str, payload: dict) -> AsyncGenerator[str, None]:
        """Yield text chunks for CLI display; JSON responses are pretty-formatted."""
        text_blocks: dict[str, str] = {}
        async for chunk in self._iter_stream_chunks(action, payload):
            if chunk.chunk_type == ChunkEnum.ERROR and chunk.metadata.get("will_retry") is not True:
                raise RuntimeError(str(chunk.chunk))
            if chunk.chunk_type == ChunkEnum.DONE:
                continue
            if chunk.chunk_type == ChunkEnum.REPLY_END and chunk.status != "completed":
                outcome = chunk.chunk if isinstance(chunk.chunk, dict) else {}
                raise RuntimeError(
                    outcome.get("error") or chunk.metadata.get("error") or f"Agent reply ended: {chunk.status}",
                )
            chunk_payload = chunk.chunk
            if chunk.chunk_type in {ChunkEnum.REPLY_START, ChunkEnum.REPLY_END}:
                continue
            if chunk_payload is None or chunk.metadata.get("visibility") == "diagnostic":
                continue
            if chunk.chunk_type == ChunkEnum.CONTENT and isinstance(chunk_payload, str) and chunk.block_id:
                previous = text_blocks.get(chunk.block_id, "")
                text_blocks[chunk.block_id] = previous + chunk_payload if chunk.event == "delta" else chunk_payload
                if chunk.event != "delta":
                    chunk_payload = (
                        chunk_payload[len(previous) :] if chunk_payload.startswith(previous) else "\n" + chunk_payload
                    )
                if not chunk_payload:
                    continue
            text = chunk_payload if isinstance(chunk_payload, str) else json.dumps(chunk_payload, ensure_ascii=False)
            yield self._format_for_display(text)

    async def _close(self) -> None:
        """Close the HTTP client."""
        if self.client is not None:
            await self.client.aclose()
            self.client = None
