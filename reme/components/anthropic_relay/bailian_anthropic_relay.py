"""Anthropic Messages relay for Bailian's server-side search tools."""

import json
import re
import secrets

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from .base_anthropic_relay import BaseAnthropicRelay
from ...core.registry import R
from ..service.loopback_http import LoopbackHTTP

_HOP_HEADERS = {
    b"connection",
    b"keep-alive",
    b"proxy-authenticate",
    b"proxy-authorization",
    b"proxy-connection",
    b"te",
    b"trailer",
    b"transfer-encoding",
    b"upgrade",
}
_BILLING_PREFIX = "x-anthropic-billing-header:"
_ENTRYPOINT = re.compile(r"\bcc_entrypoint=[^;\s]*")


def _end_to_end_headers(headers: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    headers = [(name.lower(), value) for name, value in headers]
    connection = {
        item.strip().lower() for name, value in headers if name == b"connection" for item in value.split(b",")
    }
    excluded = _HOP_HEADERS | connection
    return [(name, value) for name, value in headers if name not in excluded]


def _search_request(body: bytes) -> bytes:
    """Set the Bailian entrypoint for a hosted search while preserving prompt blocks."""
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeError):
        return body
    tools = payload.get("tools") if isinstance(payload, dict) else None
    if not isinstance(tools, list) or not any(
        isinstance(tool, dict)
        and tool.get("name") == "web_search"
        and str(tool.get("type", "")).startswith("web_search_")
        for tool in tools
    ):
        return body

    system = payload.get("system")
    if isinstance(system, str):
        system = [{"type": "text", "text": system}]
    elif system is None:
        system = []
    if not isinstance(system, list):
        return body

    for block in system:
        text = block.get("text") if isinstance(block, dict) else None
        if isinstance(text, str) and text.startswith(_BILLING_PREFIX):
            if _ENTRYPOINT.search(text):
                block["text"] = _ENTRYPOINT.sub("cc_entrypoint=cli", text, count=1)
            else:
                text = text.rstrip()
                separator = " " if text.endswith((":", ";")) else "; "
                block["text"] = f"{text}{separator}cc_entrypoint=cli;"
            break
    else:
        system.insert(0, {"type": "text", "text": f"{_BILLING_PREFIX} cc_entrypoint=cli;"})
    payload["system"] = system
    return json.dumps(payload, ensure_ascii=False).encode()


@R.register("bailian")
class BailianAnthropicRelay(BaseAnthropicRelay):
    """Forward Claude Code requests to a fixed upstream and enable hosted WebSearch."""

    def __init__(self, base_url: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._upstream = httpx.URL(base_url.rstrip("/"))
        if (
            self._upstream.scheme not in {"http", "https"}
            or not self._upstream.host
            or self._upstream.userinfo
            or self._upstream.query
            or self._upstream.fragment
        ):
            raise ValueError("Anthropic relay base_url must be an HTTP(S) base URL without credentials or query")
        self._path = f"/_reme/anthropic/{secrets.token_urlsafe(24)}"
        self._listener = LoopbackHTTP(self, lifespan="off", shutdown_timeout=5)
        self._endpoint: str | None = None
        self._client: httpx.AsyncClient | None = None

    @property
    def endpoint(self) -> str:
        """Return the protected loopback address once the component starts."""
        if self._endpoint is None:
            raise RuntimeError("Anthropic relay is not running")
        return self._endpoint

    async def _start(self) -> None:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30, read=300))
        self._endpoint = f"{await self._listener.listen()}{self._path}"

    async def _close(self) -> None:
        self._endpoint = None
        try:
            await self._listener.close()
        finally:
            if self._client is not None:
                await self._client.aclose()
                self._client = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        prefix = self._path.encode()
        raw_path = scope["raw_path"]
        if not raw_path.startswith(prefix + b"/"):
            await Response(status_code=404)(scope, receive, send)
            return

        request = Request(scope, receive)
        path = raw_path[len(prefix) :]
        query = scope["query_string"]
        target = self._upstream.raw_path.rstrip(b"/") + path
        url = self._upstream.copy_with(raw_path=target + (b"?" + query if query else b""))
        headers = [
            (key, value)
            for key, value in _end_to_end_headers(scope["headers"])
            if key not in {b"host", b"content-length"}
        ]
        body = await request.body()
        if request.method == "POST" and path == b"/v1/messages" and not request.headers.get("content-encoding"):
            body = _search_request(body)
        try:
            upstream = await self._client.send(
                httpx.Request(request.method, url, headers=headers, content=body),
                stream=True,
            )
        except httpx.RequestError:
            await JSONResponse(
                {"type": "error", "error": {"type": "api_error", "message": "Bailian relay could not reach upstream"}},
                status_code=502,
            )(scope, receive, send)
            return
        try:
            response = StreamingResponse(upstream.aiter_raw(), status_code=upstream.status_code)
            response.raw_headers = _end_to_end_headers(upstream.headers.raw)
            await response(scope, receive, send)
        finally:
            await upstream.aclose()
