"""Wrapper-owned loopback job endpoint for agent-bridge subprocesses.

The Codex job bridge used to spawn a full child Application per thread; every
thread then held its own workspace stores and became a second persistence
writer. This endpoint keeps job execution in the parent process instead: a
minimal HTTP/JSON server on 127.0.0.1 with an ephemeral port and a bearer
token, owned by one CodexAgentWrapper instance for its lifetime. Bridge
subprocesses are thin FastMCP forwarders that call this endpoint.
"""

import asyncio
import json
import secrets
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..application_context import ApplicationContext
    from ..job import BaseJob

_REASON = {
    200: "OK",
    400: "Bad Request",
    401: "Unauthorized",
    404: "Not Found",
    500: "Internal Server Error",
}


class LoopbackJobServer:
    """Serve this application's request jobs to trusted local subprocesses."""

    def __init__(self, app_context: "ApplicationContext") -> None:
        self._app_context = app_context
        self.token = secrets.token_urlsafe(24)
        self._server: asyncio.Server | None = None

    @property
    def is_running(self) -> bool:
        """Whether the endpoint is currently bound."""
        return self._server is not None

    @property
    def url(self) -> str:
        """Base URL of the bound endpoint."""
        if self._server is None:
            raise RuntimeError("Loopback job server is not started")
        port = self._server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}"

    async def start(self) -> None:
        """Bind the endpoint; idempotent so lazy bridge creation can retry."""
        if self._server is None:
            self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)

    async def close(self) -> None:
        """Stop the endpoint and release the port."""
        server, self._server = self._server, None
        if server is not None:
            server.close()
            await server.wait_closed()

    # ----- HTTP handling ---------------------------------------------------

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=30)
            request_line, *header_lines = head.decode("latin-1").split("\r\n")
            method, target, _version = request_line.split(" ", 2)
            headers = {}
            for line in header_lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    headers[key.strip().lower()] = value.strip()
            length = int(headers.get("content-length", 0) or 0)
            body = await reader.readexactly(length) if length else b""
            status, payload = await self._dispatch(method, target, headers, body)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            status, payload = 500, {"error": f"{type(exc).__name__}: {exc}"}
        data = json.dumps(payload).encode()
        writer.write(
            f"HTTP/1.1 {status} {_REASON.get(status, 'OK')}\r\n"
            "content-type: application/json\r\n"
            f"content-length: {len(data)}\r\n"
            "connection: close\r\n\r\n".encode() + data,
        )
        try:
            await writer.drain()
        finally:
            writer.close()

    async def _dispatch(
        self,
        method: str,
        target: str,
        headers: dict[str, str],
        body: bytes,
    ) -> tuple[int, dict[str, Any]]:
        if headers.get("authorization") != f"Bearer {self.token}":
            return 401, {"error": "unauthorized"}
        if method == "GET" and target == "/jobs":
            return 200, {"jobs": self._job_metadata()}
        if method == "POST" and target == "/call":
            try:
                request = json.loads(body or b"{}")
            except json.JSONDecodeError as exc:
                return 400, {"error": f"invalid JSON body: {exc}"}
            return await self._call(request)
        return 404, {"error": "not found"}

    # ----- Job surface -------------------------------------------------------

    def _request_jobs(self) -> dict[str, "BaseJob"]:
        """Return the application's non-stream, non-background jobs by name."""
        from ..job import BackgroundJob, StreamJob

        return {
            name: job for name, job in self._app_context.jobs.items() if not isinstance(job, (StreamJob, BackgroundJob))
        }

    def _job_metadata(self) -> list[dict[str, Any]]:
        """Describe every servable job for bridge tool registration."""
        return [
            {"name": name, "description": job.description, "parameters": job.parameters}
            for name, job in sorted(self._request_jobs().items())
        ]

    async def _call(self, request: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        name = str(request.get("job") or "")
        job = self._request_jobs().get(name)
        if job is None:
            return 404, {"error": f"job {name!r} not found or not a request job"}

        kwargs = dict(request.get("kwargs") or {})
        injected = dict(request.get("injected_job_kwargs") or {})
        conflicts = sorted(injected.keys() & kwargs.keys())
        if conflicts:
            return 400, {"error": f"{', '.join(conflicts)} injected by the server and cannot be provided"}
        kwargs.update(injected)
        if workspace := request.get("workspace"):
            kwargs["workspace"] = workspace
        if tool_context_id := request.get("tool_context_id"):
            kwargs["tool_context_id"] = tool_context_id

        try:
            response = await job(**kwargs)
        except ValueError as exc:
            return 400, {"error": str(exc)}
        return 200, {"success": response.success, "answer": response.answer}
