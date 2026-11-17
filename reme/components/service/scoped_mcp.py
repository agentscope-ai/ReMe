"""Session-scoped MCP tools backed by the current Application's jobs."""

import asyncio
from contextlib import contextmanager
import secrets
import socket
from typing import Any

from .mcp_tools import make_mcp_job_tool


class ScopedMCP:
    """Serve isolated tool catalogs on one authenticated, application-owned endpoint."""

    def __init__(self) -> None:
        self.path = f"/_reme/codex/{secrets.token_hex(12)}"
        self.scopes: dict[str, list[Any]] = {}
        self._app: Any = None
        self._server: Any = None
        self._task: asyncio.Task | None = None
        self._socket: socket.socket | None = None

    @property
    def app(self) -> Any:
        """Build one MCP transport; resolve tools separately for every HTTP request."""
        if self._app is None:
            from fastmcp import FastMCP
            from fastmcp.server.dependencies import get_http_headers
            from fastmcp.server.providers import Provider

            scopes = self.scopes

            class JobProvider(Provider):
                """Select the immutable tool binding carried by the request's bearer token."""

                async def _list_tools(self):
                    authorization = get_http_headers(include={"authorization"}).get("authorization", "")
                    return scopes.get(authorization.removeprefix("Bearer "), [])

            self._app = FastMCP("ReMe agent tools", providers=[JobProvider()], tasks=False).http_app(
                path=self.path,
                stateless_http=True,
            )
        return self._app

    def register(self, jobs: list[Any], injected: dict[str, Any]) -> str:
        """Bind selected jobs and hidden arguments without cloning the Application."""
        token = secrets.token_urlsafe(32)
        self.scopes[token] = [
            make_mcp_job_tool(job, injected_job_kwargs=injected, tool_error_on_failure=True) for job in jobs
        ]
        return token

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            authorization = dict(scope["headers"]).get(b"authorization", b"").decode("latin-1")
            if not authorization.startswith("Bearer ") or authorization[7:] not in self.scopes:
                from starlette.responses import Response

                await Response(status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)  # pylint: disable=not-callable

    async def listen(self) -> str:
        """Start a loopback listener for embedded/CLI hosts that have no HTTP service."""
        if self._server is None:
            import uvicorn

            class LocalServer(uvicorn.Server):
                """Leave signal handling to the containing ReMe application."""

                @contextmanager
                def capture_signals(self):
                    """Keep the caller's signal handlers intact."""
                    yield

            self._socket = socket.socket()
            self._socket.bind(("127.0.0.1", 0))
            self._server = LocalServer(uvicorn.Config(self, log_config=None, access_log=False, ws="none"))
            self._task = asyncio.create_task(self._server.serve(sockets=[self._socket]))
        while not self._server.started:
            if self._task.done():
                await self._task
                raise RuntimeError("Codex tool endpoint stopped before startup")
            await asyncio.sleep(0.01)
        return f"http://127.0.0.1:{self._socket.getsockname()[1]}{self.path}"

    async def close(self) -> None:
        """Stop the optional listener and revoke all tool bindings."""
        try:
            if self._server is not None:
                self._server.should_exit = True  # pylint: disable=attribute-defined-outside-init
                try:
                    await self._task
                finally:
                    self._socket.close()
                    self._server = self._task = self._socket = None
        finally:
            self.scopes.clear()
