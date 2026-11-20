"""Session-scoped MCP tools backed by the current Application's jobs."""

import secrets
from typing import Any

from .loopback_http import LoopbackHTTP
from .mcp_tools import make_mcp_job_tool


class ScopedMCP(LoopbackHTTP):
    """Serve isolated tool catalogs on one authenticated, application-owned endpoint."""

    def __init__(self) -> None:
        super().__init__(self)
        self.path = f"/_reme/codex/{secrets.token_hex(12)}"
        self.scopes: dict[str, list[Any]] = {}
        self._app: Any = None

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
        return f"{await super().listen()}{self.path}"

    async def close(self) -> None:
        """Stop the optional listener and revoke all tool bindings."""
        try:
            await super().close()
        finally:
            self.scopes.clear()
