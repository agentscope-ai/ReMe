"""Application-owned HTTP listener for subprocess clients."""

import asyncio
import socket
from contextlib import contextmanager

import uvicorn
from starlette.types import ASGIApp


class _LocalServer(uvicorn.Server):
    """A server embedded in an application that owns its signal handlers."""

    @contextmanager
    def capture_signals(self):
        """Leave signal handling to the containing application."""
        yield


class LoopbackHTTP:
    """Serve an ASGI application on an ephemeral IPv4 loopback port."""

    def __init__(self, app: ASGIApp, *, lifespan: str = "auto", shutdown_timeout: int | None = None) -> None:
        self._config = uvicorn.Config(
            app,
            log_config=None,
            access_log=False,
            ws="none",
            lifespan=lifespan,
            timeout_graceful_shutdown=shutdown_timeout,
        )
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task | None = None
        self._socket: socket.socket | None = None

    async def listen(self) -> str:
        """Start serving and return the bound origin."""
        if self._server is None:
            self._socket = socket.socket()
            self._socket.bind(("127.0.0.1", 0))
            self._server = _LocalServer(self._config)
            self._task = asyncio.create_task(self._server.serve(sockets=[self._socket]))
        while not self._server.started:
            if self._task.done():
                await self._task
                raise RuntimeError("Loopback HTTP endpoint stopped before startup")
            await asyncio.sleep(0.01)
        return f"http://127.0.0.1:{self._socket.getsockname()[1]}"

    async def close(self) -> None:
        """Stop serving and close the bound socket, including after failed startup."""
        try:
            if self._server is not None:
                self._server.should_exit = True  # pylint: disable=attribute-defined-outside-init
                await self._task
        finally:
            if self._socket is not None:
                self._socket.close()
            self._server = self._task = self._socket = None
