"""Minimal AgentScope 2.0.9 DockerWorkspace for direct ReMe jobs.

The stock workspace builds and starts an MCP gateway. The benchmark harness
uses only DockerBackend's exec/read/write primitives, so building that gateway
would add an unrelated network-dependent image layer and process.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from aiodocker.exceptions import DockerError
from agentscope.workspace import DockerWorkspace

DOCKER_CLEANUP_TIMEOUT = 30.0


class DirectDockerWorkspace(DockerWorkspace):
    """Use the supplied image directly and skip the unused MCP gateway."""

    _image_tag: str
    _backend: Any
    _container: Any
    _client: Any
    is_alive: bool

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lifecycle_lock = asyncio.Lock()
        self._stopped = False

    async def _build_or_reuse_image(self) -> None:
        """Select the caller-provided image instead of deriving a gateway image."""
        self._image_tag = self.base_image

    async def initialize(self) -> None:
        """Start one Docker container and expose its filesystem/exec backend."""
        async with self._lifecycle_lock:
            if self.is_alive:
                return
            if self._container is not None or self._client is not None:
                raise RuntimeError("Docker workspace cleanup must complete before initialization")
            try:
                await self._provision_backend()
                backend = self.get_backend()
                result = await backend.exec_shell(["mkdir", "-p", self.workdir], cwd="/")
                if not result.ok():
                    raise RuntimeError(
                        f"failed to create Docker workspace {self.workdir!r}: "
                        f"{result.stderr.decode('utf-8', errors='replace')}",
                    )
            except BaseException as exc:
                try:
                    await self._teardown_backend()
                except Exception as close_exc:
                    exc.add_note(f"workspace close failed: {type(close_exc).__name__}: {close_exc}")
                raise
            self.is_alive = True
            self._stopped = False

    async def stop(self) -> None:
        """Kill all container processes without asking the candidate to shut down."""
        async with self._lifecycle_lock:
            self.is_alive = False
            if self._container is None or self._stopped:
                return
            async with asyncio.timeout(DOCKER_CLEANUP_TIMEOUT):
                try:
                    await self._container.kill(signal="SIGKILL")
                    await self._container.wait(condition="not-running")
                except DockerError as exc:
                    if exc.status != 404:
                        # Docker returns 409 for an already stopped container.
                        if exc.status != 409 or (await self._container.show())["State"]["Running"]:
                            raise
                self._stopped = True

    async def _teardown_backend(self) -> None:
        """Remove the container, preserving handles until cleanup is confirmed."""
        self.is_alive = False
        if self._container is not None:
            # Preserve AgentScope's bind-mount ownership repair on native Linux.
            if (
                not self._stopped
                and self.host_workdir is not None
                and sys.platform == "linux"
                and self._backend is not None
            ):
                try:
                    await self._backend.exec_shell(
                        ["chown", "-R", f"{os.getuid()}:{os.getgid()}", self.workdir],
                        timeout=10.0,
                    )
                except Exception:
                    pass
            try:
                # Also stop processes if an earlier explicit stop failed, and
                # retain handles unless Docker confirms removal.
                await asyncio.wait_for(self._container.delete(force=True), timeout=DOCKER_CLEANUP_TIMEOUT)
            except DockerError as exc:
                if exc.status != 404:
                    raise
            self._container = None
            self._backend = None
        if self._client is not None:
            await asyncio.wait_for(self._client.close(), timeout=DOCKER_CLEANUP_TIMEOUT)
            self._client = None
        self._backend = None

    async def close(self) -> None:
        """Report cleanup errors and allow another call to retry unfinished work."""
        # The inherited close clears backend state in a finally block. This
        # direct workspace has no gateway and must retain state on failure.
        async with self._lifecycle_lock:
            await self._teardown_backend()
