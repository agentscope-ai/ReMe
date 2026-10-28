"""Minimal AgentScope 2.0.10 DockerWorkspace for direct ReMe jobs.

The stock workspace builds and starts an MCP gateway. The benchmark harness
uses only DockerBackend's exec/read/write primitives, so building that gateway
would add an unrelated network-dependent image layer and process.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

from aiodocker.exceptions import DockerError
from agentscope.workspace import DockerWorkspace

DOCKER_CLEANUP_TIMEOUT = 30.0
_PENDING_CLEANUPS: set[asyncio.Task[None]] = set()


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
        # Retain the task until its creation result has been reconciled.
        self._creation_task: asyncio.Task[None] | None = None
        self._deferred_cleanup_task: asyncio.Task[None] | None = None

    async def _build_or_reuse_image(self) -> None:
        """Select the caller-provided image instead of deriving a gateway image."""
        self._image_tag = self.base_image

    async def _create_and_start_container(self) -> None:
        """Keep creation owned until its result can be reconciled on cleanup."""
        self._creation_task = asyncio.create_task(super()._create_and_start_container())
        # Let the request settle even when initialize is cancelled. Otherwise a
        # name lookup could return 404 before Docker finishes creating it.
        await asyncio.shield(self._creation_task)
        self._creation_task = None

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
        if self._creation_task is not None:
            done, _ = await asyncio.wait({self._creation_task}, timeout=DOCKER_CLEANUP_TIMEOUT)
            if not done:
                if self._deferred_cleanup_task is None or self._deferred_cleanup_task.done():
                    self._deferred_cleanup_task = asyncio.create_task(
                        self._finish_creation_cleanup(self._creation_task),
                    )
                    # Keep cleanup owned even if factory creation returns no workspace.
                    _PENDING_CLEANUPS.add(self._deferred_cleanup_task)
                    self._deferred_cleanup_task.add_done_callback(_PENDING_CLEANUPS.discard)
                raise TimeoutError("Docker container creation has not settled; deferred cleanup is pending")
            # initialize already reports the provisioning failure. A cancelled
            # initialize may instead leave a failed creation to reconcile here.
            if not self._creation_task.cancelled():
                self._creation_task.exception()
            if self._container is None:
                try:
                    self._container = await asyncio.wait_for(
                        self._client.containers.get(f"as_ws_{self.workspace_id}"),
                        timeout=DOCKER_CLEANUP_TIMEOUT,
                    )
                except DockerError as exc:
                    if exc.status != 404:
                        raise
            self._creation_task = None
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

    async def _finish_creation_cleanup(self, creation: asyncio.Task[None]) -> None:
        """Remove a late creation result after bounded foreground cleanup expires."""
        await asyncio.wait({creation})
        async with self._lifecycle_lock:
            # A caller may have already closed and even reinitialized this workspace.
            if self._creation_task is not creation:
                return
            try:
                await self._teardown_backend()
            except Exception:
                logging.getLogger(__name__).exception("Deferred Docker cleanup failed for %s", self.workspace_id)

    async def close(self) -> None:
        """Report cleanup errors and allow another call to retry unfinished work."""
        # The inherited close clears backend state in a finally block. This
        # direct workspace has no gateway and must retain state on failure.
        async with self._lifecycle_lock:
            await self._teardown_backend()
