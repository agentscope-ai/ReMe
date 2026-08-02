"""Minimal AgentScope 2.0.4.post1 DockerWorkspace for direct ReMe jobs.

The stock workspace builds and starts an MCP gateway. The benchmark harness
uses only DockerBackend's exec/read/write primitives, so building that gateway
would add an unrelated network-dependent image layer and process.
"""

from __future__ import annotations

from typing import Any

from agentscope.workspace import DockerWorkspace


class DirectDockerWorkspace(DockerWorkspace):
    """Use the supplied image directly and skip the unused MCP gateway."""

    _image_tag: str
    _backend: Any
    is_alive: bool

    async def _build_or_reuse_image(self) -> None:
        """Select the caller-provided image instead of deriving a gateway image."""
        self._image_tag = self.base_image

    async def initialize(self) -> None:
        """Start one Docker container and expose its filesystem/exec backend."""
        if self.is_alive:
            return
        await self._provision_backend()
        backend = self.get_backend()
        result = await backend.exec_shell(["mkdir", "-p", self.workdir], cwd="/")
        if not result.ok():
            await self._teardown_backend()
            self._backend = None
            raise RuntimeError(
                f"failed to create Docker workspace {self.workdir!r}: "
                f"{result.stderr.decode('utf-8', errors='replace')}",
            )
        self.is_alive = True
