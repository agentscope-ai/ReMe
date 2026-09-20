"""Transport-independent backend contract for the Hermes provider."""

from __future__ import annotations

import contextvars
import threading

from typing import Any, Callable, Protocol


class ReMeBackendError(RuntimeError):
    """Raised when a ReMe backend cannot complete an operation."""


class ReMeBackend(Protocol):
    """Synchronous interface matching Hermes' memory-provider lifecycle."""

    label: str

    def start(self, *, deadline: float | None = None) -> None:
        """Start owned resources before an optional absolute deadline."""

    def health(self, *, timeout: float) -> dict[str, Any]:
        """Return a semantically healthy ReMe response."""

    def search(self, query: str, *, limit: int, timeout: float) -> dict[str, Any]:
        """Search the configured ReMe workspace."""

    def auto_memory(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        """Record and extract memory from one completed turn."""

    def close(self, *, timeout: float) -> None:
        """Release resources within a bounded interval."""


def spawn_profile_thread(
    target: Callable[..., Any],
    *,
    name: str,
    args: tuple[Any, ...] = (),
) -> threading.Thread:
    """Create a profile-aware daemon thread across supported Hermes versions."""
    try:
        from agent.memory_provider import spawn_context_thread
    except ImportError:
        context = contextvars.copy_context()

        def run() -> None:
            context.run(target, *args)

        return threading.Thread(target=run, name=name, daemon=True)
    return spawn_context_thread(target, name=name, args=args)


def require_healthy(response: dict[str, Any]) -> dict[str, Any]:
    """Validate the semantic health flag in a successful ReMe response."""
    metadata = response.get("metadata")
    health = metadata.get("health") if isinstance(metadata, dict) else None
    if not isinstance(health, dict) or health.get("healthy") is not True:
        raise ReMeBackendError("ReMe did not report a healthy component snapshot")
    return response
