"""HTTP implementation of the ReMe backend contract."""

from __future__ import annotations

from typing import Any

from .backend import ReMeBackendError, require_healthy
from .client import ReMeHttpClient, ReMeServiceError


class HttpReMeBackend:
    """Call an independently managed ReMe action service."""

    def __init__(self, endpoint: str, *, request_timeout: float) -> None:
        self._client = ReMeHttpClient(endpoint, timeout=request_timeout)
        self.label = self._client.endpoint

    def start(self, *, deadline: float | None = None) -> None:
        """HTTP service lifecycle is managed outside Hermes."""
        del deadline

    def _call(
        self,
        action: str,
        payload: dict[str, Any] | None,
        *,
        timeout: float,
    ) -> dict[str, Any]:
        try:
            return self._client.call(action, payload, timeout=timeout)
        except (ReMeServiceError, ValueError) as exc:
            raise ReMeBackendError(str(exc)) from exc

    def health(self, *, timeout: float) -> dict[str, Any]:
        """Call the action service health check."""
        return require_healthy(self._call("health_check", None, timeout=timeout))

    def search(self, query: str, *, limit: int, timeout: float) -> dict[str, Any]:
        """Search through the HTTP action service."""
        return self._call("search", {"query": query, "limit": limit}, timeout=timeout)

    def auto_memory(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        """Submit a completed turn through the HTTP action service."""
        return self._call(
            "auto_memory",
            {"session_id": session_id, "messages": messages},
            timeout=timeout,
        )

    def close(self, *, timeout: float) -> None:
        """Release no resources because urllib calls are request-scoped."""
        del timeout
