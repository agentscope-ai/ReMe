"""Small synchronous client for ReMe's HTTP action service."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request

from typing import Any
from urllib.parse import urlsplit


class ReMeServiceError(RuntimeError):
    """Raised when a ReMe action cannot be completed successfully."""


class ReMeHttpClient:
    """Call ReMe JSON actions without adding a runtime dependency to Hermes."""

    def __init__(self, endpoint: str, *, timeout: float) -> None:
        endpoint = str(endpoint or "").strip().rstrip("/")
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("ReMe endpoint must be an absolute http(s) URL")
        self.endpoint = endpoint
        self.timeout = max(0.1, float(timeout))

    def call(
        self,
        action: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """POST one ReMe action and return its standard response envelope."""
        if not action or not action.replace("_", "").isalnum():
            raise ValueError(f"Invalid ReMe action: {action!r}")

        body = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.endpoint}/{action}",
            data=body,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        request_timeout = self.timeout if timeout is None else max(0.1, float(timeout))

        try:
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ReMeServiceError(f"HTTP {exc.code}: {detail or exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise ReMeServiceError(str(reason)) from exc

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ReMeServiceError("ReMe returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise ReMeServiceError("ReMe returned a non-object response")
        if result.get("success") is not True:
            raise ReMeServiceError(str(result.get("answer") or "ReMe action did not report success"))
        return result

    def health(self, *, timeout: float) -> dict[str, Any]:
        """Require both a successful response and a healthy component snapshot."""
        result = self.call("health_check", timeout=timeout)
        metadata = result.get("metadata")
        health = metadata.get("health") if isinstance(metadata, dict) else None
        if not isinstance(health, dict) or health.get("healthy") is not True:
            raise ReMeServiceError("ReMe did not report a healthy component snapshot")
        return result
