"""Small serializable contracts shared by the host and sandbox worker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class JobResult:
    """Structured result returned by a direct ReMe job invocation.

    ``token_usage`` contains the input/output/total token delta for each
    configured agent wrapper during this invocation. Missing provider usage is
    represented by ``None`` at the metric level.
    """

    success: bool
    answer: Any = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    token_usage: dict[str, dict[str, int | None]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "JobResult":
        """Build a result from the worker's JSON response."""
        return cls(
            success=bool(value.get("success")),
            answer=value.get("answer", ""),
            metadata=value.get("metadata") if isinstance(value.get("metadata"), dict) else {},
            token_usage=value.get("token_usage") if isinstance(value.get("token_usage"), dict) else {},
            error=value.get("error") if isinstance(value.get("error"), str) else None,
        )


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """Audit record for one command executed inside a case sandbox."""

    name: str
    exit_code: int
    stdout: str
    stderr: str
    started_at: str
    finished_at: str
