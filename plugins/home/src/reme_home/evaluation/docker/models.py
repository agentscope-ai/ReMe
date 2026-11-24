"""Small serializable contracts shared by the host and sandbox worker."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

_CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def validate_case_id(case_id: str) -> None:
    """Keep case IDs usable in Docker names and artifact directories."""
    if not _CASE_ID_RE.fullmatch(case_id):
        raise ValueError(
            "case_id must start with a letter or number, use only letters, numbers, '.', '_' or '-', "
            "and be at most 128 characters",
        )


def validate_query_artifact_id(query_id: str) -> None:
    """Keep verbatim query-directory names inside the artifact root."""
    if not query_id or len(query_id.encode("utf-8")) > 255:
        raise ValueError("query_id must be non-empty and no longer than 255 UTF-8 bytes")
    if query_id in {".", ".."} or "/" in query_id or "\\" in query_id or "\x00" in query_id:
        raise ValueError(f"query_id is not a safe directory name: {query_id!r}")
    if query_id == "summary.json":
        raise ValueError("query_id conflicts with the queries summary: 'summary.json'")


@dataclass(frozen=True, slots=True)
class JobResult:
    """Structured result returned by a direct ReMe job invocation.

    ``token_usage`` contains the input/output/total token delta for each
    configured agent wrapper during this invocation. Missing provider usage is
    represented by ``None`` at the metric level. ``job_call_counts`` contains
    non-zero ReMe job invocation deltas, including nested job tools.
    """

    success: bool
    answer: Any = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    token_usage: dict[str, dict[str, int | None]] = field(default_factory=dict)
    job_call_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "JobResult":
        """Build a result from the worker's JSON response."""
        return cls(
            success=bool(value.get("success")),
            answer=value.get("answer", ""),
            metadata=value.get("metadata") if isinstance(value.get("metadata"), dict) else {},
            token_usage=value.get("token_usage") if isinstance(value.get("token_usage"), dict) else {},
            job_call_counts=value.get("job_call_counts") if isinstance(value.get("job_call_counts"), dict) else {},
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


@dataclass(frozen=True, slots=True)
class JobRequest:
    """One job executed as part of a shared build Application lifecycle."""

    job: str
    arguments: dict[str, Any] = field(default_factory=dict)
    memory_checkpoint: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    """One answer request plus host-only judging metadata.

    Only ``query_id``, ``question``, ``answer_arguments`` and ``answer_job``
    cross the candidate sandbox boundary.  The remaining fields are retained
    on the trusted host for post-answer evaluation.
    """

    query_id: str
    question: str
    golden_answer: Any
    answer_arguments: dict[str, Any] = field(default_factory=dict)
    answer_job: str = "agentic_answer"
