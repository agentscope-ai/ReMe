"""Public data contracts shared by the Meta-ReMe orchestration modules.

The models in this module describe persisted state and values crossing module
boundaries.  File I/O, event reduction, scoring, and orchestration belong in
their respective modules rather than here.
"""

# Persisted schema classes are intentionally named and self-describing.
# pylint: disable=missing-class-docstring,missing-function-docstring

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""

    return datetime.now(timezone.utc)


class ContractModel(BaseModel):
    """Base class for strict, JSON-serializable persisted contracts."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a supported value deterministically for hashing."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=False)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=_json_default,
    ).encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def fingerprint(value: Any) -> str:
    """Return the SHA256 of a canonical JSON representation."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


class DatasetName(StrEnum):
    LONGMEMEVAL = "longmemeval"
    BEAM = "beam"


class RunStatus(StrEnum):
    INITIALIZING = "initializing"
    SEARCHING = "searching"
    REPLAYING = "replaying"
    TESTING = "testing"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationMode(StrEnum):
    DEBUG = "debug"
    SCREENING = "screening"
    SEARCH = "search"
    REPLAY = "replay"
    TEST = "test"


class AttemptStatus(StrEnum):
    COMPLETED = "completed"
    CANDIDATE_FAILURE = "candidate_failure"
    INFRA_ERROR = "infra_error"
    INTERRUPTED = "interrupted"


class ValidationStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    INFRA_ERROR = "infra_error"


class HarnessStatus(StrEnum):
    DEBUG = "debug"
    FROZEN = "frozen"
    REJECTED = "rejected"


class EventType(StrEnum):
    RUN_INITIALIZED = "run_initialized"
    RESUME_STARTED = "resume_started"
    RESUME_COMPLETED = "resume_completed"
    ROUND_STARTED = "round_started"
    WEAKNESS_COMPLETED = "weakness_completed"
    PROPOSAL_COMPLETED = "proposal_completed"
    HARNESS_COMMITTED = "harness_committed"
    HARNESS_FROZEN = "harness_frozen"
    HARNESS_REJECTED = "harness_rejected"
    VALIDATION_STARTED = "validation_started"
    CASE_ATTEMPT_COMPLETED = "case_attempt_completed"
    VALIDATION_COMPLETED = "validation_completed"
    LEADERBOARD_UPDATED = "leaderboard_updated"
    ROUND_COMPLETED = "round_completed"
    WINNER_SELECTED = "winner_selected"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"


class DatasetSpec(ContractModel):
    name: DatasetName
    source: str
    fingerprint: str = Field(min_length=1)


class ScopeSpec(ContractModel):
    harness_paths: list[str] = Field(default_factory=list)
    debug_paths: list[str] = Field(default_factory=lambda: ["meta_debug"])
    config_paths: dict[str, list[str]] = Field(default_factory=dict)


class SandboxSpec(ContractModel):
    image: str = Field(min_length=1)
    timeout_seconds: float = Field(gt=0)
    concurrency: int = Field(default=1, ge=1)
    max_retries: int = Field(
        default=0,
        ge=0,
        description="Deprecated compatibility field; validation operations are not retried.",
    )
    max_artifact_bytes: int = Field(default=1_073_741_824, gt=0)


class ProposerSpec(ContractModel):
    model: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1)


class ValidationPolicySpec(ContractModel):
    screening_enabled: bool = True
    default_case_ids: list[str] | None = None
    default_query_ids: dict[str, list[str]] = Field(default_factory=dict)
    promotion_min_mean_query_score: float | None = None
    require_full_for_leaderboard: bool = True

    @field_validator("require_full_for_leaderboard")
    @classmethod
    def require_full_results(cls, value: bool) -> bool:
        if not value:
            raise ValueError("partial validations cannot be admitted to the leaderboard")
        return value

    @model_validator(mode="after")
    def validate_default_selection(self) -> "ValidationPolicySpec":
        ValidationSelection(
            case_ids=self.default_case_ids,
            query_ids=self.default_query_ids,
            reason="domain default screening selection",
        )
        return self


class BudgetSpec(ContractModel):
    max_proposals: int | None = Field(default=None, ge=1)
    max_validations: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_runtime_seconds: float | None = Field(default=None, gt=0)
    deadline: datetime | None = None

    @model_validator(mode="after")
    def require_a_limit(self) -> "BudgetSpec":
        if all(
            value is None
            for value in (
                self.max_proposals,
                self.max_validations,
                self.max_tokens,
                self.max_runtime_seconds,
                self.deadline,
            )
        ):
            raise ValueError("at least one search budget limit must be configured")
        return self


class DomainSpec(ContractModel):
    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    dataset: DatasetSpec
    bundle_target: str = Field(pattern=r"^(default|lme|beam)$")
    benchmark_runner: str = Field(min_length=1)
    scorer: str = Field(min_length=1)
    objective: str = Field(default="mean_query_score", pattern=r"^mean_query_score$")
    baseline_files: list[str] = Field(default_factory=list)
    scope: ScopeSpec
    sandbox: SandboxSpec
    proposer: ProposerSpec
    validation: ValidationPolicySpec = Field(default_factory=ValidationPolicySpec)
    budget: BudgetSpec


class QuerySpec(ContractModel):
    query_id: str = Field(min_length=1)
    question: str
    golden_answer: Any
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionSpec(ContractModel):
    session_id: str = Field(min_length=1)
    messages: list[dict[str, Any]]
    metadata: dict[str, Any] = Field(default_factory=dict)


class CaseSpec(ContractModel):
    case_id: str = Field(min_length=1)
    sessions: list[SessionSpec]
    queries: list[QuerySpec]
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetManifest(ContractModel):
    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    dataset: DatasetName
    source_fingerprint: str = Field(min_length=1)
    normalized_fingerprint: str = Field(min_length=1)
    case_count: int = Field(ge=0)
    query_count: int = Field(ge=0)
    created_at: datetime = Field(default_factory=utc_now)


class Fingerprints(ContractModel):
    dataset: str = Field(min_length=1)
    code: str = Field(min_length=1)
    config: str = Field(min_length=1)
    model: str = Field(min_length=1)
    image: str = Field(min_length=1)


class ValidationSelection(ContractModel):
    """A fixed case/query subset; no selection means the complete dataset."""

    case_ids: list[str] | None = None
    query_ids: dict[str, list[str]] = Field(default_factory=dict)
    reason: str = Field(default="full dataset", min_length=1)

    @model_validator(mode="after")
    def validate_selection(self) -> "ValidationSelection":
        if self.case_ids is not None:
            if not self.case_ids:
                raise ValueError("case_ids cannot be an empty selection")
            if len(self.case_ids) != len(set(self.case_ids)):
                raise ValueError("case_ids must be unique")
            unknown_cases = set(self.query_ids) - set(self.case_ids)
            if unknown_cases:
                raise ValueError(f"query selections reference unselected cases: {sorted(unknown_cases)}")
        for case_id, query_ids in self.query_ids.items():
            if not case_id or not query_ids:
                raise ValueError("query_ids must contain non-empty case and query selections")
            if len(query_ids) != len(set(query_ids)):
                raise ValueError(f"query IDs must be unique for case {case_id!r}")
        return self

    @property
    def requests_full_dataset(self) -> bool:
        return self.case_ids is None and not self.query_ids


class ValidationSpec(ContractModel):
    validation_id: str = Field(default_factory=lambda: uuid4().hex)
    commit_sha: str = Field(min_length=1)
    mode: EvaluationMode
    selection: ValidationSelection = Field(default_factory=ValidationSelection)
    selection_fingerprint: str = ""
    fingerprints: Fingerprints

    @model_validator(mode="after")
    def validate_selection_fingerprint(self) -> "ValidationSpec":
        expected = model_fingerprint(self.selection)
        if not self.selection_fingerprint:
            self.selection_fingerprint = expected
        elif self.selection_fingerprint != expected:
            raise ValueError("selection_fingerprint does not match validation selection")
        return self


class ValidationCoverage(ContractModel):
    selected_cases: int = Field(ge=0)
    total_cases: int = Field(ge=0)
    selected_queries: int = Field(ge=0)
    total_queries: int = Field(ge=0)
    is_full: bool

    @model_validator(mode="after")
    def validate_coverage(self) -> "ValidationCoverage":
        if self.selected_cases > self.total_cases or self.selected_queries > self.total_queries:
            raise ValueError("selected validation coverage cannot exceed dataset totals")
        expected_full = self.selected_cases == self.total_cases and self.selected_queries == self.total_queries
        if self.is_full != expected_full:
            raise ValueError("is_full does not match selected and total coverage")
        return self


class BudgetUsage(ContractModel):
    proposals: int = Field(default=0, ge=0)
    validations: int = Field(default=0, ge=0)
    tokens: int | None = Field(default=0, ge=0)
    runtime_seconds: float = Field(default=0, ge=0)
    has_unknown_usage: bool = False


class RunState(ContractModel):
    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    status: RunStatus = RunStatus.INITIALIZING
    last_event_seq: int = Field(default=0, ge=0)
    current_round: int = Field(default=0, ge=0)
    baseline_commit: str | None = None
    best_commit: str | None = None
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SearchEvent(ContractModel):
    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    seq: int = Field(ge=1)
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    event_type: EventType
    timestamp: datetime = Field(default_factory=utc_now)
    round_id: int | None = Field(default=None, ge=0)
    proposal_id: str | None = None
    commit_sha: str | None = None
    validation_id: str | None = None
    case_id: str | None = None
    attempt_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class WeaknessEvidence(ContractModel):
    query_id: str
    observation: str


class WeaknessReport(ContractModel):
    weakness_id: str = Field(default_factory=lambda: uuid4().hex)
    round_id: int = Field(ge=0)
    patterns: list[str]
    evidence: list[WeaknessEvidence]
    possible_causes: list[str]
    regression_risks: list[str] = Field(default_factory=list)
    input_fingerprint: str


class Proposal(ContractModel):
    proposal_id: str = Field(default_factory=lambda: uuid4().hex)
    round_id: int = Field(ge=0)
    parent_commits: list[str] = Field(min_length=1)
    hypothesis: str
    expected_improvements: list[str]
    merge_rationale: str | None = None
    input_fingerprint: str


class ScopeCheckResult(ContractModel):
    passed: bool
    changed_paths: list[str]
    violations: list[str] = Field(default_factory=list)


class HarnessManifest(ContractModel):
    commit_sha: str = Field(min_length=1)
    parent_commits: list[str]
    snapshot_sha256: str = Field(min_length=1)
    config_fingerprint: str = Field(min_length=1)
    proposal_id: str | None = None
    status: HarnessStatus
    scope: ScopeCheckResult
    created_at: datetime = Field(default_factory=utc_now)


class QueryResult(ContractModel):
    query_id: str
    question: str
    golden_answer: Any
    answer: str | None = None
    score: float
    judge: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    tokens: int | None = Field(default=None, ge=0)
    duration_seconds: float | None = Field(default=None, ge=0)


class CaseResult(ContractModel):
    case_id: str
    attempt_id: str
    status: AttemptStatus
    selected_query_ids: list[str] = Field(default_factory=list)
    queries: list[QueryResult] = Field(default_factory=list)
    completed_stages: list[str] = Field(default_factory=list)
    error: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_terminal_result(self) -> "CaseResult":
        if self.status is AttemptStatus.COMPLETED and not self.queries:
            raise ValueError("completed case result must contain at least one query")
        if len(self.selected_query_ids) != len(set(self.selected_query_ids)):
            raise ValueError("selected_query_ids must be unique")
        result_query_ids = {query.query_id for query in self.queries}
        if self.queries and result_query_ids != set(self.selected_query_ids):
            raise ValueError("case query results must match selected_query_ids")
        return self


class AttemptCompletion(ContractModel):
    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    case_id: str
    attempt_id: str
    status: AttemptStatus
    fingerprints: Fingerprints
    selection_fingerprint: str = Field(min_length=1)
    result_file: str = "case_result.json"
    result_sha256: str = Field(min_length=1)
    artifact_sha256: dict[str, str] = Field(default_factory=dict)
    completed_at: datetime = Field(default_factory=utc_now)

    @field_validator("status")
    @classmethod
    def require_reusable_status(cls, value: AttemptStatus) -> AttemptStatus:
        if value not in (AttemptStatus.COMPLETED, AttemptStatus.CANDIDATE_FAILURE):
            raise ValueError("complete marker requires a reusable terminal status")
        return value


class ValidationResult(ContractModel):
    validation_id: str
    commit_sha: str
    mode: EvaluationMode
    status: ValidationStatus
    fingerprints: Fingerprints
    selection: ValidationSelection
    coverage: ValidationCoverage
    comparable: bool = False
    mean_query_score: float | None = None
    query_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    cases: list[CaseResult] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_score(self) -> "ValidationResult":
        if self.status is ValidationStatus.COMPLETED and self.mean_query_score is None:
            raise ValueError("completed validation must contain mean_query_score")
        if self.status is ValidationStatus.INFRA_ERROR and self.mean_query_score is not None:
            raise ValueError("infra_error validation cannot have a comparable score")
        expected_comparable = (
            self.status is ValidationStatus.COMPLETED
            and self.mode in (EvaluationMode.SEARCH, EvaluationMode.REPLAY)
            and self.coverage.is_full
        )
        if self.comparable != expected_comparable:
            raise ValueError("comparable requires a completed full search or replay validation")
        return self


class LeaderboardEntry(ContractModel):
    commit_sha: str
    validation_id: str
    mean_query_score: float
    completed_at: datetime
    proposal_id: str | None = None


class WorkspaceManifest(ContractModel):
    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    workspace_id: str = Field(default_factory=lambda: uuid4().hex)
    domain_fingerprint: str
    created_at: datetime = Field(default_factory=utc_now)


class WorkspaceLockOwner(ContractModel):
    token: str = Field(default_factory=lambda: uuid4().hex)
    pid: int = Field(gt=0)
    hostname: str
    started_at: datetime = Field(default_factory=utc_now)


def model_fingerprint(model: BaseModel | Mapping[str, Any]) -> str:
    """Convenience alias documenting fingerprints derived from contracts."""

    return fingerprint(model)
