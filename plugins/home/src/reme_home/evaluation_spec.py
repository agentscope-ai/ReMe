"""Inputs for a single immutable Home evaluation."""

from typing import Literal
from pydantic import Field, field_validator, model_validator

from .preparation_spec import SpecModel


class EvaluationSpec(SpecModel):
    """Selections are injected by callers; omitted cases mean the whole dataset."""

    case_ids: list[str] | None = None
    query_ids: dict[str, list[str]] | None = None
    reuse_construction_from: str | None = Field(default=None, min_length=1)
    dataset_kind: Literal["training-set", "debug-cases"] = "training-set"
    concurrency: int = Field(default=4, ge=1)
    timeout_seconds: float = Field(default=7200.0, gt=0, allow_inf_nan=False)
    base_image: str = Field(default="reme-benchmark-base:agentscope-2.0.9", min_length=1)
    fail_fast: bool = False
    max_construction_failure_rate: float = Field(default=0.03, ge=0, le=1, allow_inf_nan=False)
    max_artifact_bytes: int = Field(default=2_000_000_000, ge=1)

    @field_validator("case_ids")
    @classmethod
    def validate_cases(cls, value):
        """An explicit selection must never silently expand or shrink."""
        if value is not None and (
            not value or any(not item.strip() for item in value) or len(set(value)) != len(value)
        ):
            raise ValueError("case_ids must be nonempty, unique strings, or null for all cases")
        return value

    @field_validator("query_ids")
    @classmethod
    def validate_queries(cls, value):
        """Validate each explicit per-case query selection."""
        if value is not None:
            if not value:
                raise ValueError("query_ids must not be empty")
            for case_id, ids in value.items():
                if not case_id.strip() or not ids or any(not item.strip() for item in ids) or len(set(ids)) != len(ids):
                    raise ValueError("query_ids must contain nonempty unique selections")
        return value

    @model_validator(mode="after")
    def validate_reuse(self):
        """Smoke datasets cannot provide or consume scored construction evidence."""
        if self.dataset_kind == "debug-cases" and self.reuse_construction_from:
            raise ValueError("Construction reuse requires training-set")
        return self
