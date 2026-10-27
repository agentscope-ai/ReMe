"""Validated inputs for a single Home experiment."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SpecModel(BaseModel):
    """Do not silently accept misspelled preparation options."""

    model_config = ConfigDict(extra="forbid", strict=True)


def relative_path(value: str) -> str:
    """Require a portable, repository-relative path without traversal."""
    if not value or value.startswith("~") or "\\" in value or ":" in value:
        raise ValueError(f"Invalid relative path: {value!r}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(f"Invalid relative path: {value!r}")
    return path.as_posix()


class BenchmarkSpec(SpecModel):
    """Benchmark adapter, explicit case selection, and source inputs."""

    name: Literal["beam", "longmemeval"]
    source: str = Field(min_length=1)
    variant: str | None = None
    case_ids: list[str] | None = None
    all_cases: bool = False
    evidence_source: str | None = None

    @model_validator(mode="after")
    def validate_selection(self):
        """Require either a nonempty unique selection or explicit all-cases."""
        if self.all_cases == (self.case_ids is not None):
            raise ValueError("Provide case_ids or all_cases=true, exclusively")
        if self.case_ids is not None:
            if not self.case_ids or any(not case_id.strip() for case_id in self.case_ids):
                raise ValueError("case_ids must contain nonempty strings")
            if len(set(self.case_ids)) != len(self.case_ids):
                raise ValueError("case_ids must not contain duplicates")
        if self.name == "beam":
            if self.variant not in {"100K", "500K", "1M", "10M"}:
                raise ValueError("BEAM requires variant: 100K, 500K, 1M, or 10M")
            if self.evidence_source is not None:
                raise ValueError("evidence_source is supported only for LongMemEval")
        elif self.variant is not None:
            raise ValueError("LongMemEval does not accept variant")
        return self


class RuntimeSpec(SpecModel):
    """An immutable source revision; plugins belong to the evaluated ReMe."""

    source_repo: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    config: str = "benchmark.yaml"
    plugins: list[str] = Field(default_factory=list)

    @field_validator("config")
    @classmethod
    def validate_config(cls, value):
        """Select a single bundled application configuration."""
        value = relative_path(value)
        if Path(value).name != value or Path(value).suffix not in {".yaml", ".yml", ".json"}:
            raise ValueError("runtime.config must name a YAML or JSON file in reme/config")
        return value

    @field_validator("plugins")
    @classmethod
    def validate_plugins(cls, values):
        """Select local repository plugin projects by name."""
        if len(set(values)) != len(values):
            raise ValueError("runtime.plugins must be unique")
        for value in values:
            if relative_path(value) != value or "/" in value:
                raise ValueError("runtime.plugins must name plugin directories")
        return values


class CandidateSpec(SpecModel):
    """Repository paths copied into the independently versioned candidate."""

    paths: list[str] = Field(min_length=1)

    @field_validator("paths")
    @classmethod
    def validate_paths(cls, values):
        """Reject traversal and overlapping candidate roots."""
        paths = [relative_path(value) for value in values]
        for index, value in enumerate(paths):
            for other in paths[:index]:
                if Path(value).is_relative_to(other) or Path(other).is_relative_to(value):
                    raise ValueError("candidate.paths must not overlap or contain duplicates")
        return paths


class DebugSpec(SpecModel):
    """Deterministic smoke-data truncation, independent of scoring."""

    enabled: bool = True
    sessions_per_case: int = Field(default=3, ge=1)
    messages_per_session: int = Field(default=6, ge=1)


class PreparationSpec(SpecModel):
    """Public Home preparation configuration, accepted as YAML or a dict."""

    schema_version: Literal[1] = 1
    benchmark: BenchmarkSpec
    runtime: RuntimeSpec
    candidate: CandidateSpec
    debug: DebugSpec = Field(default_factory=DebugSpec)
