"""Strict input contracts for candidate version and code-navigation Jobs."""

from pydantic import Field

from ..preparation.spec import SpecModel


class NoArguments(SpecModel):
    """An operation with no public inputs."""


class BranchArguments(SpecModel):
    """One existing or new local candidate branch."""

    branch_name: str = Field(
        min_length=1,
        max_length=100,
        description=(
            "Valid local Git branch name used as a single evaluation-directory name; '/' and '\\' are forbidden."
        ),
    )


class CommitArguments(SpecModel):
    """Commit the complete candidate change with an explicit message."""

    message: str = Field(min_length=1, max_length=10_000, pattern=r"\S")


class DiffArguments(SpecModel):
    """Omit revisions for tracked working changes against HEAD."""

    base_revision: str | None = Field(
        default=None,
        description="Base commit; defaults to HEAD. Without revision, compare against the tracked working tree.",
    )
    revision: str | None = Field(
        default=None,
        description="Target commit; requires non-null base_revision. Null or omitted means the tracked working tree.",
    )


class TreeArguments(SpecModel):
    """Resolve HEAD by default; hosts can pin revision with injected_job_kwargs."""

    revision: str = Field(default="HEAD", min_length=1)
    path: str | None = None
    max_results: int = Field(default=500, ge=1, le=2000)


class ReadArguments(SpecModel):
    """A bounded line range from a committed regular text file."""

    path: str = Field(min_length=1)
    revision: str = Field(default="HEAD", min_length=1)
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)


class SearchArguments(SpecModel):
    """Python regex search with multiline matching and bounded result blocks."""

    pattern: str = Field(min_length=1, max_length=500)
    revision: str = Field(default="HEAD", min_length=1)
    path: str | None = None
    max_results: int = Field(default=30, ge=1, le=30)


ARGUMENTS = {
    "status": NoArguments,
    "list_branches": NoArguments,
    "create_branch": BranchArguments,
    "switch_branch": BranchArguments,
    "merge_branch": BranchArguments,
    "commit": CommitArguments,
    "diff": DiffArguments,
    "list_code_tree": TreeArguments,
    "read_code_file": ReadArguments,
    "search_code": SearchArguments,
}
