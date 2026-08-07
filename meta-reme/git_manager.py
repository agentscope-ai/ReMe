"""Git lifecycle operations for Meta-ReMe code repositories."""

from __future__ import annotations

from pathlib import Path
import subprocess

INITIAL_BRANCH = "main"
INITIAL_COMMIT_MESSAGE = "Initial version"
GIT_AUTHOR_NAME = "Meta-ReMe"
GIT_AUTHOR_EMAIL = "meta-reme@localhost"


class GitManagerError(RuntimeError):
    """Raised when a managed Git operation fails."""


def _git(repository: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise GitManagerError(f"git {' '.join(arguments)} failed in {repository}: {detail}")
    return result


def initialize_repository(repository: Path) -> str:
    """Create the immutable initial commit, or return an existing repository's HEAD."""

    repository = Path(repository).resolve()
    if not repository.is_dir():
        raise GitManagerError(f"Code repository does not exist: {repository}")

    if (repository / ".git").exists():
        _git(repository, "rev-parse", "--is-inside-work-tree")
        head = _git(repository, "rev-parse", "--verify", "HEAD", check=False)
        if head.returncode == 0:
            return head.stdout.strip()
        _git(repository, "symbolic-ref", "HEAD", f"refs/heads/{INITIAL_BRANCH}")
    else:
        _git(repository, "init", "--initial-branch", INITIAL_BRANCH)

    _git(repository, "add", "--all")
    _git(
        repository,
        "-c",
        f"user.name={GIT_AUTHOR_NAME}",
        "-c",
        f"user.email={GIT_AUTHOR_EMAIL}",
        "commit",
        "--message",
        INITIAL_COMMIT_MESSAGE,
    )
    branch = _git(repository, "branch", "--show-current").stdout.strip()
    if branch != INITIAL_BRANCH:
        raise GitManagerError(f"Initial repository branch is {branch!r}, expected {INITIAL_BRANCH!r}")
    return _git(repository, "rev-parse", "HEAD").stdout.strip()
