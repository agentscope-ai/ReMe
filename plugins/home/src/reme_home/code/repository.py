"""Local candidate Git operations shared by Home Jobs and evaluation snapshots."""

from pathlib import Path

from ..evaluation.artifacts import contained, read_json
from ..preparation.code import git
from ..preparation.spec import relative_path

INITIAL_BRANCH = "init"
MAX_OUTPUT = 100_000


def resolve_revision(repository: Path, revision: str) -> str:
    """Resolve a revision once, without accepting command-line options."""
    if not isinstance(revision, str) or not revision.strip() or revision.startswith("-"):
        raise ValueError("revision must be a nonempty Git revision without a leading '-'")
    return git(repository, "rev-parse", "--verify", "--end-of-options", revision + "^{commit}")


def validate_repository(home: Path, record: dict | None = None) -> tuple[Path, dict]:
    """Require the prepared standalone repository and its unchanged baseline ref."""
    record = record if record is not None else read_json(contained(home, "metadata/preparation.json"))
    repository = contained(home, "code/repo", directory=True)
    metadata = contained(repository, ".git", directory=True)
    if not metadata.is_dir() or git(repository, "rev-parse", "--show-toplevel") != str(repository):
        raise ValueError("Candidate must be a standalone Git repository")
    if resolve_revision(repository, "refs/heads/init") != record["code"]["initial_commit"]:
        raise ValueError("The protected init branch has changed since preparation")
    git(repository, "merge-base", "--is-ancestor", record["code"]["initial_commit"], "HEAD")
    return repository, record


def current_branch(repository: Path) -> str:
    """Reject detached HEAD so every candidate commit remains branch-reachable."""
    branch = git(repository, "branch", "--show-current")
    validate_branch(repository, branch)
    return branch


def validate_branch(repository: Path, branch: str) -> str:
    """Branch names also serve as single evaluation-directory names."""
    if not branch or branch.startswith("-") or any(char in branch for char in "/\\\0\n\r"):
        raise ValueError("branch_name must be one path-safe local branch name")
    if git(repository, "check-ref-format", "--branch", branch) != branch:
        raise ValueError("Use a literal local branch name")
    return branch


def require_clean(repository: Path) -> None:
    """Do not carry uncommitted or untracked files across branch operations."""
    if git(repository, "status", "--porcelain=v2", "--untracked-files=all", "--ignore-submodules=none"):
        raise ValueError("Candidate repository must be clean before changing branches")
    if (repository / ".git/MERGE_HEAD").exists():
        raise ValueError("Finish the pending merge first")


def branches(repository: Path) -> dict:
    """Describe local refs and the current immutable commit."""
    local = sorted(git(repository, "for-each-ref", "--format=%(refname:short)", "refs/heads").splitlines())
    candidates = [branch for branch in local if branch != INITIAL_BRANCH]
    return {
        "current_branch": current_branch(repository),
        "commit_sha": resolve_revision(repository, "HEAD"),
        "local_branches": local,
        "candidate_branches": candidates,
        "candidate_branch_count": len(candidates),
    }


def status(repository: Path, _record: dict) -> dict:
    """Report exact paths, including names containing whitespace."""
    result = branches(repository)
    output = git(repository, "status", "--porcelain=v1", "-z", "--untracked-files=all", strip=False)
    entries = iter(output.split("\0") if output else [])
    changes = []
    for entry in entries:
        if not entry:
            continue
        change = {"status": entry[:2], "path": entry[3:]}
        if "R" in entry[:2] or "C" in entry[:2]:
            change["original_path"] = next(entries)
        changes.append(change)
    result.update(changes=changes, clean=not changes, merge_pending=(repository / ".git/MERGE_HEAD").exists())
    return result


def create_branch(repository: Path, record: dict, *, branch_name: str) -> dict:
    """Fork from the current candidate or baseline and switch to the new branch."""
    validate_branch(repository, branch_name)
    if branch_name in branches(repository)["local_branches"]:
        raise ValueError("Local branch already exists")
    require_clean(repository)
    validate_tree(repository, record, "HEAD")
    git(repository, "switch", "--no-track", "--create", branch_name)
    return {"status": "created", **branches(repository)}


def switch_branch(repository: Path, record: dict, *, branch_name: str) -> dict:
    """Switch only to an existing local ref, including the baseline."""
    validate_branch(repository, branch_name)
    if branch_name not in branches(repository)["local_branches"]:
        raise ValueError("Unknown local branch")
    require_clean(repository)
    validate_tree(repository, record, resolve_revision(repository, "refs/heads/" + branch_name))
    git(repository, "switch", "--no-guess", branch_name)
    return {"status": "switched", **branches(repository)}


def validate_tree(repository: Path, record: dict, revision: str) -> None:
    """Only declared editable roots may differ from the prepared baseline."""
    roots = [Path(relative_path(path)) for path in record["code"]["candidate_paths"]]
    changed = git(
        repository,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--name-only",
        "--no-renames",
        "-z",
        record["code"]["initial_commit"],
        revision,
        "--",
        strip=False,
    )
    for name in filter(None, changed.split("\0")):
        path = Path(relative_path(name))
        if ".git" in path.parts or not any(path.is_relative_to(root) for root in roots):
            raise ValueError(f"Change outside candidate_paths: {name}")
    for entry in filter(None, git(repository, "ls-tree", "-r", "-z", revision, strip=False).split("\0")):
        header, name = entry.split("\t", 1)
        mode, kind, _oid = header.split()
        if mode not in {"100644", "100755"} or kind != "blob" or ".git" in Path(relative_path(name)).parts:
            raise ValueError(f"Candidate must contain regular files only: {name}")


def merge_branch(repository: Path, record: dict, *, branch_name: str) -> dict:
    """Stage compatible candidates; automatically abort conflicts."""
    validate_branch(repository, branch_name)
    state = branches(repository)
    if state["current_branch"] == INITIAL_BRANCH:
        raise ValueError("Cannot merge into init")
    if branch_name in (INITIAL_BRANCH, state["current_branch"]):
        raise ValueError("Select another candidate branch")
    if branch_name not in state["candidate_branches"]:
        raise ValueError("Unknown candidate branch")
    require_clean(repository)
    validate_tree(repository, record, "HEAD")
    validate_tree(repository, record, resolve_revision(repository, "refs/heads/" + branch_name))
    try:
        git(
            repository,
            "-c",
            "user.name=Home",
            "-c",
            "user.email=home@localhost",
            "merge",
            "--no-commit",
            "--no-ff",
            branch_name,
        )
    except Exception:
        if (repository / ".git/MERGE_HEAD").exists():
            git(repository, "merge", "--abort")
        raise
    merge_status = "merged_not_committed" if (repository / ".git/MERGE_HEAD").exists() else "already_up_to_date"
    return {"status": merge_status, "source_branch": branch_name, **branches(repository)}


def commit(repository: Path, record: dict, *, message: str) -> dict:
    """Commit all candidate changes without rewriting existing history."""
    if current_branch(repository) == INITIAL_BRANCH:
        raise ValueError("Create a candidate branch before committing; init is protected")
    # Validate existing staged changes before staging anything else, then the complete index.
    validate_tree(repository, record, git(repository, "write-tree"))
    changes = status(repository, record)["changes"]
    roots = [Path(value) for value in record["code"]["candidate_paths"]]
    for change in changes:
        for name in (change["path"], change.get("original_path", change["path"])):
            path = Path(relative_path(name))
            if ".git" in path.parts or not any(path.is_relative_to(root) for root in roots):
                raise ValueError(f"Change outside candidate_paths: {name}")
            target = repository / path
            if target.is_symlink():
                raise ValueError(f"Candidate must contain regular files only: {name}")
    git(repository, "add", "--all", "--", ".")
    validate_tree(repository, record, git(repository, "write-tree"))
    git(repository, "-c", "user.name=Home", "-c", "user.email=home@localhost", "commit", "--message", message)
    return {"status": "committed", **branches(repository)}


def diff(repository: Path, _record: dict, *, base_revision: str | None = None, revision: str | None = None) -> dict:
    """Compare commits or all tracked changes against HEAD; never run external diff helpers."""
    if revision is not None and base_revision is None:
        raise ValueError("revision requires base_revision")
    base = resolve_revision(repository, base_revision or "HEAD")
    target = resolve_revision(repository, revision) if revision is not None else None
    output = git(
        repository,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "--ignore-submodules=all",
        base,
        *([target] if target else []),
        "--",
    )
    return {
        "base_revision": base,
        "revision": target,
        "content": output[:MAX_OUTPUT],
        "truncated": len(output) > MAX_OUTPUT,
    }
