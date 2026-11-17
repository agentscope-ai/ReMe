"""Read-only navigation and comparison of exported memory Git checkpoints."""

import difflib
import hashlib
import re
import subprocess

from pydantic import Field, field_validator, model_validator

from reme.steps.base_step import BaseStep, Ref

from ..code.repository import MAX_OUTPUT, resolve_revision
from ..code.snapshot import blob, code_path, tree
from ..preparation.code import git as safe_git
from ..preparation.spec import SpecModel
from ..workspace import HomeWorkspace
from .inputs import entity_id
from .artifacts import contained
from .inspection_step import _execute, _object, _page, _run_path

MAX_TREE_FILES = 2000
MAX_TREE_BYTES = 20_000_000


def git(repository, *arguments):
    """Ignore replace refs while reusing the shared noninteractive Git executor."""
    return safe_git(repository, "--no-replace-objects", *arguments)


class MemoryScopeInput(SpecModel):
    """An existing evaluation and one manifest-selected case."""

    validation_path: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    path: str | None = None

    @field_validator("case_id")
    @classmethod
    def validate_case(cls, value):
        """Accept a single case directory segment."""
        return entity_id(value)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value):
        """Accept literal repository-relative paths only."""
        return code_path(value) if value is not None else None


class MemoryHistoryInput(MemoryScopeInput):
    """Page through checkpoints, newest first."""

    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)


class CompareMemoryInput(MemoryScopeInput):
    """Compare two commits in one or two exported repositories."""

    base_ref: str | None = Field(default=None, min_length=1)
    target_ref: str = Field(default="HEAD", min_length=1)
    target_validation_path: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_base(self):
        """Require an explicit earlier checkpoint for a same-repository comparison."""
        if self.target_validation_path is None and self.base_ref is None:
            raise ValueError("base_ref is required when target_validation_path is omitted")
        return self


def _repository(home, validation_path, case_id):
    run = _run_path(home, validation_path)
    manifest = _object(run, "manifest.json", required=True)
    if case_id not in manifest.get("case_ids", []):
        raise ValueError("case_id is not selected by this evaluation manifest")
    repository = contained(run, f"cases/{case_id}/memory_construction/reme_workspace", directory=True)
    return run, manifest, repository


def _head(repository):
    """Reject indirect Git storage and distinguish absent/empty/corrupt repositories."""
    if not repository.exists():
        return "workspace_missing", None
    metadata = contained(repository, ".git")
    if not metadata.exists():
        return "history_missing", None
    if not metadata.is_dir():
        raise ValueError("Memory history requires a standalone .git directory")
    # Exported artifacts must not redirect reads to another repository or host path.
    if any(path.is_symlink() for path in metadata.rglob("*")):
        raise ValueError("Memory Git metadata must not contain symbolic links")
    for name in ("commondir", "objects/info/alternates", "objects/info/http-alternates", "shallow"):
        if (metadata / name).exists():
            raise ValueError(f"Unsupported indirect or incomplete memory history: {name}")
    if any(metadata.glob("objects/pack/*.promisor")):
        raise ValueError("Partial memory repositories are unsupported")
    if git(repository, "for-each-ref", "--format=%(refname)", "refs/replace"):
        raise ValueError("Memory replace refs are unsupported")
    configuration = git(repository, "config", "--local", "--list")
    if any(line.lower().startswith(("include.", "includeif.")) for line in configuration.splitlines()):
        raise ValueError("Memory Git configuration must not include external files")
    if any(
        "promisor=" in line.lower() or line.lower().startswith("extensions.partialclone=")
        for line in configuration.splitlines()
    ):
        raise ValueError("Partial memory repositories are unsupported")
    if git(repository, "rev-parse", "--show-toplevel") != str(repository):
        raise ValueError("Memory history must be a standalone repository")
    if git(repository, "rev-parse", "--is-shallow-repository") != "false":
        raise ValueError("Memory history must not be shallow")
    try:
        return "available", resolve_revision(repository, "HEAD")
    except subprocess.CalledProcessError:
        # Only an unborn symbolic HEAD is an empty repository, not a broken ref.
        reference = git(repository, "symbolic-ref", "-q", "HEAD")
        if git(repository, "for-each-ref", "--format=%(refname)", reference):
            raise ValueError("Memory HEAD cannot be resolved") from None
        if not (metadata / "objects").is_dir():
            raise ValueError("Memory object database is missing") from None
        return "history_empty", None


def _resolve(repository, reference, head):
    if re.fullmatch(r"[0-9a-fA-F]+", reference or ""):
        matches = [
            sha for sha in git(repository, "rev-list", head, "--").splitlines() if sha.startswith(reference.lower())
        ]
        if len(matches) != 1:
            raise ValueError(f"Memory commit prefix has {len(matches)} matches reachable from HEAD; require one")
        return matches[0]
    revision = resolve_revision(repository, reference)
    git(repository, "merge-base", "--is-ancestor", revision, head)
    return revision


def inspect_history(home, spec):
    """Return saved Git history without refreshing the worktree or index."""
    run, _manifest, repository = _repository(home, spec.validation_path, spec.case_id)
    status, head = _head(repository)
    result = {
        "validation_path": str(run),
        "case_id": spec.case_id,
        "repository": str(repository),
        "workspace_exists": repository.exists(),
        "has_git_history": head is not None,
        "status": status,
        "head": head,
        "path": spec.path,
    }
    if head is None:
        return {**result, **_page([], spec)}
    # Literal pathspecs prevent user-controlled wildcard/pathspec expansion.
    path_args = ["--", ":(literal)" + spec.path] if spec.path is not None else ["--"]
    count = int(git(repository, "rev-list", "--count", head, *path_args))
    raw = git(
        repository,
        "log",
        "--no-show-signature",
        f"--skip={spec.offset}",
        f"--max-count={spec.limit}",
        "--format=%H%x00%P%x00%aI%x00%s",
        head,
        *path_args,
    )
    rows = []
    for line in raw.splitlines():
        sha, parents, timestamp, subject = line.split("\0", 3)
        rows.append(
            {
                "commit_sha": sha,
                "parents": parents.split(),
                "timestamp": timestamp,
                "subject": subject[:2000],
                "subject_truncated": len(subject) > 2000,
                "session_id": subject[9:2009] if subject.startswith("session: ") else None,
            },
        )
    return {
        **result,
        "row_count": count,
        "rows": rows,
        "next_offset": spec.offset + len(rows) if spec.offset + len(rows) < count else None,
    }


def _files(repository, revision, path):
    entries = tree(repository, revision, path)
    if len(entries) > MAX_TREE_FILES:
        raise ValueError("Memory comparison exceeds 2000 files; narrow path")
    result = {}
    total = 0
    for name, oid in entries:
        total += int(git(repository, "cat-file", "-s", oid))
        if total > MAX_TREE_BYTES:
            raise ValueError("Memory comparison exceeds 20 MB per snapshot; narrow path")
        result[name] = blob(repository, oid)
    return result


def _hash(files):
    digest = hashlib.sha256()
    for name, data in sorted(files.items()):
        digest.update(name.encode() + b"\0" + hashlib.sha256(data).digest())
    return digest.hexdigest()


def compare_memory(home, spec):
    """Compare committed regular-file contents, independent of dirty checkout files."""
    base_run, base_manifest, base_repo = _repository(home, spec.validation_path, spec.case_id)
    target_run, target_manifest, target_repo = _repository(
        home,
        spec.target_validation_path or spec.validation_path,
        spec.case_id,
    )
    base_status, base_head = _head(base_repo)
    target_status, target_head = _head(target_repo)
    if base_head is None or target_head is None:
        raise ValueError(f"Comparison requires memory commits on both sides: {base_status}, {target_status}")
    base_sha = _resolve(base_repo, spec.base_ref or "HEAD", base_head)
    target_sha = _resolve(target_repo, spec.target_ref, target_head)
    before = _files(base_repo, base_sha, spec.path)
    after = _files(target_repo, target_sha, spec.path)
    added = sorted(after.keys() - before.keys())
    removed = sorted(before.keys() - after.keys())
    modified = sorted(name for name in before.keys() & after.keys() if before[name] != after[name])
    parts = []
    length = 0
    truncated = False
    for name in sorted(added + removed + modified):
        left, right = before.get(name, b""), after.get(name, b"")
        try:
            if b"\0" in left or b"\0" in right:
                raise UnicodeDecodeError("utf-8", b"\0", 0, 1, "binary")
            lines = difflib.unified_diff(
                left.decode("utf-8").splitlines(keepends=True),
                right.decode("utf-8").splitlines(keepends=True),
                fromfile="base/" + name,
                tofile="target/" + name,
            )
        except UnicodeDecodeError:
            lines = iter([f"Binary files differ: {name}\n"])
        for line in lines:
            if not line.endswith("\n"):
                line += "\n\\ No newline at end of file\n"
            available = MAX_OUTPUT - length
            parts.append(line[:available])
            length += min(len(line), available)
            if len(line) > available:
                truncated = True
                break
        if truncated:
            break
    return {
        "base_validation_path": str(base_run),
        "target_validation_path": str(target_run),
        "case_id": spec.case_id,
        "path": spec.path,
        "base_commit": base_sha,
        "target_commit": target_sha,
        "base_candidate_commit": base_manifest.get("commit_sha"),
        "target_candidate_commit": target_manifest.get("commit_sha"),
        "base_fingerprints": base_manifest.get("fingerprints"),
        "target_fingerprints": target_manifest.get("fingerprints"),
        "added": added,
        "removed": removed,
        "modified": modified,
        "base_tree_sha256": _hash(before),
        "target_tree_sha256": _hash(after),
        "diff": "".join(parts),
        "truncated": truncated,
    }


class HomeMemoryInspectionStep(BaseStep):
    """Two read-only Jobs sharing one invocation-scoped Step implementation."""

    workspace: HomeWorkspace = Ref(HomeWorkspace, "home.workspace")

    async def execute(self):
        """Select the operation from host configuration, never model input."""
        operations = {"history": (MemoryHistoryInput, inspect_history), "compare": (CompareMemoryInput, compare_memory)}
        model, function = operations[self.kwargs["operation"]]
        return await _execute(self, model, function)
