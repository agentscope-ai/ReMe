"""Filesystem boundary and lifecycle management for a Meta-ReMe workspace."""

from __future__ import annotations

from contextlib import AbstractContextManager
import errno
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import tempfile
from typing import Any

from pydantic import BaseModel, ValidationError
import yaml

from models import DomainSpec, WorkspaceLockOwner, WorkspaceManifest, canonical_json_bytes, model_fingerprint

WORKSPACE_MANIFEST = ".meta-reme-workspace.json"
WORKSPACE_LOCK = ".meta-reme.lock"
_DIRECTORIES = (
    "code/repo",
    "code/worktrees",
    "dataset/cases",
    "weaknesses",
    "proposals",
    "evaluations",
    "logs",
)


class WorkspaceError(RuntimeError):
    """Base error for invalid or unsafe workspace operations."""


class WorkspaceLockedError(WorkspaceError):
    """Raised when another process owns a workspace lock."""


class WorkspaceFormatError(WorkspaceError):
    """Raised when a directory is not a valid compatible workspace."""


class Workspace:
    """A validated Meta-ReMe workspace rooted at an absolute path."""

    def __init__(self, root: Path, manifest: WorkspaceManifest) -> None:
        self.root = Path(root).resolve()
        self.manifest = manifest

    @classmethod
    def create(cls, root: Path, domain_spec: DomainSpec) -> "Workspace":
        """Create a new workspace without overwriting an existing directory."""

        root = Path(root).resolve()
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise WorkspaceError(f"Refusing to initialize non-empty workspace: {root}")

        root.mkdir(parents=True, exist_ok=True)
        manifest = WorkspaceManifest(domain_fingerprint=model_fingerprint(domain_spec))
        workspace = cls(root, manifest)
        try:
            for relative in _DIRECTORIES:
                (root / relative).mkdir(parents=True, exist_ok=False)
            workspace.atomic_write_text(
                "domain_spec.yaml",
                yaml.safe_dump(
                    domain_spec.model_dump(mode="json", exclude_none=False),
                    allow_unicode=True,
                    sort_keys=True,
                ),
            )
            # The manifest is the publication marker and must be written last.
            workspace.atomic_write_json(WORKSPACE_MANIFEST, manifest)
            return workspace
        except BaseException:
            # Only remove a directory created by this call and only while it has
            # not yet become a published workspace.
            if not (root / WORKSPACE_MANIFEST).exists():
                shutil.rmtree(root, ignore_errors=True)
            raise

    @classmethod
    def open(cls, root: Path, domain_spec: DomainSpec | None = None) -> "Workspace":
        """Open and validate an existing workspace."""

        root = Path(root).resolve()
        manifest_path = root / WORKSPACE_MANIFEST
        if not root.is_dir() or not manifest_path.is_file():
            raise WorkspaceFormatError(f"Not a Meta-ReMe workspace: {root}")
        try:
            manifest = WorkspaceManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as exc:
            raise WorkspaceFormatError(f"Invalid workspace manifest: {manifest_path}") from exc
        legacy_dataset = root / "datasets/search"
        if legacy_dataset.is_dir() and not (root / "dataset").exists():
            raise WorkspaceFormatError(
                "Legacy dataset layout detected at datasets/search; move that directory to dataset before opening",
            )
        for relative in _DIRECTORIES:
            if not (root / relative).is_dir():
                raise WorkspaceFormatError(f"Workspace directory is missing: {relative}")
        if domain_spec is not None:
            actual = model_fingerprint(domain_spec)
            if actual != manifest.domain_fingerprint:
                raise WorkspaceFormatError("Domain spec fingerprint does not match this workspace")
        return cls(root, manifest)

    def path(self, relative: str | Path) -> Path:
        """Resolve a safe workspace-relative path."""

        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise WorkspaceError(f"Path must be workspace-relative: {relative}")
        resolved = (self.root / relative).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceError(f"Path escapes workspace: {relative}") from exc
        return resolved

    def entity_path(self, base: str | Path, identifier: str) -> Path:
        """Resolve a path for an externally supplied entity ID."""

        if not identifier or identifier in {".", ".."} or Path(identifier).name != identifier:
            raise WorkspaceError(f"Unsafe workspace identifier: {identifier!r}")
        if any(character in identifier for character in ("/", "\\", "\x00")):
            raise WorkspaceError(f"Unsafe workspace identifier: {identifier!r}")
        return self.path(Path(base) / identifier)

    def validation_case_dir(
        self,
        code_id: str,
        validation_id: str,
        case_id: str,
        *,
        create: bool = False,
    ) -> Path:
        """Return the canonical validation result directory for one case."""

        current = self.entity_path("evaluations", code_id)
        for identifier in (validation_id, "cases", case_id):
            current = self.entity_path(current.relative_to(self.root), identifier)
        if create:
            current.mkdir(parents=True, exist_ok=False)
        return current

    def atomic_write_json(self, relative: str | Path, value: BaseModel | Any) -> Path:
        """Atomically publish JSON in the workspace."""

        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json", exclude_none=False)
        content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
        return self.atomic_write_text(relative, content)

    def atomic_write_text(self, relative: str | Path, content: str) -> Path:
        """Atomically publish UTF-8 text in the workspace."""

        return self.atomic_write_bytes(relative, content.encode("utf-8"))

    def atomic_write_bytes(self, relative: str | Path, content: bytes) -> Path:
        """Write, fsync, and atomically replace a file in its destination directory."""

        destination = self.path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
            return destination
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def acquire_lock(self) -> "WorkspaceLock":
        """Acquire the single-writer workspace lock."""

        lock = WorkspaceLock(self.path(WORKSPACE_LOCK))
        lock.acquire()
        return lock

    def install_dataset(self, source: Path) -> Path:
        """Copy a normalized dataset into the workspace and make it read-only."""

        source = Path(source).resolve()
        destination = self.path("dataset")
        if not source.is_dir():
            raise WorkspaceError(f"Normalized search dataset is not a directory: {source}")
        if any(path.is_file() or path.is_symlink() for path in destination.rglob("*")):
            raise WorkspaceError(f"Dataset has already been installed: {destination}")
        for path in source.rglob("*"):
            if path.is_symlink():
                raise WorkspaceError(f"Dataset may not contain symbolic links: {path.relative_to(source)}")
        temporary = self.path(".dataset-installing")
        if temporary.exists():
            raise WorkspaceError(f"Incomplete dataset installation requires inspection: {temporary}")
        shutil.copytree(source, temporary, symlinks=True)
        try:
            shutil.rmtree(destination)
            os.replace(temporary, destination)
            _make_tree_read_only(destination)
            _fsync_directory(destination.parent)
            return destination
        except BaseException:
            if not destination.exists():
                shutil.rmtree(temporary, ignore_errors=True)
                destination.mkdir(parents=True, exist_ok=True)
            raise


class WorkspaceLock(AbstractContextManager["WorkspaceLock"]):
    """An atomic, ownership-checked filesystem lock."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.owner: WorkspaceLockOwner | None = None

    def acquire(self) -> None:
        """Acquire this lock, taking over only a confirmed stale local owner."""

        if self.owner is not None:
            raise WorkspaceLockedError(f"Lock is already held by this object: {self.path}")
        owner = WorkspaceLockOwner(pid=os.getpid(), hostname=socket.gethostname())
        for _ in range(2):
            try:
                descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError as exc:
                existing = self._read_owner()
                if not _is_confirmed_stale(existing):
                    raise WorkspaceLockedError(_lock_message(self.path, existing)) from exc
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(canonical_json_bytes(owner) + b"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                _fsync_directory(self.path.parent)
            except BaseException:
                self.path.unlink(missing_ok=True)
                raise
            self.owner = owner
            return
        raise WorkspaceLockedError(f"Could not acquire workspace lock: {self.path}")

    def _read_owner(self) -> WorkspaceLockOwner | None:
        """Read the current lock owner, returning None for an invalid lock."""

        try:
            return WorkspaceLockOwner.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (OSError, ValidationError):
            return None

    def release(self) -> None:
        """Release the lock only if its ownership token is unchanged."""

        if self.owner is None:
            return
        existing = self._read_owner()
        if existing is None or existing.token != self.owner.token:
            raise WorkspaceLockedError(f"Workspace lock ownership changed: {self.path}")
        self.path.unlink()
        _fsync_directory(self.path.parent)
        self.owner = None

    def __enter__(self) -> "WorkspaceLock":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.release()


def _is_confirmed_stale(owner: WorkspaceLockOwner | None) -> bool:
    if owner is None or owner.hostname != socket.gethostname():
        return False
    try:
        os.kill(owner.pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    except OSError as exc:
        return exc.errno == errno.ESRCH
    return False


def _lock_message(path: Path, owner: WorkspaceLockOwner | None) -> str:
    if owner is None:
        return f"Workspace has an unreadable lock requiring manual inspection: {path}"
    return f"Workspace is locked by pid {owner.pid} on {owner.hostname} since {owner.started_at.isoformat()}"


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _make_tree_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        mode = stat.S_IMODE(path.stat().st_mode)
        if path.is_dir():
            path.chmod(mode & ~0o222 | 0o555)
        else:
            path.chmod(mode & ~0o222 | 0o444)
    mode = stat.S_IMODE(root.stat().st_mode)
    root.chmod(mode & ~0o222 | 0o555)
