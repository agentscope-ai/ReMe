"""Candidate-code inputs shared by one or more isolated case sandboxes."""

from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import gzip
import hashlib
import io
import os
from pathlib import Path
from pathlib import PurePosixPath
import tarfile

DEFAULT_EXCLUDES = (
    ".git",
    ".reme",
    ".env",
    ".env.*",
    ".ssh",
    ".codex",
    ".claude",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "datasets",
    "benchmark",
    "baselines",
    "meta-reme",
    "output",
    "tests",
    "cookbook",
    "docs",
    "github-pages",
    "integrations",
    "skills",
    "reme_studio",
    "node_modules",
    "memory_workspaces",
    "logs",
    "artifacts",
    "build",
    "dist",
    "*.egg-info",
    "*.pyc",
    "*.pyo",
    "*.log",
)


def _is_excluded(relative_path: Path, patterns: tuple[str, ...]) -> bool:
    """Return whether any path component or full POSIX path is excluded."""
    posix = relative_path.as_posix()
    return any(
        fnmatch.fnmatch(posix, pattern) or any(fnmatch.fnmatch(part, pattern) for part in relative_path.parts)
        for pattern in patterns
    )


def _normalized_info(path: Path, arcname: str) -> tarfile.TarInfo:
    """Create deterministic tar metadata for a regular source file."""
    stat = path.stat()
    info = tarfile.TarInfo(arcname)
    info.size = stat.st_size
    info.mode = stat.st_mode & 0o777
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """A deterministic source archive reusable across many case sandboxes."""

    archive: bytes
    sha256: str
    file_count: int

    @classmethod
    def from_directory(
        cls,
        source_dir: str | Path,
        *,
        excludes: tuple[str, ...] = DEFAULT_EXCLUDES,
    ) -> "SourceSnapshot":
        """Snapshot a candidate directory without runtime or VCS state.

        Symbolic links are rejected instead of followed. This prevents a
        candidate archive from silently capturing files outside its root.
        """
        root = Path(source_dir).resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"candidate source is not a directory: {root}")

        files: list[tuple[Path, Path]] = []
        # Component matches exclude every descendant. Full-path matches may
        # exclude only the directory itself, so they must not prune traversal.
        directory_patterns = tuple(pattern for pattern in excludes if "/" not in pattern)
        for current, directories, filenames in os.walk(root, topdown=True, followlinks=False):
            directory = Path(current)
            for name in directories[:]:
                path = directory / name
                if any(fnmatch.fnmatch(name, pattern) for pattern in directory_patterns):
                    directories.remove(name)
                elif path.is_symlink() and not _is_excluded(path.relative_to(root), excludes):
                    raise ValueError(f"candidate source contains a symbolic link: {path.relative_to(root)}")
            for name in filenames:
                path = directory / name
                relative = path.relative_to(root)
                if _is_excluded(relative, excludes):
                    continue
                if path.is_symlink():
                    raise ValueError(f"candidate source contains a symbolic link: {relative}")
                if path.is_file():
                    files.append((relative, path))

        raw = io.BytesIO()
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for relative, path in sorted(files, key=lambda item: item[0].as_posix()):
                    info = _normalized_info(path, relative.as_posix())
                    with path.open("rb") as source:
                        archive.addfile(info, source)

        payload = raw.getvalue()
        return cls(
            archive=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
            file_count=len(files),
        )


@dataclass(frozen=True, slots=True)
class SourceCandidate:
    """A changing source snapshot containing one or more Python projects.

    ``install_projects`` is a trusted, archive-relative installation plan.  A
    plugin directory merely being present in the snapshot is insufficient:
    installing it is what publishes its ``reme.plugins`` entry point.
    """

    snapshot: SourceSnapshot
    base_image: str = "reme-benchmark-base:agentscope-2.0.9"
    install_projects: tuple[str, ...] = (".",)

    def __post_init__(self) -> None:
        if not self.install_projects:
            raise ValueError("source candidate requires at least one install project")
        normalized: list[str] = []
        for value in self.install_projects:
            if not isinstance(value, str) or not value:
                raise TypeError("candidate install projects must be non-empty strings")
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or "\\" in value:
                raise ValueError(f"candidate install project must stay inside the archive: {value!r}")
            name = path.as_posix()
            if name in normalized:
                raise ValueError(f"candidate install project is duplicated: {value!r}")
            normalized.append(name)
        if normalized[0] != ".":
            raise ValueError("the candidate root project must be installed before plugin projects")
        object.__setattr__(self, "install_projects", tuple(normalized))

    @property
    def candidate_id(self) -> str:
        """Stable identifier for the exact source snapshot."""
        return self.snapshot.sha256


@dataclass(frozen=True, slots=True)
class ImageCandidate:
    """A ReMe candidate already installed in an immutable Docker image."""

    image: str
    candidate_id: str | None = None

    @property
    def resolved_candidate_id(self) -> str:
        """Return the caller-supplied digest/ID or the image reference."""
        return self.candidate_id or self.image
