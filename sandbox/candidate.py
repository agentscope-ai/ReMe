"""Candidate-code inputs shared by one or more isolated case sandboxes."""

from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import gzip
import hashlib
import io
from pathlib import Path
import tarfile

DEFAULT_EXCLUDES = (
    ".git",
    ".reme",
    ".env",
    ".env.*",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "datasets",
    "benchmark/datasets",
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
        for path in root.rglob("*"):
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
    """A changing ReMe source snapshot installed after sandbox creation."""

    snapshot: SourceSnapshot
    base_image: str = "reme-sandbox-base:agentscope-2.0.4-post1"

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
