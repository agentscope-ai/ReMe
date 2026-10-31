"""Atomic, transparent processing state and operation journals."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from reme.steps.file_io._path import resolve_path


def digest(value: str | bytes) -> str:
    """Stable internal key, never an experience ID."""
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def json_text(value) -> str:
    """Deterministic UTF-8 JSON for versions and files."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def checked_path(workspace: Path, relative: str, root: Path | None = None) -> Path:
    """Use core workspace containment and additionally confine to a plugin directory."""
    if not relative or Path(relative).is_absolute():
        raise ValueError("Path must be workspace-relative")
    target, error = resolve_path(workspace, relative)
    if error or target is None:
        raise ValueError(error or "Invalid path")
    if root is not None and (target == root.resolve() or not target.is_relative_to(root.resolve())):
        raise ValueError("Path is outside the configured directory")
    return target


def atomic_json(path: Path, value) -> None:
    """Publish a complete JSON file with replace, never a partially written document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".procedure-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json_text(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_json(path: Path) -> dict:
    """Load one state file, failing explicitly on corrupt state."""
    return json.loads(path.read_text(encoding="utf-8"))
