"""Recoverable publication of Home preparation outputs under the workspace lock."""

import json
import os
from pathlib import Path
import shutil
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .workspace import _validate_directory

OUTPUTS = ("code/runtime", "code/repo", "datasets/training-set", "datasets/debug-cases", "domain_spec.yaml")
TRANSACTION = "preparation-transaction"


class Journal(BaseModel):
    """Only fixed output locations may be restored or removed during recovery."""

    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1] = 1
    existed: list[bool]


def sync_directory(path: Path) -> None:
    """Persist directory entries on POSIX."""
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def write_json(path: Path, value: dict) -> None:
    """Write a new durable transaction file."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)


def targets(root: Path, state: Path) -> list[Path]:
    """Keep completion publication last."""
    return [*[root / value for value in OUTPUTS], state / "preparation.json"]


def check_targets(root: Path, state: Path) -> list[bool]:
    """Only the original empty Home skeleton can be replaced."""
    result = []
    for path in targets(root, state):
        _validate_directory(root, path.parent)
        if path.is_symlink():
            raise ValueError(f"Symbolic link at preparation output: {path}")
        if path.exists():
            if path in {root / "domain_spec.yaml", state / "preparation.json"}:
                raise ValueError(f"Preparation output already exists: {path}")
            if not path.is_dir():
                raise ValueError(f"Preparation output already exists: {path}")
            permitted = path.relative_to(root).as_posix() in {"datasets/training-set", "datasets/debug-cases"}
            children = list(path.iterdir())
            if permitted:
                children = [
                    child
                    for child in children
                    if not (
                        child.name == "cases" and not child.is_symlink() and child.is_dir() and not any(child.iterdir())
                    )
                ]
            if children:
                raise ValueError(f"Preparation output is not empty: {path}")
        result.append(path.exists())
    return result


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def rollback(root: Path, state: Path) -> None:
    """Undo publication, restoring original skeleton directories by rename."""
    transaction = state / TRANSACTION
    journal_path = transaction / "journal.json"
    if not journal_path.exists():
        _remove(transaction)
        return
    if journal_path.is_symlink():
        raise ValueError("Invalid preparation transaction journal")
    journal = Journal.model_validate_json(journal_path.read_text(encoding="utf-8"))
    locations = targets(root, state)
    if len(journal.existed) != len(locations):
        raise ValueError("Invalid preparation transaction output count")
    for index in reversed(range(len(locations))):
        path = locations[index]
        _validate_directory(root, path.parent)
        backup = transaction / f"backup-{index}"
        staged = transaction / f"output-{index}"
        if backup.exists():
            if backup.is_symlink() or not backup.is_dir():
                raise ValueError("Invalid preparation backup")
            _remove(path)
            os.replace(backup, path)
        elif not journal.existed[index] and not staged.exists():
            _remove(path)
        sync_directory(path.parent)
    if check_targets(root, state) != journal.existed:
        raise RuntimeError("Preparation rollback could not restore the original output locations")
    _remove(transaction)
    sync_directory(state)


def recover_preparation(root: Path, state: Path) -> None:
    """Recover interrupted work before initialization or another preparation."""
    transaction = state / TRANSACTION
    if transaction.is_symlink() or (transaction.exists() and not transaction.is_dir()):
        raise ValueError("Invalid Home preparation transaction directory")
    if not transaction.exists():
        return
    marker = state / "preparation.json"
    if marker.is_symlink():
        raise ValueError("Invalid Home preparation completion marker")
    if marker.exists():
        # Completion is published last; a crash during cleanup must not undo success.
        json.loads(marker.read_text(encoding="utf-8"))
        _remove(transaction)
        sync_directory(state)
    else:
        rollback(root, state)


def publish(root: Path, state: Path, existed: list[bool], check_cancelled) -> None:
    """Publish all staged outputs, rolling back any reported failure."""
    transaction = state / TRANSACTION
    # fsync file contents and directory entries before publishing the journal.
    for path in transaction.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    for path in sorted((p for p in transaction.rglob("*") if p.is_dir()), reverse=True):
        sync_directory(path)
    temporary = transaction / "journal.tmp"
    write_json(temporary, Journal(existed=existed).model_dump())
    os.replace(temporary, transaction / "journal.json")
    sync_directory(transaction)
    for index, path in enumerate(targets(root, state)):
        check_cancelled()
        if existed[index]:
            os.replace(path, transaction / f"backup-{index}")
            sync_directory(path.parent)
            sync_directory(transaction)
        os.replace(transaction / f"output-{index}", path)
        sync_directory(path.parent)
        sync_directory(transaction)
