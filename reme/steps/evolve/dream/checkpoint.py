"""Workspace-local, atomic checkpoints for interrupted Dream invocations."""

import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from ....schema import DreamState, DreamUnit


def _receipt(state: DreamState, index: int) -> dict:
    unit = state.units[index]
    return next(
        result
        for result in state.integrate_results
        if (result.get("unit"), result.get("bucket"), result.get("paths"))
        == (unit["name"], unit["bucket"], unit["paths"])
    )


def integration_lock(step) -> asyncio.Lock:
    """Serialize digest integration and catalog finalization within one Application."""
    if step.app_context is None:
        raise RuntimeError("dream integration requires app_context")
    key = "dream_integration_lock"
    lock = step.app_context.metadata.get(key)
    if lock is None:
        lock = asyncio.Lock()
        step.app_context.metadata[key] = lock
    if not isinstance(lock, asyncio.Lock):
        raise TypeError(f"app_context.metadata[{key!r}] must be an asyncio.Lock")
    return lock


class DreamCheckpoint(BaseModel):
    """Versioned intermediate results; source watermarks remain in the dream catalog."""

    version: Literal[1] = 1
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    sources: dict[str, tuple[int, int, str]]
    state: DreamState
    completed: list[int] = Field(default_factory=list)
    targets: dict[str, str] = Field(default_factory=dict)
    active: int | None = None
    active_before: dict[str, tuple[int, int]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_progress(self):
        """Reject invalid progress rather than silently skipping unprocessed units."""
        for unit in self.state.units:
            parsed = DreamUnit.model_validate(unit)
            if not parsed.paths or any(
                p not in self.sources or p not in self.state.changed_paths for p in parsed.paths
            ):
                raise ValueError("dream checkpoint unit references an unscanned source")
        indices = [*self.completed, *([] if self.active is None else [self.active])]
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(self.state.units) for i in indices):
            raise ValueError("invalid dream checkpoint unit indices")
        try:
            for index in self.completed:
                _receipt(self.state, index)
        except StopIteration as error:
            raise ValueError("completed dream unit has no integration receipt") from error
        return self


def source_versions(workspace: Path, paths: dict[str, float]) -> dict[str, tuple[int, int, str]]:
    """Fingerprint inputs before extraction, including edits that preserve mtimes."""
    versions = {}
    for rel in paths:
        path = (workspace / rel).resolve()
        if not path.is_relative_to(workspace):
            raise ValueError(f"dream source escapes workspace: {rel}")
        stat = path.stat()
        if stat.st_mtime != paths[rel]:
            raise RuntimeError(f"dream source changed after scanning; retry the job: {rel}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = path.stat()
        if (after.st_mtime_ns, after.st_size) != (stat.st_mtime_ns, stat.st_size):
            raise RuntimeError(f"dream source changed while fingerprinting; retry the job: {rel}")
        versions[rel] = (stat.st_mtime_ns, stat.st_size, digest)
    return versions


class CheckpointStore:
    """One checkpoint per scan window and extraction options, owned by this request."""

    def __init__(self, step, state: DreamState, max_units: int):
        workspace = Path(state.workspace).resolve()
        self.workspace = workspace
        options = [state.dates, state.daily_dir, state.hint, max_units, step.config_value("digest_dir")]
        options.append(step.file_catalog.name)
        key = hashlib.sha256(json.dumps(options, ensure_ascii=False).encode("utf-8")).hexdigest()
        self.path = (workspace / step.config_value("metadata_dir") / "dream" / f"{key}.json").resolve()
        if not self.path.is_relative_to(workspace):
            raise ValueError("dream checkpoint escapes workspace")
        self.logger = step.logger
        self.record: DreamCheckpoint | None = None
        self.resumed = False

    def read(self) -> DreamCheckpoint | None:
        """Treat malformed/unsupported saved state as unusable, with a visible diagnostic."""
        try:
            return DreamCheckpoint.model_validate_json(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except ValueError as error:
            self.logger.warning(f"[AutoDream] ignoring invalid checkpoint {self.path.name}: {error}")
            return None

    def restore(self, sources: dict[str, tuple[int, int, str]]) -> DreamState | None:
        """Reuse extraction only when the entire scanned input window is unchanged."""
        record = self.read()
        if record is None:
            return None
        if record.sources != sources:
            self.logger.info("[AutoDream] checkpoint invalidated: source files changed; extracting from scratch")
            return None
        if Path(record.state.workspace).resolve() != self.workspace:
            self.logger.warning("[AutoDream] ignoring checkpoint for another workspace")
            return None
        self.record = record
        self.resumed = True
        state = self._prepare()
        self.logger.info(
            f"[AutoDream] resuming: {len(record.completed)} of {len(state.units)} units completed from checkpoint, "
            f"{len(sources)} sources unchanged; {len(state.units) - len(record.completed)} units pending",
        )
        return state

    def publish(self, state: DreamState, sources: dict[str, tuple[int, int, str]]) -> DreamState:
        """First extraction wins when identical invocations overlap within an Application."""
        restored = self.restore(sources)
        if restored is not None:
            return restored
        self.record = DreamCheckpoint(sources=sources, state=state)
        self._write()
        return state

    def refresh(self) -> DreamState:
        """Reload progress after acquiring the existing application-wide integration lock."""
        self.record = self._current()
        return self._prepare()

    def complete(self, index: int, state: DreamState) -> None:
        """Acknowledge a unit together with the content version of its actual digest target."""
        assert self.record is not None
        target = _receipt(state, index)["target_path"]
        version = self._target_version(target)
        if version is None:
            raise RuntimeError(f"Dream digest target disappeared before completion: {target}")
        self.record.targets[target] = version
        self.record.completed.append(index)

    def ensure_current(self, *, complete: bool) -> None:
        """Reject stale finalizers and do not checkpoint missing or unacknowledged digest content."""
        current = self._current()
        if complete:
            if len(current.completed) != len(current.state.units):
                raise RuntimeError("Dream integration progress changed before Finish; retry the job")
            if self._changed_targets(current):
                raise RuntimeError("Dream digest changed before Finish; retry the job")

    def _current(self) -> DreamCheckpoint:
        current = self.read()
        if current is None or self.record is None or current.run_id != self.record.run_id:
            raise RuntimeError("Dream checkpoint was replaced or completed by another invocation; retry the job")
        return current

    def _changed_targets(self, record: DreamCheckpoint) -> set[str]:
        targets = {_receipt(record.state, index)["target_path"] for index in record.completed}
        return {
            path
            for path in targets
            if not record.targets.get(path) or record.targets[path] != self._target_version(path)
        }

    def _target_version(self, relative: str) -> str | None:
        path = (self.workspace / relative).resolve()
        if not path.is_relative_to(self.workspace):
            raise ValueError("dream digest target escapes workspace")
        try:
            with path.open("rb") as stream:
                return hashlib.file_digest(stream, "sha256").hexdigest()
        except FileNotFoundError:
            return None

    def _prepare(self) -> DreamState:
        """Recheck landed content before skipping completed units, including after waiting for the lock."""
        assert self.record is not None
        state = self.record.state
        if Path(state.workspace).resolve() != self.workspace:
            raise ValueError("dream checkpoint belongs to another workspace")
        invalid = self._changed_targets(self.record)
        if invalid:
            self.logger.info(f"[AutoDream] resume recheck: completed digest targets changed: {sorted(invalid)}")
            self.record.completed = [
                index for index in self.record.completed if _receipt(state, index)["target_path"] not in invalid
            ]
            state.integrate_results = [
                result for result in state.integrate_results if result["target_path"] not in invalid
            ]
            state.modified_paths = list(dict.fromkeys([*state.modified_paths, *sorted(invalid)]))
            self.resumed = True
        state.failed_units, state.failed_paths, state.errors, state.checkpoint_paths = [], [], [], []
        return state

    def save(self, state: DreamState) -> None:
        """Fail closed on persistence errors so Finish cannot acknowledge lost progress."""
        self._current()
        self.record.state = state
        self._write()

    def remove(self) -> None:
        """Only remove this generation, after the final catalog checkpoint is durable."""
        current = self.read()
        if current is not None and self.record is not None and current.run_id == self.record.run_id:
            self.path.unlink()

    def _write(self) -> None:
        """Publish complete JSON with atomic replacement; never truncate the last checkpoint."""
        assert self.record is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{self.path.stem}-", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(self.record.model_dump_json(indent=2))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)
