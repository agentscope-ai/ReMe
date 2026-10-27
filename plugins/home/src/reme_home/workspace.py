"""Initialize Home's layout inside an existing ReMe workspace."""

import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict

from reme.components.base_component import BaseComponent

DIRECTORIES = (
    "code/repo",
    "code/runtime",
    "datasets/training-set/cases",
    "datasets/debug-cases/cases",
    "docs",
    "evaluations",
    "sampling/plans",
    "tmp",
    "logs",
)
ROOT_DIRECTORIES = tuple(dict.fromkeys(path.split("/", maxsplit=1)[0] for path in DIRECTORIES))


class WorkspaceManifest(BaseModel):
    """Layout identity, independent of future experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    workspace_id: UUID
    created_at: AwareDatetime


def _validate_directory(root: Path, path: Path) -> None:
    """Reject links and non-directories anywhere along a managed path."""
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise ValueError(f"Home workspace requires a real directory: {current}")


@contextmanager
def _initialization_lock(path: Path):
    """Hold a process lock; the persistent lock file is not an ownership marker."""
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"Invalid Home workspace lock: {path}")
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt  # pylint: disable=import-outside-toplevel

            stream.seek(0, os.SEEK_END)
            if not stream.tell():
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)

            def acquire():
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)

            def release():
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)

        else:
            import fcntl  # pylint: disable=import-outside-toplevel

            def acquire():
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release():
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

        try:
            acquire()
        except OSError as exc:
            raise RuntimeError(f"Home workspace initialization is already in progress: {path}") from exc
        try:
            yield
        finally:
            stream.seek(0)
            release()


def _publish_manifest(path: Path, manifest: WorkspaceManifest) -> None:
    """Publish the initialization marker only after the layout is complete."""
    descriptor, name = tempfile.mkstemp(prefix=".workspace-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


class HomeWorkspace(BaseComponent):
    """Create Home directories at application startup, preserving ReMe data."""

    component_type = "home.workspace"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.manifest: WorkspaceManifest | None = None

    async def _start(self) -> None:
        if self.app_context is None:
            raise RuntimeError("HomeWorkspace requires an ApplicationContext")
        root = self.workspace_path.resolve()
        config = self.app_context.app_config
        metadata = root / config.metadata_dir
        if (
            Path(config.metadata_dir).is_absolute()
            or metadata == root
            or not metadata.is_relative_to(root)
            or ".." in metadata.relative_to(root).parts
        ):
            raise ValueError("Home metadata_dir must be a non-empty workspace-relative directory")
        for name in (
            config.metadata_dir,
            config.session_dir,
            config.mem_session_dir,
            config.resource_dir,
            config.daily_dir,
            config.digest_dir,
        ):
            if not name:
                continue
            configured = (root / name).resolve()
            for directory in ROOT_DIRECTORIES:
                managed = root / directory
                if configured.is_relative_to(managed) or managed.is_relative_to(configured):
                    raise ValueError(f"Home directory {directory!r} conflicts with ReMe directory {name!r}")
        state = metadata / "home"
        _validate_directory(root, state)
        state.mkdir(parents=True, exist_ok=True)
        with _initialization_lock(state / "workspace.lock"):
            from .preparation_transaction import recover_preparation  # pylint: disable=import-outside-toplevel

            recover_preparation(root, state)
            self._initialize(root, state / "workspace.json")

    def _initialize(self, root: Path, marker: Path) -> None:
        if marker.is_symlink() or (marker.exists() and not marker.is_file()):
            raise ValueError(f"Invalid Home workspace manifest: {marker}")
        manifest = (
            WorkspaceManifest.model_validate_json(marker.read_text(encoding="utf-8")) if marker.exists() else None
        )
        for relative in DIRECTORIES:
            _validate_directory(root, root / relative)
        if manifest is None:
            for directory in ROOT_DIRECTORIES:
                path = root / directory
                if path.exists() and any(path.iterdir()):
                    raise ValueError(f"Refusing to initialize non-empty Home directory: {path}")
        created: list[Path] = []
        try:
            for relative in DIRECTORIES:
                current = root
                for part in Path(relative).parts:
                    current = current / part
                    if not current.exists():
                        current.mkdir()
                        created.append(current)
            if manifest is None:
                manifest = WorkspaceManifest(workspace_id=uuid4(), created_at=datetime.now(timezone.utc))
                _publish_manifest(marker, manifest)
            self.manifest = manifest
        except BaseException:
            if not marker.exists():
                for path in reversed(created):
                    try:
                        path.rmdir()
                    except OSError:
                        pass  # Preserve files created concurrently; never recursively delete user data.
            raise
