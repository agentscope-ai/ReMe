"""Per-workspace path views for multi-workspace applications."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .application_config import ApplicationConfig


@dataclass(frozen=True)
class WorkspaceContext:
    """Resolved paths for one named workspace of an application.

    Built from ApplicationConfig when the application context is created. The
    configured subdirectory names stay workspace-relative, matching the layout
    a single-workspace application derives from the same config.
    """

    alias: str
    workspace_path: Path
    metadata_path: Path | None
    session_path: Path | None
    mem_session_path: Path | None
    resource_path: Path | None
    daily_path: Path | None
    digest_path: Path | None

    @classmethod
    def from_config(cls, alias: str, workspace: str, config: "ApplicationConfig") -> "WorkspaceContext":
        """Resolve one alias of the config's workspaces mapping into absolute paths."""
        root = Path(workspace).absolute()

        def child(subdir: str) -> Path | None:
            return root / subdir if subdir else None

        return cls(
            alias=alias,
            workspace_path=root,
            metadata_path=child(config.metadata_dir),
            session_path=child(config.session_dir),
            mem_session_path=child(config.mem_session_dir),
            resource_path=child(config.resource_dir),
            daily_path=child(config.daily_dir),
            digest_path=child(config.digest_dir),
        )

    def storage_dirs(self) -> list[Path]:
        """Subdirectories the application materializes at startup, in configuration order."""
        return [
            path
            for path in (
                self.metadata_path,
                self.session_path,
                self.mem_session_path,
                self.resource_path,
                self.daily_path,
                self.digest_path,
            )
            if path is not None
        ]
