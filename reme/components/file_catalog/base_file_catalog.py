"""Abstract base class for file catalog backends."""

from abc import abstractmethod

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ...schema import ApplicationConfig, FileNode
from ...utils.file_watch import WatchRule, build_watch_rules


class BaseFileCatalog(BaseComponent):
    """File-catalog backend recording FileNode entries keyed by path."""

    component_type = ComponentEnum.FILE_CATALOG

    def __init__(
        self,
        watch_dirs: list[str] | None = None,
        watch_suffixes: list[str] | None = None,
        watch_filenames: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        for value in (watch_dirs, watch_suffixes, watch_filenames):
            if value is not None and (not isinstance(value, list) or any(not isinstance(item, str) for item in value)):
                raise ValueError("watch_dirs, watch_suffixes, and watch_filenames must be lists of strings")
        self.watch_dirs = list(watch_dirs or [])
        self.watch_suffixes = list(watch_suffixes or [])
        self.watch_filenames = list(watch_filenames or [])

    @property
    def watch_rules(self) -> list[WatchRule]:
        """Return the shared workspace-contained scope for this catalog."""
        config = self.app_context.app_config if self.app_context else ApplicationConfig()
        rules = build_watch_rules(
            config,
            self.workspace_path,
            watch_dirs=self.watch_dirs,
            watch_suffixes=self.watch_suffixes,
            watch_filenames=self.watch_filenames,
        )
        if not rules:
            raise ValueError("catalog watch_dirs must not be empty")
        workspace = self.workspace_path.resolve()
        for rule in rules:
            if rule.path == workspace or not rule.path.is_relative_to(workspace) or rule.path.resolve() != rule.path:
                raise ValueError("catalog directories must stay inside the workspace and must not contain symlinks")
        return rules

    async def _start(self) -> None:
        await super()._start()
        await self.load()

    async def _close(self) -> None:
        if self.is_started:
            await self.dump()
        await super()._close()

    async def load(self) -> None:
        """Load persisted state. No-op without local files."""

    async def dump(self) -> None:
        """Persist state. No-op without local files."""

    @abstractmethod
    async def upsert(self, nodes: list[FileNode]) -> None:
        """Insert or update nodes keyed by path."""

    @abstractmethod
    async def delete(self, path: str | list[str]) -> None:
        """Delete nodes by path; missing paths are skipped."""

    @abstractmethod
    async def get_nodes(self, paths: list[str] | None = None) -> list[FileNode]:
        """Return nodes by paths; None = all; missing paths are skipped."""
