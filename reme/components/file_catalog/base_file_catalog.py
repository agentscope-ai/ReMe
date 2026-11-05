"""Abstract base class for file catalog backends."""

from abc import abstractmethod
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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
        ttl_days: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        for value in (watch_dirs, watch_suffixes, watch_filenames):
            if value is not None and (not isinstance(value, list) or any(not isinstance(item, str) for item in value)):
                raise ValueError("watch_dirs, watch_suffixes, and watch_filenames must be lists of strings")
        if ttl_days is not None and (type(ttl_days) is not int or ttl_days < 1):
            raise ValueError("ttl_days must be a positive integer")
        self.watch_dirs = list(watch_dirs or [])
        self.watch_suffixes = list(watch_suffixes or [])
        self.watch_filenames = list(watch_filenames or [])
        self.ttl_days = ttl_days

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

    def _today(self) -> date:
        tz_name = self.app_context.app_config.timezone if self.app_context else None
        return datetime.now(ZoneInfo(tz_name) if tz_name else None).date()

    @staticmethod
    def _expired_date(name: str, today: date, ttl_days: int | None) -> bool:
        if ttl_days is None:
            return False
        try:
            day = date.fromisoformat(name)
        except ValueError:
            return False
        return day.isoformat() == name and (today - day).days >= ttl_days

    def _expired(self, path: Path, rules: list[WatchRule], today: date) -> bool:
        for rule in rules:
            if path.is_relative_to(rule.path):
                parts = path.relative_to(rule.path).parts
                if len(parts) > 1 and self._expired_date(parts[0], today, self.ttl_days):
                    return True
        return False

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

    async def changes(self, limit: int = 20) -> list[str]:
        """Return up to ``limit`` existing, unexpired new or modified paths awaiting processing in path order."""
        raise NotImplementedError(f"{type(self).__name__} does not support listing changes")

    async def processed(self, paths: list[str]) -> bool:
        """Mark current file versions as processed in one batch; sources must remain immutable during processing."""
        raise NotImplementedError(f"{type(self).__name__} does not support marking paths as processed")

    async def cleanup_expired(self, directory: str | None = None, ttl_days: int | None = None) -> dict:
        """Delete expired YYYY-MM-DD directories and their catalog entries.

        Retention uses directory dates in the application timezone, not file mtimes.
        Backends must explicitly implement filesystem cleanup; ``delete`` remains metadata-only.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support directory cleanup")
