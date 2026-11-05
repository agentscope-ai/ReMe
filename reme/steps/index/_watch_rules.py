"""Resolve watch rules from Job inputs or the selected catalog."""

from pathlib import Path
from typing import TYPE_CHECKING

from ...utils.file_watch import WatchRule, build_watch_rules, collect_existing, match_file

if TYPE_CHECKING:
    from ...schema import ApplicationConfig
    from ...components.file_catalog import BaseFileCatalog
    from ...components.runtime_context import RuntimeContext

__all__ = ["WatchRule", "build_watch_rules", "collect_existing", "match_file", "build_context_watch_rules"]


def build_context_watch_rules(
    app_config: "ApplicationConfig | None",
    workspace_path: Path,
    context: "RuntimeContext",
    catalog: "BaseFileCatalog | None" = None,
) -> list[WatchRule]:
    """Keep explicit Job rules; otherwise use the selected catalog's shared scope."""
    if "watch_dirs" not in context and catalog is not None:
        return catalog.watch_rules
    if app_config is None:
        return []
    watch_dirs: list[str] = context.get("watch_dirs", [])
    watch_suffixes: list[str] = context.get("watch_suffixes", [])
    watch_filenames: list[str] = context.get("watch_filenames", [])
    if not watch_dirs:
        return []
    return build_watch_rules(
        app_config,
        workspace_path,
        watch_dirs=watch_dirs,
        watch_suffixes=watch_suffixes,
        watch_filenames=watch_filenames,
    )
