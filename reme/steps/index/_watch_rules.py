"""Shared watch-rule logic for init_changes and watch_changes steps."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...schema import ApplicationConfig
    from ...components.runtime_context import RuntimeContext


@dataclass
class WatchRule:
    """A single directory-monitoring rule."""

    path: Path
    suffixes: list[str] = field(default_factory=list)


def build_watch_rules(
    app_config: "ApplicationConfig",
    workspace_path: Path,
    *,
    watch_dirs: list[str],
    watch_suffixes: list[str],
) -> list[WatchRule]:
    """Build watch rules from application config fields and suffix whitelist."""
    rules: list[WatchRule] = []
    for dir_field in watch_dirs:
        dir_name = getattr(app_config, dir_field, dir_field)
        rules.append(WatchRule(path=workspace_path / dir_name, suffixes=list(watch_suffixes)))
    return rules


def build_context_watch_rules(
    app_config: "ApplicationConfig | None",
    workspace_path: Path,
    context: "RuntimeContext",
) -> list[WatchRule]:
    """Build watch rules from context-level watch_dirs/watch_suffixes."""
    if app_config is None:
        return []
    watch_dirs: list[str] = context.get("watch_dirs", [])
    watch_suffixes: list[str] = context.get("watch_suffixes", [])
    if not watch_dirs:
        return []
    return build_watch_rules(app_config, workspace_path, watch_dirs=watch_dirs, watch_suffixes=watch_suffixes)


def collect_existing(rules: list[WatchRule], recursive: bool) -> dict[str, float]:
    """Walk rule paths and return {abs_path: st_mtime} for matching files."""
    existing: dict[str, float] = {}
    for rule in rules:
        if not rule.path.exists():
            continue
        candidates = rule.path.rglob("*") if recursive else rule.path.iterdir()
        for p in candidates:
            if not p.is_file():
                continue
            if not _match_rule(p, rule):
                continue
            abs_p = p.absolute()
            existing[str(abs_p)] = abs_p.stat().st_mtime
    return existing


def match_file(file_path: str, rules: list[WatchRule]) -> bool:
    """Return True if a file path matches any of the watch rules."""
    p = Path(file_path)
    for rule in rules:
        try:
            p.relative_to(rule.path)
        except ValueError:
            continue
        if _match_rule(p, rule):
            return True
    return False


def _match_rule(p: Path, rule: WatchRule) -> bool:
    """Check if a single path matches a rule's suffix constraint."""
    if rule.suffixes and not any(p.name.endswith("." + s.strip(".")) for s in rule.suffixes):
        return False
    return True
