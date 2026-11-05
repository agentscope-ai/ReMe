"""Shared directory rules and file snapshots for watchers and catalogs."""

from dataclasses import dataclass, field
from pathlib import Path
import posixpath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..schema import ApplicationConfig


@dataclass
class WatchRule:
    """A single directory-monitoring rule."""

    path: Path
    suffixes: list[str] = field(default_factory=list)
    filenames: list[str] = field(default_factory=list)


def build_watch_rules(
    app_config: "ApplicationConfig",
    workspace_path: Path,
    *,
    watch_dirs: list[str],
    watch_suffixes: list[str],
    watch_filenames: list[str] | None = None,
) -> list[WatchRule]:
    """Build shared directory, suffix, and exact-filename rules."""
    rules: list[WatchRule] = []
    for dir_field in watch_dirs:
        literal_path = Path(dir_field)
        if literal_path.is_absolute():
            rule_path = literal_path
        else:
            config_field, separator, child_path = dir_field.partition("/")
            dir_value = getattr(app_config, config_field, config_field)
            if hasattr(app_config, config_field) and dir_value in (None, ""):
                dir_value = getattr(type(app_config)(), config_field)
            if separator:
                dir_value = posixpath.normpath(f"{dir_value}/{child_path}".strip().replace("\\", "/").strip("/"))
                if dir_value == ".":
                    dir_value = ""
            dir_name = Path(dir_value)
            rule_path = dir_name if dir_name.is_absolute() else workspace_path / dir_name
        rules.append(WatchRule(path=rule_path, suffixes=list(watch_suffixes), filenames=list(watch_filenames or [])))
    return rules


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
    """Require both configured filters; filenames match exactly."""
    if rule.filenames and p.name not in rule.filenames:
        return False
    filename = p.name.casefold()
    if rule.suffixes and not any(filename.endswith("." + suffix.strip(".").casefold()) for suffix in rule.suffixes):
        return False
    return True


def diff_files(existing: dict[str, float], previous: dict[str, float]) -> tuple[list[dict], dict[str, int]]:
    """Compare path/mtime snapshots without changing either one."""
    groups = {
        "added": sorted(existing.keys() - previous.keys()),
        "modified": sorted(path for path in existing.keys() & previous.keys() if existing[path] != previous[path]),
        "deleted": sorted(previous.keys() - existing.keys()),
    }
    return (
        [{"change": change, "path": path} for change, paths in groups.items() for path in paths],
        {change: len(paths) for change, paths in groups.items()},
    )
