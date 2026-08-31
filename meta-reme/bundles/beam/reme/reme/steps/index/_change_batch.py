"""Helpers for the common file change batch shape."""

from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

from watchfiles import Change


def _normalize_change(raw) -> Change | None:
    if isinstance(raw, Change):
        return raw
    if isinstance(raw, str):
        return Change.__members__.get(raw)
    return None


def coalesce_changes(changes: list[dict], path_exists: Callable[[str], bool] | None = None) -> list[dict]:
    """Collapse duplicate path events to the final on-disk state."""
    by_path: OrderedDict[str, set[Change]] = OrderedDict()
    for item in changes:
        if not isinstance(item, dict) or "path" not in item:
            continue
        change = _normalize_change(item.get("change"))
        if change not in (Change.added, Change.modified, Change.deleted):
            continue
        by_path.setdefault(item["path"], set()).add(change)

    def exists(path: str) -> bool:
        return path_exists(path) if path_exists is not None else Path(path).is_file()

    result: list[dict] = []
    for path, seen in by_path.items():
        if not exists(path):
            result.append({"change": Change.deleted.name, "path": path})
        elif seen == {Change.added}:
            result.append({"change": Change.added.name, "path": path})
        else:
            result.append({"change": Change.modified.name, "path": path})
    return result


def bucket_changes(changes: list[dict], path_exists: Callable[[str], bool] | None = None) -> dict[Change, list[str]]:
    """Group changes by watchfiles.Change."""
    buckets: dict[Change, list[str]] = {Change.added: [], Change.modified: [], Change.deleted: []}
    for item in coalesce_changes(changes, path_exists=path_exists):
        change = _normalize_change(item["change"])
        if isinstance(change, Change) and change in buckets:
            buckets[change].append(item["path"])
    return buckets
