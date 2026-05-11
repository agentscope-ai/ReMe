"""Wikilink resolution — pure functions over a `BaseFileStore`'s graph.

Wikilink resolution is a vault convention (path/stem forms, folder-note
preference) layered on top of the engine's plain `dict[path, FileNode]`.
Keeping it out of `BaseFileStore` keeps the engine domain-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reme2.component.file_store.base_file_store import BaseFileStore


def wikilink_candidates(store: BaseFileStore, target: str) -> list[str]:
    """Paths `[[target]]` would resolve to (folder-note hit wins).

    Folder-note convention: `topics/X/X.md` is the cluster head for
    bare `[[X]]`, beating any sibling that just shares the stem.
    """
    folder_hits = sorted(
        p for p in store._stems.get(target, ()) if Path(p).parent.name == target
    )
    if folder_hits:
        return folder_hits
    return store.get_paths_by_stem(target)


def resolve_wikilink(store: BaseFileStore, wikilink: str) -> str | None:
    """Resolve a `[[target]]` to one absolute path.

    - Path form (`topics/X/X` or `topics/X/X.md`) → relative to `working_dir`.
    - Stem form (`X`) → folder-note preference, else unique stem hit.
    - Ambiguous stems return None and log a warning.
    """
    target = wikilink.strip()
    if not target:
        return None

    if "/" in target or target.endswith(".md"):
        if store.working_dir is None:
            return None
        candidate = target if target.endswith(".md") else f"{target}.md"
        abs_candidate = str((store.working_dir / candidate).resolve())
        return abs_candidate if abs_candidate in store.nodes else None

    candidates = wikilink_candidates(store, target)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        store.logger.warning(
            f"Wikilink [[{target}]] is ambiguous, candidates: {candidates}",
        )
    return None
