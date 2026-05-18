"""Path resolver — vault path resolution.

Vault paths come in two forms:

    * **relative path** — canonical primary key (e.g. ``"topics/Alice/Alice.md"``).
    * **short path** — no directory component (e.g. ``"Alice"``). May map
      to multiple relative paths when more than one file shares the basename.

Two layers:

    Layer 1 — vault key (graph-backed)
        ``resolve(graph, path)`` — short / relative → canonical
        vault-relative key. Raises ``PathAmbiguous`` on multi-match,
        ``PathNotFound`` on miss.

    Layer 2 — disk path (file_store-backed)
        ``to_absolute(file_store, relative_path)`` — vault-relative
        path → absolute on-disk ``Path``. Pure ``working_dir`` join.

    Combined
        ``resolve_to_absolute(file_store, path)`` — Layer 1 + Layer 2.
        Standard entry point for operations on **existing** files.

## Conventions

    1. Implicit ``.md`` — last segment without an extension is
       completed to ``X.md``. ``image.png`` is left alone.
    2. Folder-note rule — when both ``X.md`` and ``X/X.md`` exist,
       the folder-note (``X/X.md``) wins. Other basename collisions
       stay ambiguous.
"""

from __future__ import annotations

from pathlib import Path

from ..components.file_graph.base_file_graph import BaseFileGraph


class PathError(Exception):
    """Base for path-resolution failures."""


class PathNotFound(PathError):
    """No node matches the given path."""

    def __init__(self, target: str):
        super().__init__(f"path not in vault: {target!r}")
        self.target = target


class PathAmbiguous(PathError):
    """A short path matches more than one relative path."""

    def __init__(self, target: str, candidates: list[str]):
        super().__init__(f"path {target!r} is ambiguous: {candidates}")
        self.target = target
        self.candidates = list(candidates)


def is_short_path(path: str) -> bool:
    """``True`` when ``path`` has no directory component (short form)."""
    return bool(path) and "/" not in path


def _complete(path: str) -> str:
    """Apply the implicit ``.md`` rule."""
    if not path:
        return path
    last = path.rsplit("/", 1)[-1]
    return path if "." in last else path + ".md"


def _filter_folder_note(basename: str, paths: list[str]) -> list[str]:
    """Apply folder-note rule. Sorted for determinism."""
    if not paths:
        return []
    stem = Path(basename).stem
    folder_hits = sorted(p for p in paths if Path(p).parent.name == stem)
    return folder_hits or sorted(paths)


async def resolve(graph: BaseFileGraph, path: str) -> str:
    """Resolve ``path`` to a single vault-relative key.

    Applies implicit ``.md`` completion before lookup. Literal paths
    (containing ``/``) hit ``get_nodes`` directly; short paths match
    on basename then apply the folder-note rule.
    """
    if not path:
        raise PathNotFound(path)
    target = _complete(path)

    if not is_short_path(target):
        if await graph.get_nodes([target]):
            return target
        raise PathNotFound(target)

    matches = [n.path for n in await graph.get_nodes() if Path(n.path).name == target]
    candidates = _filter_folder_note(target, matches)
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise PathAmbiguous(target, candidates)
    raise PathNotFound(target)


def to_absolute(file_store, relative_path: str) -> Path:
    """Compose absolute on-disk ``Path`` from a vault-relative path.

    Absolute inputs pass through unchanged. For short-form inputs,
    route through ``resolve_to_absolute`` so ambiguity surfaces.
    """
    working_dir = getattr(file_store, "working_dir", None) or "."
    return (Path(working_dir) / relative_path).resolve()


async def resolve_to_absolute(file_store, path: str) -> Path:
    """User path → absolute on-disk ``Path``.

    Resolves short / relative paths via ``resolve``, then composes
    the absolute path via ``to_absolute``. Use this for any operation
    that touches an **existing** vault file so short-path ambiguity
    surfaces as ``PathAmbiguous`` instead of silently picking a wrong
    file.
    """
    graph = getattr(file_store, "file_graph", None)
    if graph is None:
        raise PathNotFound(path)
    relative = await resolve(graph, path)
    return to_absolute(file_store, relative)
