"""Path resolver — single source of truth for vault path resolution.

Vault paths come in two forms:

    * **relative path** — the canonical primary key for each file
      (e.g. ``"topics/Alice/Alice.md"``). One path, one file.
    * **short path** — a simplification with no directory component
      (e.g. ``"Alice"``, ``"Alice.md"``). Convenient to type, but
      **may map to multiple relative paths** when more than one file
      shares the basename.

Two layers of resolution:

    Layer 1 — vault key (graph-backed)
        ``resolve(graph, path)`` — short path / relative path → the
        canonical vault-relative primary key. Raises ``PathAmbiguous``
        on short-link multi-match, ``PathNotFound`` on miss.

    Layer 2 — disk path (file_store-backed)
        ``to_absolute(file_store, relative_path)`` — vault-relative
        path → absolute on-disk ``Path``. Pure ``working_dir`` join.

    Combined — file access
        ``resolve_to_absolute(file_store, path)`` — Layer 1 + Layer 2.
        The standard entry point for any operation that **accesses an
        existing file** (read, update, delete). Short-link callers get
        ambiguity errors; relative-path callers get plain join.

For **file creation** callers must pass a vault-relative path
(use ``to_absolute`` directly) — short links are meaningless until a
file exists.

This module is the bottom of the path stack — it knows nothing about
``FileLink`` or text syntax. Wikilink/Link concerns live in
``link_parser``, which calls into here.

## Conventions applied here

    1. Implicit ``.md`` — a path whose last segment has no extension
       is completed to ``X.md``. ``image.png`` is left alone.
    2. Folder-note rule — when both ``X.md`` and ``X/X.md`` exist,
       the folder-note (``X/X.md``) wins as the canonical resolution
       for ``X``. Other basename collisions stay ambiguous.
"""

from __future__ import annotations

from pathlib import Path

from ..component.file_graph.base_file_graph import BaseFileGraph


# ===========================================================================
# Exceptions
# ===========================================================================


class PathError(Exception):
    """Base for path-resolution failures."""


class PathNotFound(PathError):
    """No node matches the given path."""

    def __init__(self, target: str):
        super().__init__(f"path not in vault: {target!r}")
        self.target = target


class PathAmbiguous(PathError):
    """A short path matches more than one relative path.

    Caller must qualify the path (add directory components) or pick
    one of ``self.candidates`` explicitly. Step callers should surface
    ``self.candidates`` to the user and abort the operation.
    """

    def __init__(self, target: str, candidates: list[str]):
        super().__init__(f"path {target!r} is ambiguous: {candidates}")
        self.target = target
        self.candidates = list(candidates)


# ===========================================================================
# Predicates
# ===========================================================================


def is_short_path(path: str) -> bool:
    """``True`` when ``path`` has no directory component (short form)."""
    return bool(path) and "/" not in path


# ===========================================================================
# Internal helpers
# ===========================================================================


def _complete(path: str) -> str:
    """Apply the implicit ``.md`` rule to ``path``."""
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


# ===========================================================================
# Layer 1 — graph-backed vault key resolution
# ===========================================================================


async def resolve(graph: BaseFileGraph, path: str) -> str:
    """Resolve ``path`` to a single vault-relative key. Raises on failure.

    Applies implicit ``.md`` completion before lookup. Dispatches by
    shape:
      * literal path (contains ``/``) → direct ``get_nodes`` lookup
      * short path (no ``/``)         → basename match + folder-note rule

    Raises:
        ``PathNotFound``   — no matching node.
        ``PathAmbiguous``  — short path with multiple matches.
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


# ===========================================================================
# Layer 2 — filesystem path composition
# ===========================================================================


def to_absolute(file_store, relative_path: str) -> Path:
    """Compose absolute on-disk ``Path`` from a vault-relative path.

    ``relative_path`` should be vault-relative; absolute inputs pass
    through unchanged (Python's ``Path / abs`` join). For paths that
    may be short links, route through ``resolve_to_absolute`` instead
    so ambiguity surfaces.
    """
    working_dir = getattr(file_store, "working_dir", None) or "."
    return (Path(working_dir) / relative_path).resolve()


# ===========================================================================
# Combined — file access
# ===========================================================================


async def resolve_to_absolute(file_store, path: str) -> Path:
    """User path → absolute on-disk ``Path`` (the file-access entry point).

    Resolves short links / relative paths via ``resolve`` (graph), then
    composes the absolute ``Path`` via ``to_absolute``. Any operation
    that touches an existing vault file should go through here so
    short-link ambiguity surfaces as ``PathAmbiguous`` (with candidates)
    rather than silently picking a wrong file.

    Raises:
        ``PathNotFound``   — no matching node.
        ``PathAmbiguous``  — short path with multiple matches.
    """
    graph = getattr(file_store, "file_graph", None)
    if graph is None:
        raise PathNotFound(path)
    relative = await resolve(graph, path)
    return to_absolute(file_store, relative)
