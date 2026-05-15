"""Memory File System engine API — minimal surface for the agent toolkit.

The .md files are the SSOT. The engine layer is:

    Memory File System  →  Watcher & Parser  →  Projections (vector / FTS)
       (write entry)         (incremental)        (read entry, derived)

This module is the single public surface over that engine for the
agent toolkit. ``BaseFileGraph`` deliberately doesn't expose "scan
everything" (only ``get_nodes(paths)``); when we need to enumerate
(file_list, collisions check, filter resolution) we walk the
filesystem directly. The graph is consulted only for adjacency
lookups around a known path.

Public surface (everything else has been removed):

    Reads:    get_file, list_files
    Writes:   create_file, update_body, update_meta, delete_file,
              rename_file
    Filter:   make_filter

Pure-disk writes (``update_body`` / ``update_meta`` / ``delete_file``)
don't take ``file_store`` — they hit the filesystem and the watcher
picks them up. Engine-aware writes (``create_file`` / ``rename_file``)
take ``file_store`` because they consult the vault index for
collision/adjacency gates before writing.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import frontmatter

from ..component.file_store.base_file_store import BaseFileStore
from ..schema import ChunkFilter, FileNode
from ..utils.wikilink_resolver import _WIKILINK_RE


# ===========================================================================
# Internal helpers
# ===========================================================================


def _vault_root(file_store: BaseFileStore) -> Path | None:
    """Resolve the vault root from the file_store. ``working_dir`` is the
    convention; falls back to ``working_path`` if set on the component."""
    wd = getattr(file_store, "working_dir", None)
    if wd:
        return Path(wd).resolve()
    wp = getattr(file_store, "working_path", None)
    if wp:
        return Path(wp).resolve()
    return None


_SKIP_DIRS = {".git", ".obsidian", ".reme", ".reme2", "__pycache__", "Archive"}


def _walk_vault(
    file_store: BaseFileStore,
    *,
    suffix: str = ".md",
) -> Iterator[Path]:
    """Yield every file under the vault matching ``suffix``.

    Filesystem-as-SSOT: the .md files are authoritative for "what
    exists". Hidden directories and ``Archive/`` are skipped (Archive
    lives in the vault but isn't part of the active set).
    """
    root = _vault_root(file_store)
    if root is None or not root.is_dir():
        return
    for path in root.rglob(f"*{suffix}"):
        if any(part in _SKIP_DIRS for part in path.relative_to(root).parts[:-1]):
            continue
        if path.is_file():
            yield path.resolve()


def _walk_vault_meta(
    file_store: BaseFileStore,
    *,
    suffix: str = ".md",
) -> Iterator[tuple[str, dict]]:
    """Walk the vault, yielding ``(absolute_path_str, frontmatter_dict)``.

    Files that fail to parse get an empty dict — that's a lint concern,
    not the listing concern.
    """
    for abs_path in _walk_vault(file_store, suffix=suffix):
        try:
            raw = abs_path.read_text(encoding="utf-8")
            meta = dict(frontmatter.loads(raw).metadata)
        except Exception:
            meta = {}
        yield str(abs_path), meta


async def _get_node(file_store: BaseFileStore, path: str) -> FileNode | None:
    """Single-node fetch via the file_graph contract."""
    if not file_store.file_graph:
        return None
    nodes = await file_store.file_graph.get_nodes([path])
    return nodes[0] if nodes else None


def _replace_wikilink_targets(text: str, mapping: dict[str, str]) -> str:
    """Rewrite wikilink targets in raw text. Anchors / aliases / embed prefix kept."""
    if not mapping:
        return text

    def sub(m: re.Match) -> str:
        target_raw = m.group(1)
        target = target_raw.strip()
        if target in mapping:
            return m.group(0).replace(target_raw, mapping[target], 1)
        return m.group(0)

    return _WIKILINK_RE.sub(sub, text)


def collisions_after_create(
    file_store: BaseFileStore,
    proposed_path: str | Path,
) -> list[str]:
    """Existing paths that would conflict with adding `proposed_path`.

    Folder-note rule: if `proposed_path`'s parent dir name == its stem,
    only colliding folder-notes are returned (siblings with the same
    stem don't ambiguate). Otherwise both folder-notes AND stem hits
    are returned.

    Walks the filesystem since the engine contract has no "scan all
    paths" API on the graph. Public so other write paths (sync.py)
    can reuse the same gate.
    """
    p = Path(proposed_path)
    stem = p.stem
    proposed_abs = str(p.resolve())
    is_folder_note = p.parent.name == stem

    folder_hits: list[str] = []
    stem_hits: list[str] = []
    for abs_path in _walk_vault(file_store):
        path = str(abs_path)
        if path == proposed_abs:
            continue
        if abs_path.stem != stem:
            continue
        if abs_path.parent.name == stem:
            folder_hits.append(path)
        else:
            stem_hits.append(path)

    if is_folder_note:
        return folder_hits
    return folder_hits + stem_hits


# ===========================================================================
# Reads
# ===========================================================================


async def get_file(file_store: BaseFileStore, path: str) -> dict:
    """Read frontmatter + body from disk; attach links from the graph.

    On-disk frontmatter is the source of truth — the graph cache may
    lag a write that the watcher hasn't picked up yet. The graph is
    consulted only for the link list (adjacency).
    """
    result: dict = {"path": path, "exists": False}

    file_path = Path(path)
    if file_path.is_file():
        raw = file_path.read_text(encoding="utf-8")
        post = frontmatter.loads(raw)
        result["exists"] = True
        result["content"] = post.content
        result["metadata"] = dict(post.metadata)

    node = await _get_node(file_store, path)
    if node is not None:
        result["link"] = [link.model_dump(exclude_none=True) for link in node.links]
        if not result["exists"]:
            result["exists"] = True
            result.setdefault("metadata", node.front_matter.model_dump())
    else:
        result["link"] = []
    return result


def list_files(
    file_store: BaseFileStore,
    *,
    path_prefix: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
    limit: int = 100,
) -> dict:
    """List vault files filtered by frontmatter exact-match, tags, and prefix.

    Filesystem walk — the graph contract has no enumeration API.
    ``path_prefix`` matches the absolute path string.
    """
    metadata_filter = metadata or {}
    tag_filter = tags or []
    items: list[dict] = []
    for path, md in _walk_vault_meta(file_store):
        if path_prefix and not path.startswith(path_prefix):
            continue
        if metadata_filter and any(md.get(k) != v for k, v in metadata_filter.items()):
            continue
        if tag_filter:
            file_tags = set(md.get("tags") or [])
            if not all(t in file_tags for t in tag_filter):
                continue
        items.append({"path": path, "metadata": md})
        if len(items) >= limit:
            break
    return {"items": items, "count": len(items)}


# ===========================================================================
# Writes
# ===========================================================================


def create_file(
    file_store: BaseFileStore,
    path: Path,
    *,
    metadata: dict,
    content: str,
    overwrite: bool = False,
    force: bool = False,
) -> tuple[bool, dict]:
    """Create a markdown file. Refuses when:

    - file already exists (unless ``overwrite=True``)
    - creating it would make ``[[stem]]`` ambiguous (unless ``force=True``)
    """
    if path.exists() and not overwrite:
        return False, {"path": str(path), "error": "file already exists"}

    if not force:
        conflicts = collisions_after_create(file_store, path)
        if conflicts:
            return False, {
                "path": str(path),
                "error": (
                    f"stem `[[{path.stem}]]` would resolve ambiguously "
                    f"to {len(conflicts) + 1} paths after this create"
                ),
                "conflicts": conflicts,
                "hint": (
                    f"either rename to a unique stem, or have callers "
                    f"link via the explicit-path form "
                    f"`[[{path.parent.name}/{path.stem}]]`; pass "
                    f"force=true only if you accept the ambiguity"
                ),
            }

    path.parent.mkdir(parents=True, exist_ok=True)
    post = frontmatter.Post(content, **metadata)
    path.write_text(frontmatter.dumps(post), encoding="utf-8")
    return True, {"path": str(path), "created": True}


def update_body(
    path: Path | str,
    *,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> tuple[bool, dict]:
    """Edit-style content update — replace ``old_string`` with ``new_string``."""
    target = Path(path)
    if not target.is_file():
        return False, {"path": str(target), "error": "file not found"}
    if not old_string:
        return False, {
            "path": str(target),
            "error": "old_string is required (use create_file to write a new file)",
        }
    raw = target.read_text(encoding="utf-8")
    occurrences = raw.count(old_string)
    if occurrences == 0:
        return False, {"path": str(target), "error": "old_string not found in file"}
    if occurrences > 1 and not replace_all:
        return False, {
            "path": str(target),
            "error": f"old_string appears {occurrences} times; pass replace_all=true to replace all",
            "occurrences": occurrences,
        }
    if replace_all:
        new_raw = raw.replace(old_string, new_string)
    else:
        new_raw = raw.replace(old_string, new_string, 1)
    target.write_text(new_raw, encoding="utf-8")
    return True, {
        "path": str(target),
        "replaced": occurrences if replace_all else 1,
    }


def update_meta(path: Path | str, *, key: str, value) -> tuple[bool, dict]:
    """Update a single YAML frontmatter key. ``value=None`` deletes the key."""
    target = Path(path)
    if not target.is_file():
        return False, {"path": str(target), "error": "file not found"}
    raw = target.read_text(encoding="utf-8")
    post = frontmatter.loads(raw)
    if value is None:
        post.metadata.pop(key, None)
    else:
        post.metadata[key] = value
    target.write_text(frontmatter.dumps(post), encoding="utf-8")
    return True, {"path": str(target), "key": key, "value": value}


def delete_file(path: Path | str) -> tuple[bool, dict]:
    """Delete a file. Watcher removes from store + graph asynchronously."""
    target = Path(path)
    if not target.exists():
        return False, {"path": str(target), "error": "not found"}
    target.unlink()
    return True, {"path": str(target), "deleted": True}


def rename_file(
    file_store: BaseFileStore,
    working_dir: Path | str,
    *,
    old_path: Path | str,
    new_path: Path | str,
) -> tuple[bool, dict]:
    """Rename a file and rewrite incoming wikilinks across the vault.

    Walks the filesystem to find referencing files (the graph
    contract has no source-path lookup for inbound edges, and the
    rewrite is text-level anyway). Same-stem renames within the same
    folder are no-op for link rewrite.
    """
    old_p = Path(old_path).resolve()
    new_p = Path(new_path).resolve()

    if not old_p.is_file():
        return False, {"old_path": str(old_p), "error": "old_path not found"}
    if new_p.exists():
        return False, {"new_path": str(new_p), "error": "new_path already exists"}
    if old_p == new_p:
        return False, {"error": "old_path and new_path are the same"}

    conflicts = collisions_after_create(file_store, new_p)
    if conflicts:
        return False, {
            "error": (
                f"stem `[[{new_p.stem}]]` would resolve ambiguously "
                f"to {len(conflicts) + 1} paths after this rename"
            ),
            "conflicts": conflicts,
            "hint": (
                f"either rename to a unique stem (consider a "
                f"domain-specific suffix), or have callers link via "
                f"the explicit-path form `[[{new_p.parent.name}/{new_p.stem}]]`"
            ),
        }

    working_dir_p = Path(working_dir).resolve()
    old_stem = old_p.stem
    new_stem = new_p.stem
    replacements: dict[str, str] = {}
    if old_stem != new_stem:
        replacements[old_stem] = new_stem
    try:
        old_rel = str(old_p.relative_to(working_dir_p).with_suffix(""))
        new_rel = str(new_p.relative_to(working_dir_p).with_suffix(""))
        if old_rel != new_rel:
            replacements[old_rel] = new_rel
            replacements[old_rel + ".md"] = new_rel + ".md"
    except ValueError:
        pass

    new_p.parent.mkdir(parents=True, exist_ok=True)
    old_p.rename(new_p)

    updated_files: list[str] = []
    write_errors: list[dict] = []
    if replacements:
        for abs_path in _walk_vault(file_store):
            if abs_path == new_p:
                continue
            try:
                raw = abs_path.read_text(encoding="utf-8")
                new_raw = _replace_wikilink_targets(raw, replacements)
                if new_raw != raw:
                    abs_path.write_text(new_raw, encoding="utf-8")
                    updated_files.append(str(abs_path))
            except Exception as exc:
                write_errors.append({"path": str(abs_path), "error": str(exc)})

    return True, {
        "old_path": str(old_p),
        "new_path": str(new_p),
        "stem_changed": old_stem != new_stem,
        "replacements": replacements,
        "updated_files": updated_files,
        "write_errors": write_errors,
    }


# ===========================================================================
# Filter helper
# ===========================================================================


def make_filter(
    file_store: BaseFileStore,
    *,
    paths: list[str] | None = None,
    tags: list[str] | None = None,
    exclude_paths: list[str] | None = None,
) -> ChunkFilter | None:
    """Compile a ChunkFilter and resolve it against the vault.

    Walks the filesystem to determine which paths match the filter's
    metadata clauses (the engine contract has no path enumeration on
    the graph). Empty filters skip the walk.
    """
    cf = ChunkFilter(paths=paths, tags=tags, exclude_paths=exclude_paths)
    if cf.is_empty():
        return cf
    cf.resolved_paths = {
        p for p, md in _walk_vault_meta(file_store) if cf.match_metadata(p, md)
    }
    return cf
