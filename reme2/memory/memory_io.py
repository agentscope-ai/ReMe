"""Memory File System engine API — the core engine's outward surface.

The .md files are the SSOT (per `structure.md` §"核心引擎"). The engine
is layered:

    Memory File System  →  Watcher & Parser  →  Projections (vector / FTS / graph)
       (write entry)         (incremental)        (read entry, derived)

This module is the **single public API surface** over that engine. Every
consumer — MCP step shells, the three memory services (Retriever,
Ingestor, Maintainer), and the agent toolkit (`memory_toolkit`) — talks
to the engine through these functions, not by reaching into
`BaseFileStore` directly. That keeps `file_store` an implementation
detail (could be local sqlite, remote, etc.) and gives the layering one
place to evolve.

Naming convention:
    - Verb-first: `get_file`, `create_file`, `search_vector`.
    - `file_store` is always the first positional argument when the
      function needs the engine handle; remaining args are keyword-only.
    - Pure-disk writes (`update_body`, `update_meta`, `delete_file`,
      `archive_file`) don't take `file_store` — they hit the filesystem
      and the watcher picks them up. The asymmetry is honest.

Three sections:

    1. MFS Reads        — get_file / list_files / get_links / get_backlinks
                          / resolve_wikilink / iter_files / count_tokens.
                          Primary-key lookups against the file_store cache
                          (with disk fallthrough for the file body).
    2. MFS Writes       — create_file / update_body / update_meta
                          / rename_file / delete_file / archive_file.
                          The MFS write entry. All return (ok, payload).
    3. Projection Queries — search_vector / search_keyword / expand_neighbors
                            / extract_anchors / get_chunks / make_filter
                            / find_collisions. Read-only — projections
                            are derived by the Watcher; callers don't
                            write them directly.

Schema policy and the agent toolkit projection live one layer up in
`reme2/memory/memory_toolkit.py` — this file is policy-free and remains
the engine's outward API.

Lives in `reme2/memory/` (not `reme2/mcp/`) so memory services and the
MCP transport layer can both consume it without forming an import cycle
through the transport layer.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path

import frontmatter

from ..schema import ChunkFilter, FileChunk, FileMetadata
from ..utils.wikilink import WIKILINK_RE


# ===========================================================================
# Section 1 — MFS Reads
# ===========================================================================
#
# Primary-key lookups against the file_store's in-memory cache (file
# meta + edges + stem index). `get_file` falls through to disk for the
# body so callers see the latest text even if the watcher hasn't picked
# up a write yet.


async def get_file(
    file_store,
    path: str,
    *,
    include_chunks: bool = False,
) -> dict:
    """Read frontmatter + body for one path. Optionally include parsed chunks.

    On-disk frontmatter is the source of truth — the file_store cache may
    lag a write that hasn't been picked up by the watcher yet.
    """
    meta = file_store.get_file_meta(path)
    result: dict = {"path": path, "exists": False}
    if meta is not None:
        edges = file_store.get_edges(path)
        result.update({
            "exists": True,
            "metadata": meta.metadata,
            "link": [e.model_dump(exclude_none=True) for e in edges],
        })

    file_path = Path(path)
    if file_path.is_file():
        raw = file_path.read_text(encoding="utf-8")
        post = frontmatter.loads(raw)
        result["exists"] = True
        result["content"] = post.content
        result["metadata"] = dict(post.metadata)

    if include_chunks:
        chunks = await file_store.get_chunks(path)
        result["chunks"] = [c.model_dump(exclude_none=True) for c in chunks]
    return result


def list_files(
    file_store,
    *,
    path_prefix: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
    limit: int = 100,
) -> dict:
    """List indexed files filtered by frontmatter exact-match, tags, and prefix.

    Returns {items: [{path, metadata}], count}.
    """
    metadata_filter = metadata or {}
    tag_filter = tags or []
    items: list[dict] = []
    for path, meta in file_store.nodes.items():
        if path_prefix and not path.startswith(path_prefix):
            continue
        md = meta.metadata or {}
        if metadata_filter and any(md.get(k) != v for k, v in metadata_filter.items()):
            continue
        if tag_filter:
            file_tags = set(md.get("tags", []) or [])
            if not all(t in file_tags for t in tag_filter):
                continue
        items.append({"path": path, "metadata": md})
        if len(items) >= limit:
            break
    return {"items": items, "count": len(items)}


def _edge_to_dict(file_meta, edge) -> dict:
    return {
        "path": file_meta.path,
        "metadata": file_meta.metadata,
        "predicate": edge.predicate,
        "anchor": edge.anchor,
        "alias": edge.alias,
        "embed": edge.embed,
        "source": edge.source,
        "confidence": edge.confidence,
    }


def get_links(file_store, path: str) -> dict:
    """Files that `path` links TO (resolved). Each entry carries the typed-edge predicate."""
    return {
        "path": path,
        "links": [_edge_to_dict(m, e) for m, e in file_store.get_links(path)],
    }


def get_backlinks(file_store, path: str) -> dict:
    """Files that link TO `path`. Each entry carries the typed-edge predicate."""
    return {
        "path": path,
        "backlinks": [_edge_to_dict(m, e) for m, e in file_store.get_backlinks(path)],
    }


def resolve_wikilink(file_store, wikilink: str) -> dict:
    """Resolve a `[[target]]` wikilink with full ambiguity context.

    Distinct from `file_store.resolve_wikilink(target)` (which returns
    just the path or None) — this surface returns the rich payload
    callers need to disambiguate:

        unique resolution → {wikilink, path, exists: True,
                             ambiguous: False, candidates: [path]}
        ambiguous         → {wikilink, path: None, exists: False,
                             ambiguous: True, candidates: [...]}
        dangling          → {wikilink, path: None, exists: False,
                             ambiguous: False, candidates: []}
    """
    # Path-form (`a/b` or `a/b.md`): file_store already returns
    # exactly the one path that exists, or None.
    if "/" in wikilink or wikilink.endswith(".md"):
        hit = file_store.resolve_wikilink(wikilink)
        return {
            "wikilink": wikilink,
            "path": hit,
            "exists": hit is not None,
            "ambiguous": False,
            "candidates": [hit] if hit else [],
        }

    # Stem-form: candidates list reveals 0/1/N resolution.
    candidates = file_store.wikilink_candidates(wikilink)
    if len(candidates) == 1:
        return {
            "wikilink": wikilink,
            "path": candidates[0],
            "exists": True,
            "ambiguous": False,
            "candidates": candidates,
        }
    return {
        "wikilink": wikilink,
        "path": None,
        "exists": False,
        "ambiguous": len(candidates) > 1,
        "candidates": candidates,
    }


def iter_files(file_store) -> Iterator[tuple[str, FileMetadata]]:
    """Walk every indexed (path, FileMetadata). Used by Maintainer scans.

    Equivalent to `file_store.nodes.items()`, exposed here so consumers
    don't have to know about the underlying cache attribute name.
    """
    return iter(file_store.nodes.items())


async def count_tokens(
    token_counter,
    *,
    path: str | None = None,
    text: str | None = None,
) -> dict:
    """Estimate tokens for a file body (frontmatter excluded) or raw text.

    Powers the Maintainer's split-trigger. Exactly one of `path` / `text`
    must be provided. Takes a `token_counter` rather than `file_store`
    because the engine's tokenization is a separate component — this
    function lives here to keep the engine's outward surface in one place.
    """
    if path:
        target = Path(path)
        if not target.is_file():
            return {"path": str(target), "error": "file not found"}
        raw = target.read_text(encoding="utf-8")
        post = frontmatter.loads(raw)
        body = post.content
        tokens = await token_counter.count(messages=[], text=body)
        return {
            "source": "file",
            "path": str(target.resolve()),
            "tokens": tokens,
            "body_chars": len(body),
        }
    if text:
        tokens = await token_counter.count(messages=[], text=text)
        return {
            "source": "text",
            "tokens": tokens,
            "body_chars": len(text),
        }
    return {"error": "one of `path` or `text` is required"}


# ===========================================================================
# Section 2 — MFS Writes
# ===========================================================================
#
# Every mutation in the system funnels through these. Hot-write MCP shells
# (sync, memory_*) call them directly; cold-write services
# (Ingestor R-M-W, Maintainer decay) compose them.
#
# `create_file` and `rename_file` take `file_store` because they need the
# wikilink-uniqueness gate / backlinks index. The other writes don't —
# they just touch disk and let the Watcher catch up.


def _replace_wikilink_targets(text: str, mapping: dict[str, str]) -> str:
    """Rewrite wikilink targets in raw text.

    Only the `target` portion of `[[target]]` / `[[target#anchor]]` /
    `[[target|alias]]` / `![[target]]` is replaced; anchors, aliases,
    and embed prefixes are preserved.
    """
    if not mapping:
        return text

    def sub(m: re.Match) -> str:
        target_raw = m.group(1)
        target = target_raw.strip()
        if target in mapping:
            return m.group(0).replace(target_raw, mapping[target], 1)
        return m.group(0)

    return WIKILINK_RE.sub(sub, text)


def create_file(
    file_store,
    path: Path,
    *,
    metadata: dict,
    content: str,
    overwrite: bool = False,
    force: bool = False,
) -> tuple[bool, dict]:
    """Single L1 entry point for creating a markdown file.

    All file creation in the project must funnel through this so the
    wikilink uniqueness invariant is enforced in exactly one place.

    Refuses (returns (False, payload)) when:
        - file already exists (unless overwrite=True)
        - creating it would make `[[stem]]` resolve ambiguously against
          the current file_store (unless force=True)
    """
    if path.exists() and not overwrite:
        return False, {"path": str(path), "error": "file already exists"}

    if not force:
        conflicts = file_store.collisions_after_create(path)
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
    """Edit-style content update — replace `old_string` with `new_string`."""
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
    """Update a single YAML frontmatter key. value=None deletes the key."""
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


def rename_file(
    file_store,
    vault_root: Path | str,
    *,
    old_path: Path | str,
    new_path: Path | str,
) -> tuple[bool, dict]:
    """Rename a file and rewrite incoming wikilinks across the vault.

    Atomically moves `old_path` to `new_path`, then rewrites short-form
    `[[old_stem]]` and path-form `[[old_relative]]` wikilinks in every
    file that already had a *resolved* link to old_path. Look-up uses
    `file_store.get_backlinks(old_path)` so the work is O(K) where K is
    the number of incoming references — not a full vault scan.

    Refuses if:
        - `old_path` doesn't exist
        - `new_path` already exists
        - the rename would make `[[new_stem]]` resolve ambiguously
    """
    old_p = Path(old_path).resolve()
    new_p = Path(new_path).resolve()

    if not old_p.is_file():
        return False, {"old_path": str(old_p), "error": "old_path not found"}
    if new_p.exists():
        return False, {"new_path": str(new_p), "error": "new_path already exists"}
    if old_p == new_p:
        return False, {"error": "old_path and new_path are the same"}

    conflicts = file_store.collisions_after_create(new_p)
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

    vault_root_p = Path(vault_root).resolve()
    old_stem = old_p.stem
    new_stem = new_p.stem
    replacements: dict[str, str] = {}
    if old_stem != new_stem:
        replacements[old_stem] = new_stem
    try:
        old_rel = str(old_p.relative_to(vault_root_p).with_suffix(""))
        new_rel = str(new_p.relative_to(vault_root_p).with_suffix(""))
        if old_rel != new_rel:
            replacements[old_rel] = new_rel
            replacements[old_rel + ".md"] = new_rel + ".md"
    except ValueError:
        pass  # paths outside vault root — skip path-form rewrite

    referring_paths = [m.path for m, _ in file_store.get_backlinks(str(old_p))]

    new_p.parent.mkdir(parents=True, exist_ok=True)
    old_p.rename(new_p)

    updated_files: list[str] = []
    write_errors: list[dict] = []
    if replacements and referring_paths:
        for path in referring_paths:
            file_path = Path(path)
            if not file_path.is_file():
                continue
            try:
                raw = file_path.read_text(encoding="utf-8")
                new_raw = _replace_wikilink_targets(raw, replacements)
                if new_raw != raw:
                    file_path.write_text(new_raw, encoding="utf-8")
                    updated_files.append(path)
            except Exception as exc:
                write_errors.append({"path": path, "error": str(exc)})

    return True, {
        "old_path": str(old_p),
        "new_path": str(new_p),
        "stem_changed": old_stem != new_stem,
        "replacements": replacements,
        "referring_count": len(referring_paths),
        "updated_files": updated_files,
        "write_errors": write_errors,
    }


def delete_file(path: Path | str) -> tuple[bool, dict]:
    """Delete a file. Watcher removes from store + graph."""
    target = Path(path)
    if not target.exists():
        return False, {"path": str(target), "error": "not found"}
    target.unlink()
    return True, {"path": str(target), "deleted": True}


def archive_file(
    vault_root: Path | str,
    path: Path | str,
    *,
    archive_dir: str = "Archive",
) -> tuple[bool, dict]:
    """Archive a file: flip `status: archived`, then move under `<vault>/<archive_dir>/`.

    Composed by the Maintainer's decay pass when an event falls past its
    freshness window. Backlinks are *not* rewritten — dangling links to
    archived files are the intended audit trail.
    """
    src = Path(path).resolve()
    if not src.is_file():
        return False, {"path": str(src), "error": "file not found"}

    vault = Path(vault_root).resolve()
    try:
        rel = src.relative_to(vault)
    except ValueError:
        return False, {
            "path": str(src),
            "error": f"path is outside vault_root {vault}",
        }

    dst = vault / archive_dir / rel
    if dst.exists():
        return False, {
            "path": str(src),
            "error": f"archive destination already exists: {dst}",
        }

    ok, prop_payload = update_meta(src, key="status", value="archived")
    if not ok:
        return False, {**prop_payload, "stage": "update_meta"}

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(src), str(dst))
    except OSError as exc:
        return False, {
            "path": str(src),
            "error": f"move failed: {exc}",
            "stage": "move",
        }
    return True, {
        "old_path": str(src),
        "new_path": str(dst),
        "archived": True,
    }


# ===========================================================================
# Section 3 — Projection Queries
# ===========================================================================
#
# Per `structure.md` §"核心引擎", the Vector index, FTS5 index, and File
# Graph are downstream **projections** of the MFS — they can be wholly
# rebuilt from disk. These functions are the read entry to those
# projections; the Retriever composes them with policy (V+K weighting,
# graph BFS, intent routing) for ranked retrieval.
#
# Projections are write-only-via-Watcher: the API surface deliberately
# exposes no setters. To "update" a projection, write to the MFS (Section
# 2) and let the Watcher rebuild.


async def search_vector(
    file_store,
    query: str,
    *,
    limit: int,
    chunk_filter: ChunkFilter | None = None,
) -> list[FileChunk]:
    """Vector similarity over the chunk-level Vector projection."""
    return await file_store.vector_search(query, limit, chunk_filter)


async def search_keyword(
    file_store,
    query: str,
    *,
    limit: int,
    chunk_filter: ChunkFilter | None = None,
) -> list[FileChunk]:
    """FTS5 keyword search over the chunk-level keyword projection."""
    return await file_store.keyword_search(query, limit, chunk_filter)


def expand_neighbors(
    file_store,
    seeds: Iterable[str],
    *,
    depth: int = 1,
    direction: str = "both",
) -> dict[str, int]:
    """BFS over the File Graph projection. Returns {path: hop_distance}."""
    return file_store.expand_neighbors(seeds, depth=depth, direction=direction)


def extract_anchors(file_store, text: str) -> list[str]:
    """Pull anchor paths from `[[target]]` references inside `text`."""
    return file_store.extract_anchor_paths(text)


async def get_chunks(file_store, paths: Iterable[str]) -> list[FileChunk]:
    """Batch fetch chunks across many paths (used by graph-walk retrieval)."""
    return await file_store.get_chunks_by_paths(paths)


def make_filter(
    file_store,
    *,
    paths: list[str] | None = None,
    tags: list[str] | None = None,
    exclude_paths: list[str] | None = None,
) -> ChunkFilter | None:
    """Build a chunk-filter against the file_store's path/tag indexes."""
    return file_store.filter(paths=paths, tags=tags, exclude_paths=exclude_paths)


def find_collisions(file_store) -> dict[str, list[str]]:
    """Every stem that resolves to >1 path. Used by Maintainer.lint."""
    return file_store.all_ambiguous_wikilinks()

