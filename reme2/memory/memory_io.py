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

Vault-convention helpers — wikilink resolution, graph-walk ranking,
chunk filtering, stem collisions — live here on top of the slim
`BaseFileStore` (which only manages graph + chunks). The engine itself
remains domain-agnostic.
"""

from __future__ import annotations

import re
import shutil
from collections import deque
from collections.abc import Iterable, Iterator
from pathlib import Path

import frontmatter

from ..schema import ChunkFilter, FileChunk, FileNode, extract_wikilinks
from ..schema.file_edge import WIKILINK_RE
from ..utils.wikilink_resolver import (
    resolve_wikilink as _resolve_wikilink,
    wikilink_candidates,
)


# ===========================================================================
# Section 1 — MFS Reads
# ===========================================================================


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
    node = file_store.get_node(path)
    result: dict = {"path": path, "exists": False}
    if node is not None:
        result.update({
            "exists": True,
            "metadata": node.metadata,
            "link": [e.model_dump(exclude_none=True) for e in node.edges],
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
    """List indexed files filtered by frontmatter exact-match, tags, and prefix."""
    metadata_filter = metadata or {}
    tag_filter = tags or []
    items: list[dict] = []
    for path, node in file_store.nodes.items():
        if path_prefix and not path.startswith(path_prefix):
            continue
        md = node.metadata or {}
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


def _edge_to_dict(node, edge) -> dict:
    return {
        "path": node.path,
        "metadata": node.metadata,
        "predicate": edge.predicate,
        "anchor": edge.anchor,
        "alias": edge.alias,
        "embed": edge.get_embeddings,
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

    Returns:
        unique resolution → {wikilink, path, exists: True,
                             ambiguous: False, candidates: [path]}
        ambiguous         → {wikilink, path: None, exists: False,
                             ambiguous: True, candidates: [...]}
        dangling          → {wikilink, path: None, exists: False,
                             ambiguous: False, candidates: []}
    """
    if "/" in wikilink or wikilink.endswith(".md"):
        hit = _resolve_wikilink(file_store, wikilink)
        return {
            "wikilink": wikilink,
            "path": hit,
            "exists": hit is not None,
            "ambiguous": False,
            "candidates": [hit] if hit else [],
        }

    candidates = wikilink_candidates(file_store, wikilink)
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


def iter_files(file_store) -> Iterator[tuple[str, FileNode]]:
    """Walk every indexed (path, FileNode). Used by Maintainer scans."""
    return iter(file_store.nodes.items())


async def count_tokens(
    token_counter,
    *,
    path: str | None = None,
    text: str | None = None,
) -> dict:
    """Estimate tokens for a file body (frontmatter excluded) or raw text."""
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

    Refuses when:
        - file already exists (unless overwrite=True)
        - creating it would make `[[stem]]` ambiguous (unless force=True)
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
    working_dir: Path | str,
    *,
    old_path: Path | str,
    new_path: Path | str,
) -> tuple[bool, dict]:
    """Rename a file and rewrite incoming wikilinks across the vault."""
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
    working_dir: Path | str,
    path: Path | str,
    *,
    archive_dir: str = "Archive",
) -> tuple[bool, dict]:
    """Archive a file: flip `status: archived`, then move under `<vault>/<archive_dir>/`."""
    src = Path(path).resolve()
    if not src.is_file():
        return False, {"path": str(src), "error": "file not found"}

    vault = Path(working_dir).resolve()
    try:
        rel = src.relative_to(vault)
    except ValueError:
        return False, {
            "path": str(src),
            "error": f"path is outside working_dir {vault}",
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


async def get_chunks(file_store, paths: Iterable[str]) -> list[FileChunk]:
    """Batch fetch chunks across many paths."""
    return await file_store.get_chunks_by_paths(paths)


# ===========================================================================
# Section 4 — Graph helpers (BFS / scoring / collisions)
# ===========================================================================
#
# Layered on top of the slim `BaseFileStore` (which only owns the
# graph + chunks). These are vault conventions: folder-note preference,
# wikilink-uniqueness gates, BFS for memory_graph_search.


def expand_neighbors(
    file_store,
    seeds: Iterable[str],
    *,
    depth: int = 1,
    direction: str = "both",
    per_node_cap: int | None = 50,
) -> dict[str, int]:
    """BFS over wikilink edges. Returns `{path: hop_distance}`."""
    if direction not in ("out", "in", "both"):
        raise ValueError(f"direction must be one of out/in/both, got {direction!r}")
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")

    seen: dict[str, int] = {}
    frontier: deque[tuple[str, int]] = deque()
    for path in seeds:
        if path in file_store and path not in seen:
            seen[path] = 0
            frontier.append((path, 0))

    while frontier:
        path, dist = frontier.popleft()
        if dist >= depth:
            continue

        neighbors: list[str] = []
        if direction in ("out", "both"):
            neighbors.extend(m.path for m, _ in file_store.get_links(path))
        if direction in ("in", "both"):
            neighbors.extend(m.path for m, _ in file_store.get_backlinks(path))

        if per_node_cap is not None and len(neighbors) > per_node_cap:
            neighbors = neighbors[:per_node_cap]

        for nb in neighbors:
            if nb in seen:
                continue
            seen[nb] = dist + 1
            frontier.append((nb, dist + 1))

    return seen


def subgraph_score(
    file_store,
    seeds: Iterable[str],
    *,
    decay: float = 0.5,
    depth: int = 1,
    direction: str = "both",
    per_node_cap: int | None = 50,
) -> dict[str, float]:
    """Decayed score per path: seed=1.0, 1-hop=decay, 2-hop=decay²..."""
    if not (0.0 <= decay <= 1.0):
        raise ValueError(f"decay must be in [0, 1], got {decay}")
    hops = expand_neighbors(
        file_store, seeds, depth=depth, direction=direction, per_node_cap=per_node_cap,
    )
    return {path: decay ** hop for path, hop in hops.items()}


def extract_anchors(file_store, text: str) -> list[str]:
    """Pull `[[X]]` anchors from `text` and resolve each (deduped)."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in extract_wikilinks(text):
        hit = _resolve_wikilink(file_store, raw)
        if hit is not None and hit not in seen:
            seen.add(hit)
            out.append(hit)
    return out


def collisions_after_create(file_store, proposed_path: str | Path) -> list[str]:
    """Existing paths that would conflict with adding `proposed_path`.

    Folder-note rule: if `proposed_path`'s parent dir name == its stem,
    only colliding folder-notes are returned (siblings with the same
    stem don't ambiguate). Otherwise both folder-notes AND stem hits
    are returned.
    """
    p = Path(proposed_path)
    stem = p.stem
    proposed_abs = str(p.resolve())
    is_folder_note = p.parent.name == stem

    folder_hits = [
        path for path in file_store.nodes
        if Path(path).stem == stem and Path(path).parent.name == stem
        and path != proposed_abs
    ]
    stem_hits = [
        path for path in file_store.get_paths_by_stem(stem)
        if path != proposed_abs
    ]

    if is_folder_note:
        return folder_hits
    return folder_hits + [sp for sp in stem_hits if sp not in folder_hits]


def make_filter(
    file_store,
    *,
    paths: list[str] | None = None,
    tags: list[str] | None = None,
    exclude_paths: list[str] | None = None,
) -> ChunkFilter | None:
    """Compile a ChunkFilter against the file_store's path/tag indexes."""
    cf = ChunkFilter(paths=paths, tags=tags, exclude_paths=exclude_paths)
    if cf.is_empty():
        return cf
    cf.resolved_paths = {
        p for p, n in file_store.nodes.items() if cf.match_metadata(p, n.metadata)
    }
    return cf


def find_collisions(file_store) -> dict[str, list[str]]:
    """Every stem that resolves to >1 path. Used by Maintainer.lint."""
    out: dict[str, list[str]] = {}
    for stem in file_store._stems:
        cands = wikilink_candidates(file_store, stem)
        if len(cands) > 1:
            out[stem] = cands
    return out
