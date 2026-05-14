"""Memory File System engine API — the core engine's outward surface.

The .md files are the SSOT (per `structure.md` §"核心引擎"). The engine
is layered:

    Memory File System  →  Watcher & Parser  →  Projections (vector / FTS)
       (write entry)         (incremental)        (read entry, derived)

This module is the **single public API surface** over that engine. Every
consumer — agent-facing steps, the three memory services (Retriever,
Ingestor, Maintainer), and the toolkit (`memory_toolkit`) — talks to
the engine through these functions, not by reaching into
``BaseFileStore`` directly.

Layering note. The slim ``BaseFileStore`` interface only owns the
search projections (vector / FTS) plus atomic file upserts/deletes.
Iteration, single-node lookup, and the wikilink graph live HERE — we
walk ``LocalFileStore._nodes`` directly (engine-layer peer) and compute
links on-the-fly from each ``FileNode.links``. There is no precomputed
graph index; the graph is a derivation of the SSoT.

Naming convention:
    - Verb-first: `get_file`, `create_file`, `search_vector`.
    - `file_store` is always the first positional argument when the
      function needs the engine handle; remaining args are keyword-only.
    - Pure-disk writes (`update_body`, `update_meta`, `delete_file`,
      `archive_file`) don't take `file_store` — they hit the filesystem
      and the watcher picks them up. The asymmetry is honest.
"""

from __future__ import annotations

import re
import shutil
from collections import deque
from collections.abc import Iterable, Iterator
from pathlib import Path

import frontmatter

from ..component.file_store.base_file_store import BaseFileStore
from ..schema import ChunkFilter, FileChunk, FileLink, FileNode, extract_wikilinks
from ..schema.file_link import _WIKILINK_RE
from ..utils.wikilink_resolver import (
    resolve_wikilink as _resolve_wikilink,
    wikilink_candidates,
)


# ===========================================================================
# Internal helpers — engine-layer access to the concrete node index
# ===========================================================================
#
# ``BaseFileStore`` only declares the search/upsert contract; iteration
# and single-node lookup live on the concrete impl (``LocalFileStore``).
# memory_io is a peer at the engine layer, so reaching into ``_nodes``
# is intentional — every iteration / graph walk funnels through here so
# the day the engine grows a public ``iter_nodes()``, this is the only
# place to swap.


def _nodes(file_store: BaseFileStore) -> dict[str, FileNode]:
    """The concrete in-memory ``{path: FileNode}`` index."""
    return file_store._nodes  # type: ignore[attr-defined]


def _meta(node: FileNode) -> dict:
    """Full frontmatter dict for a node — typed fields (title/description/
    tags) merged with any ``extra=allow`` extras."""
    return node.front_matter.model_dump()


def _get_outlinks(
    file_store: BaseFileStore,
    path: str,
) -> list[tuple[FileNode, FileLink]]:
    """Resolved outgoing links from ``path`` — ``[(target_node, link), ...]``."""
    node = _nodes(file_store).get(path)
    if node is None:
        return []
    out: list[tuple[FileNode, FileLink]] = []
    for link in node.links:
        target = _resolve_wikilink(file_store, link.path)
        if target is None:
            continue
        target_node = _nodes(file_store).get(target)
        if target_node is not None:
            out.append((target_node, link))
    return out


def _get_inlinks(
    file_store: BaseFileStore,
    path: str,
) -> list[tuple[FileNode, FileLink]]:
    """Resolved incoming links to ``path`` — linear scan over all nodes."""
    out: list[tuple[FileNode, FileLink]] = []
    for src_node in _nodes(file_store).values():
        for link in src_node.links:
            target = _resolve_wikilink(file_store, link.path)
            if target == path:
                out.append((src_node, link))
    return out


def _filter_to_dict(chunk_filter: ChunkFilter | None) -> dict:
    """Serialize ``ChunkFilter`` for the search engine's ``search_filter`` arg."""
    if chunk_filter is None:
        return {}
    return chunk_filter.model_dump(mode="json")


# ===========================================================================
# Section 1 — MFS Reads
# ===========================================================================


async def get_file(
    file_store: BaseFileStore,
    path: str,
    *,
    include_chunks: bool = False,
) -> dict:
    """Read frontmatter + body for one path. Optionally include parsed chunks.

    On-disk frontmatter is the source of truth — the file_store cache may
    lag a write that hasn't been picked up by the watcher yet.
    """
    node = await file_store.get_node_by_path(path)  # type: ignore[attr-defined]
    result: dict = {"path": path, "exists": False}
    if node is not None:
        result.update(
            {
                "exists": True,
                "metadata": _meta(node),
                "link": [link.model_dump(exclude_none=True) for link in node.links],
            }
        )

    file_path = Path(path)
    if file_path.is_file():
        raw = file_path.read_text(encoding="utf-8")
        post = frontmatter.loads(raw)
        result["exists"] = True
        result["content"] = post.content
        result["metadata"] = dict(post.metadata)

    if include_chunks:
        chunks = await file_store.get_chunks_by_path(path)  # type: ignore[attr-defined]
        result["chunks"] = [c.model_dump(exclude_none=True) for c in chunks]
    return result


def list_files(
    file_store: BaseFileStore,
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
    for path, node in _nodes(file_store).items():
        if path_prefix and not path.startswith(path_prefix):
            continue
        md = _meta(node)
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


def _link_to_dict(node: FileNode, link: FileLink) -> dict:
    return {
        "path": node.path,
        "metadata": _meta(node),
        "predicate": link.predicate,
        "anchor": link.anchor,
    }


def get_links(file_store: BaseFileStore, path: str) -> dict:
    """Files that `path` links TO (resolved). Each entry carries the typed-link predicate."""
    return {
        "path": path,
        "links": [_link_to_dict(m, link) for m, link in _get_outlinks(file_store, path)],
    }


def get_backlinks(file_store: BaseFileStore, path: str) -> dict:
    """Files that link TO `path`. Each entry carries the typed-link predicate."""
    return {
        "path": path,
        "backlinks": [_link_to_dict(m, link) for m, link in _get_inlinks(file_store, path)],
    }


def resolve_wikilink(file_store: BaseFileStore, wikilink: str) -> dict:
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


def iter_files(file_store: BaseFileStore) -> Iterator[tuple[str, FileNode]]:
    """Walk every indexed (path, FileNode). Used by Maintainer scans."""
    return iter(_nodes(file_store).items())


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

    return _WIKILINK_RE.sub(sub, text)


def create_file(
    file_store: BaseFileStore,
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
    file_store: BaseFileStore,
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
                f"stem `[[{new_p.stem}]]` would resolve ambiguously " f"to {len(conflicts) + 1} paths after this rename"
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

    referring_paths = [m.path for m, _ in _get_inlinks(file_store, str(old_p))]

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
    file_store: BaseFileStore,
    query: str,
    *,
    limit: int,
    chunk_filter: ChunkFilter | None = None,
) -> list[FileChunk]:
    """Vector similarity over the chunk-level Vector projection."""
    return await file_store.vector_search(query, limit, _filter_to_dict(chunk_filter))


async def search_keyword(
    file_store: BaseFileStore,
    query: str,
    *,
    limit: int,
    chunk_filter: ChunkFilter | None = None,
) -> list[FileChunk]:
    """FTS5 keyword search over the chunk-level keyword projection."""
    return await file_store.keyword_search(query, limit, _filter_to_dict(chunk_filter))


async def get_chunks(
    file_store: BaseFileStore,
    paths: Iterable[str],
) -> list[FileChunk]:
    """Batch fetch chunks across many paths."""
    out: list[FileChunk] = []
    for p in paths:
        out.extend(await file_store.get_chunks_by_path(p))  # type: ignore[attr-defined]
    return out


# ===========================================================================
# Section 4 — Graph helpers (BFS / scoring / collisions)
# ===========================================================================
#
# All graph traversal walks ``FileNode.links`` directly via ``_get_outlinks``
# / ``_get_inlinks``. There's no precomputed adjacency index — every BFS
# resolves wikilinks per call. Cheap enough for typical vault sizes;
# revisit if the maintainer's lint pass becomes a hotspot.


def expand_neighbors(
    file_store: BaseFileStore,
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

    nodes = _nodes(file_store)
    seen: dict[str, int] = {}
    frontier: deque[tuple[str, int]] = deque()
    for path in seeds:
        if path in nodes and path not in seen:
            seen[path] = 0
            frontier.append((path, 0))

    while frontier:
        path, dist = frontier.popleft()
        if dist >= depth:
            continue

        neighbors: list[str] = []
        if direction in ("out", "both"):
            neighbors.extend(m.path for m, _ in _get_outlinks(file_store, path))
        if direction in ("in", "both"):
            neighbors.extend(m.path for m, _ in _get_inlinks(file_store, path))

        if per_node_cap is not None and len(neighbors) > per_node_cap:
            neighbors = neighbors[:per_node_cap]

        for nb in neighbors:
            if nb in seen:
                continue
            seen[nb] = dist + 1
            frontier.append((nb, dist + 1))

    return seen


def subgraph_score(
    file_store: BaseFileStore,
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
        file_store,
        seeds,
        depth=depth,
        direction=direction,
        per_node_cap=per_node_cap,
    )
    return {path: decay**hop for path, hop in hops.items()}


def extract_anchors(file_store: BaseFileStore, text: str) -> list[str]:
    """Pull `[[X]]` anchors from `text` and resolve each (deduped)."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in extract_wikilinks(text):
        hit = _resolve_wikilink(file_store, raw)
        if hit is not None and hit not in seen:
            seen.add(hit)
            out.append(hit)
    return out


def collisions_after_create(
    file_store: BaseFileStore,
    proposed_path: str | Path,
) -> list[str]:
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

    folder_hits: list[str] = []
    stem_hits: list[str] = []
    for path in _nodes(file_store):
        if path == proposed_abs:
            continue
        path_obj = Path(path)
        if path_obj.stem != stem:
            continue
        if path_obj.parent.name == stem:
            folder_hits.append(path)
        else:
            stem_hits.append(path)

    if is_folder_note:
        return folder_hits
    return folder_hits + stem_hits


def make_filter(
    file_store: BaseFileStore,
    *,
    paths: list[str] | None = None,
    tags: list[str] | None = None,
    exclude_paths: list[str] | None = None,
) -> ChunkFilter | None:
    """Compile a ChunkFilter against the file_store's path/tag indexes."""
    cf = ChunkFilter(paths=paths, tags=tags, exclude_paths=exclude_paths)
    if cf.is_empty():
        return cf
    cf.resolved_paths = {p for p, n in _nodes(file_store).items() if cf.match_metadata(p, _meta(n))}
    return cf


def find_collisions(file_store: BaseFileStore) -> dict[str, list[str]]:
    """Every stem that resolves to >1 path. Used by Maintainer.lint."""
    by_stem: dict[str, list[str]] = {}
    for path in _nodes(file_store):
        by_stem.setdefault(Path(path).stem, []).append(path)
    return {s: ps for s, ps in by_stem.items() if len(ps) > 1}
