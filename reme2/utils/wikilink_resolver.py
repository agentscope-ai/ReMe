"""Wikilink resolver — vault convention over ``BaseFileGraph``.

Stem-form wikilinks like ``[[Foo]]`` are a vault convention. The
file_graph engine doesn't know about them — it only stores nodes and
trusts ``FileLink.path`` for adjacency. This module bridges the gap,
applying:

* **folder-note rule** — ``[[X]]`` prefers ``topics/X/X.md`` over a
  sibling ``topics/X.md`` when both exist
* **stem ambiguity** — ``[[Foo]]`` matching multiple paths is handled
  by **emitting multiple ``FileLink`` records, one per candidate**
  (file_graph then has unambiguous adjacency)
* **path-form passthrough** — ``[[foo/bar]]`` and ``[[foo/bar.md]]``
  resolve directly against the graph (vault-relative paths; no
  working_dir absolutization)

All paths are vault-relative — ``graph.iter_nodes()`` returns the
key form file_graph stores, and emitted ``FileLink.path`` matches.

All functions are stateless — they walk ``graph.iter_nodes()`` per
call. For batch operations (``resolve_links`` over many links) the
stem index is built once and reused.

Seven entry points mapped to call sites:

    resolve            single link → path | None
                       used by: ``extract_anchors``, ``memory_resolve_wikilink``
    candidates         stem → [path] (folder-note ordered first)
                       used by: ``memory_resolve_wikilink`` (ambiguity report)
    collisions         all stems with >1 path
                       used by: ``maintainer.lint``
    collisions_for     paths conflicting with a proposed new path
                       used by: ``memory_create``, ``sync`` (preflight)
    extract_anchors    parse [[X]] from text + resolve, dedup
                       used by: ``retriever`` (query anchor seeds)
    resolve_links      ``[FileLink]`` (raw path) → ``[FileLink]`` (resolved)
                       core resolution; stem ambiguity expands
    text_to_links      one-shot: ``iter_links(text)`` + ``resolve_links``
                       used by: parser pipeline (before ``upsert_node``)
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..component.file_graph.base_file_graph import BaseFileGraph
from ..schema import FileLink
from ..schema.file_link import extract_wikilinks, iter_links
from .logger_utils import get_logger


_logger = get_logger()


# =========================================================================
# Internal helpers
# =========================================================================


async def _build_stem_index(graph: BaseFileGraph) -> dict[str, list[str]]:
    """Walk once, group paths by stem. Used by batch hot paths."""
    out: dict[str, list[str]] = {}
    async for path, _ in graph.iter_nodes():
        out.setdefault(Path(path).stem, []).append(path)
    return out


def _split_link(link: str) -> tuple[str, str]:
    """Return ``(target, anchor)``. Anchor empty if no ``#``.

    For raw link strings supplied by external callers (which may still
    arrive in ``[[A#section]]`` form). Pre-extracted ``FileLink``
    records already carry ``path`` and ``anchor`` separately.
    """
    if not link:
        return "", ""
    if "#" not in link:
        return link.strip(), ""
    target_raw, anchor_raw = link.split("#", 1)
    return target_raw.strip(), anchor_raw.strip()


def _is_path_form(target: str) -> bool:
    return "/" in target or target.endswith(".md")


def _filter_stem_candidates(stem: str, paths: list[str]) -> list[str]:
    """Apply folder-note rule: if any folder-note exists, ONLY folder-notes
    are candidates; otherwise all sibling-stems are candidates."""
    if not paths:
        return []
    folder_hits = sorted(p for p in paths if Path(p).parent.name == stem)
    if folder_hits:
        return folder_hits
    return sorted(paths)


# =========================================================================
# Public API — single-shot lookups
# =========================================================================


async def resolve(graph: BaseFileGraph, link: str) -> str | None:
    """Resolve a single wikilink to **one** vault-relative path, or None.

    Returns None if dangling or ambiguous (with warning on ambiguity).
    Use ``resolve_links`` for the multi-link expansion semantics.
    """
    target, _ = _split_link(link)
    if not target:
        return None

    if _is_path_form(target):
        candidate = target if target.endswith(".md") else f"{target}.md"
        return candidate if await graph.get_node(candidate) else None

    paths = [p async for p, _ in graph.iter_nodes() if Path(p).stem == target]
    candidates_for_stem = _filter_stem_candidates(target, paths)
    if len(candidates_for_stem) == 1:
        return candidates_for_stem[0]
    if len(candidates_for_stem) > 1:
        _logger.warning(
            f"Wikilink [[{target}]] is ambiguous, " f"candidates: {candidates_for_stem}",
        )
    return None


async def candidates(graph: BaseFileGraph, stem: str) -> list[str]:
    """All paths a ``[[stem]]`` could match. Folder-note hits ordered first."""
    folder_hits: list[str] = []
    stem_hits: list[str] = []
    async for path, _ in graph.iter_nodes():
        p = Path(path)
        if p.stem != stem:
            continue
        if p.parent.name == stem:
            folder_hits.append(path)
        else:
            stem_hits.append(path)
    if folder_hits:
        return sorted(folder_hits)
    return sorted(stem_hits)


async def collisions(graph: BaseFileGraph) -> dict[str, list[str]]:
    """Every stem that resolves to >1 path. Used by maintainer.lint."""
    stem_index = await _build_stem_index(graph)
    return {stem: sorted(paths) for stem, paths in stem_index.items() if len(paths) > 1}


async def collisions_for(
    graph: BaseFileGraph,
    proposed_path: str | Path,
) -> list[str]:
    """Existing paths that would conflict with adding ``proposed_path``.

    ``proposed_path`` is treated as vault-relative (matching the form
    ``graph.iter_nodes()`` returns). Folder-note rule: when the
    proposed path is itself a folder-note (parent dir name == stem),
    only colliding folder-notes are returned. Otherwise all paths
    sharing the stem are returned.
    """
    p = Path(proposed_path)
    stem = p.stem
    proposed_str = str(p)
    is_folder_note = p.parent.name == stem

    folder_hits: list[str] = []
    stem_hits: list[str] = []
    async for path, _ in graph.iter_nodes():
        if path == proposed_str:
            continue
        path_obj = Path(path)
        if path_obj.stem != stem:
            continue
        if path_obj.parent.name == stem:
            folder_hits.append(path)
        else:
            stem_hits.append(path)

    if is_folder_note:
        return sorted(folder_hits)
    return sorted(folder_hits) + sorted(stem_hits)


async def extract_anchors(graph: BaseFileGraph, text: str) -> list[str]:
    """Pull ``[[X]]`` from ``text``, resolve each, dedup in source order.

    Uses single-target ``resolve`` semantics — ambiguous stems return
    no anchor. (For multi-link expansion at *write* time, see
    ``resolve_links``; ``extract_anchors`` is for read-time seeding
    where a single deterministic target is wanted.)
    """
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for raw in extract_wikilinks(text):
        hit = await resolve(graph, raw)
        if hit is not None and hit not in seen:
            seen.add(hit)
            out.append(hit)
    return out


# =========================================================================
# Public API — parser pipeline
# =========================================================================


async def resolve_links(
    graph: BaseFileGraph,
    links: Iterable[FileLink],
) -> list[FileLink]:
    """Resolve pre-resolution ``FileLink`` records against the graph.

    Each input link has ``path`` holding a raw wikilink target (the
    extractor form). Returns FileLinks with ``path`` rewritten to the
    vault-relative resolved path file_graph stores. ``anchor`` and
    ``predicate`` pass through unchanged.

    Stem expansion semantics — one input link produces zero, one, or
    many output links:

        * path-form, target indexed              → 1 link
        * path-form, dangling                    → 0 links
        * stem-form, 1 folder-note hit           → 1 link
        * stem-form, N folder-note hits          → N links (one per)
        * stem-form, 0 folder-notes, 1 sibling   → 1 link
        * stem-form, 0 folder-notes, N siblings  → N links (one per)
        * stem-form, dangling                    → 0 links

    Build the stem index lazily: only paid if at least one input is
    stem-form.
    """
    link_list = list(links)
    if not link_list:
        return []

    stem_index: dict[str, list[str]] | None = None
    out: list[FileLink] = []
    for link in link_list:
        target = link.path
        if not target:
            continue

        if _is_path_form(target):
            candidate = target if target.endswith(".md") else f"{target}.md"
            if await graph.get_node(candidate) is None:
                continue
            out.append(
                FileLink(
                    path=candidate,
                    anchor=link.anchor,
                    predicate=link.predicate,
                ),
            )
            continue

        # Stem form — may expand into multiple links.
        if stem_index is None:
            stem_index = await _build_stem_index(graph)
        for chosen in _filter_stem_candidates(target, stem_index.get(target, [])):
            out.append(
                FileLink(
                    path=chosen,
                    anchor=link.anchor,
                    predicate=link.predicate,
                ),
            )

    return out


async def text_to_links(
    graph: BaseFileGraph,
    text: str,
) -> list[FileLink]:
    """One-shot: extract wikilinks from ``text`` and resolve to safe links.

    Equivalent to ``await resolve_links(graph, iter_links(text))``.
    The parser pipeline's single-call entry point.
    """
    return await resolve_links(graph, iter_links(text))
