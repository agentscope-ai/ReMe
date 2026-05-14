"""Wikilink syntax + resolver — vault convention over ``BaseFileGraph``.

This module is the single home for wikilink **syntax** (regex
extraction, predicate detection) and wikilink **resolution** (mapping
raw targets to vault-relative paths via the file_graph).

The ``FileLink`` schema (see ``schema.file_link``) is just a typed
record; here we define how it's produced from text and resolved
against the graph.

## Two-layer link semantics

    1. Implicit extension — ``[[Foo]]`` has no extension on its last
       segment, so it's completed to ``Foo.md`` (markdown is the
       default vault content type). ``[[image.png]]`` already has an
       extension; left as-is. Done in ``iter_links`` at extraction
       time, so all FileLinks emerge with extension-bearing paths.

    2. Short link — a path with no ``/`` (after implicit completion)
       is matched against the basename of every node in the graph.
       The folder-note rule applies: when both ``X.md`` and
       ``X/X.md`` exist, the folder-note wins. Ambiguity expands into
       multiple FileLink records (one per candidate path). Targets
       containing ``/`` are treated as literal paths and looked up
       directly.

So ``[[Foo]]`` → ``Foo.md`` (implicit) → search basename ``Foo.md``
across the vault, returning e.g. ``topics/Foo/Foo.md``;
``[[topics/Bar]]`` → ``topics/Bar.md`` (implicit) → literal lookup;
``[[image.png]]`` → ``image.png`` (no completion needed) → search
basename ``image.png``; ``[[topics/image.png]]`` → literal lookup.

All paths are vault-relative — ``graph.iter_nodes()`` returns the
key form file_graph stores, and emitted ``FileLink.path`` matches.

All resolver functions are stateless — they walk ``graph.iter_nodes()``
per call. For batch operations (``resolve_links``) the basename index
is built once and reused.

## Inline forms recognised by ``iter_links``

    [[X]]                           bare wikilink         → predicate=None
    extends:: [[X]]                 line-level Dataview   → predicate="extends"
    [extends:: [[X]]]               inline-bracketed      → predicate="extends"

Multi-target — every wikilink under one typed context inherits its
predicate (any separator works, not just commas):

    extends:: [[A]], [[B]]          line-level multi      → 2 links, both "extends"
    extends:: [[A]] and [[B]]       prose-style multi     → 2 links, both "extends"
    [concerns:: [[A]], [[B]]]       inline multi          → 2 links, both "concerns"
    extends:: [[A#s1]], [[B#s2]]    multi w/ anchors      → anchors preserved per link

Context precedence is **inline-bracketed > line-level > bare** —
a wikilink inside a ``[predicate:: …]`` envelope is typed by that
envelope even if the line happens to start ``predicate:: …``.

## Entry points

    Extraction (no graph)
        iter_links         text → [FileLink] (path = target after implicit .md)
        extract_wikilinks  text → [str]      (raw target list, no completion)

    Resolution (graph-backed)
        resolve            single link → path | None
                           used by: ``extract_anchors``, ``memory_resolve_wikilink``
        candidates         target → [path] (folder-note ordered first)
                           used by: ``memory_resolve_wikilink`` (ambiguity report)
        collisions         all basenames with >1 path
                           used by: ``maintainer.lint``
        collisions_for     paths conflicting with a proposed new path
                           used by: ``memory_create``, ``sync`` (preflight)
        extract_anchors    parse [[X]] from text + resolve, dedup
                           used by: ``retriever`` (query anchor seeds)
        resolve_links      ``[FileLink]`` (raw path) → ``[FileLink]`` (resolved)
                           core resolution; short-link ambiguity expands
        text_to_links      one-shot: ``iter_links(text)`` + ``resolve_links``
                           used by: parser pipeline (before ``upsert_node``)
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from ..component.file_graph.base_file_graph import BaseFileGraph
from ..schema import FileLink
from .logger_utils import get_logger


_logger = get_logger()


# =========================================================================
# Wikilink syntax — regex extraction + predicate detection
# =========================================================================


_WIKILINK_RE = re.compile(
    r"""
    (?:!)?
    \[\[
        (?P<target>[^\]\|\#\n]+?)
        (?:\#(?P<anchor>[^\]\|\n]+))?
        (?:\|[^\]\n]+)?
    \]\]
    """,
    re.VERBOSE,
)

_DATAVIEW_LINE_RE = re.compile(
    r"^[ \t]*(?:[-*+][ \t]+)?(?P<predicate>[A-Za-z][A-Za-z0-9_]*)\s*::\s*(?P<value>.+?)\s*$",
    re.MULTILINE,
)

_INLINE_FIELD_OPEN_RE = re.compile(r"\[(?P<predicate>[A-Za-z][A-Za-z0-9_]*)\s*::\s*")


def _iter_inline_fields(text: str) -> list[tuple[int, int, str]]:
    """Find inline-bracketed ``[predicate:: …]`` field spans by depth scan."""
    out: list[tuple[int, int, str]] = []
    for m in _INLINE_FIELD_OPEN_RE.finditer(text):
        depth = 1
        i = m.end()
        n = len(text)
        while i < n:
            c = text[i]
            if c == "\n":
                break
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    out.append((m.start(), i + 1, m.group("predicate")))
                    break
            i += 1
    return out


def _predicate_for(
    text: str,
    pos: int,
    inline_spans: list[tuple[int, int, str]],
) -> str | None:
    """Resolve the predicate governing a wikilink at offset ``pos``."""
    for field_start, field_end, predicate in inline_spans:
        if field_start <= pos < field_end:
            return predicate
    line_start = text.rfind("\n", 0, pos) + 1
    line_end = text.find("\n", pos)
    if line_end == -1:
        line_end = len(text)
    m = _DATAVIEW_LINE_RE.match(text[line_start:line_end])
    if m and line_start + m.start("value") <= pos:
        return m.group("predicate")
    return None


def _complete_path(target: str) -> str:
    """Append ``.md`` if the last path segment has no extension.

    Implements the implicit-markdown rule: ``[[Foo]]`` → ``Foo.md``,
    ``[[topics/Bar]]`` → ``topics/Bar.md``, but ``[[image.png]]`` is
    left alone. ``"."`` in the last segment counts as "has extension".
    """
    if not target:
        return target
    last = target.rsplit("/", 1)[-1]
    if "." in last:
        return target
    return target + ".md"


def iter_links(text: str) -> list[FileLink]:
    """Extract every wikilink in ``text`` as a pre-resolution ``FileLink``.

    Each emitted ``FileLink`` has ``path`` set to the wikilink target
    with the implicit ``.md`` rule applied (so ``[[Foo]]`` emerges as
    ``path="Foo.md"``); the ``#anchor`` lives in its own field.
    Predicate is decided by surrounding context with precedence
    **inline-bracketed > line-level > bare**.

    Pure function: no graph access. Pass the result through
    ``resolve_links`` (or one-shot ``text_to_links``) to apply
    short-link resolution and get the form file_graph stores.
    """
    if not text:
        return []
    inline_spans = _iter_inline_fields(text)
    links: list[FileLink] = []
    for wm in _WIKILINK_RE.finditer(text):
        target = wm.group("target").strip()
        anchor_raw = wm.group("anchor")
        anchor = anchor_raw.strip() if anchor_raw else ""
        links.append(FileLink(
            path=_complete_path(target),
            anchor=anchor or None,
            predicate=_predicate_for(text, wm.start(), inline_spans),
        ))
    return links


def extract_wikilinks(text: str) -> list[str]:
    """Flat list of wikilink **file targets** in body text (no dedup).

    Returns just the file part of each wikilink **as written** —
    no implicit ``.md`` completion (callers like ``resolve`` apply
    completion themselves). Single regex pass — cheaper than
    ``iter_links`` when callers don't need predicates.
    """
    if not text:
        return []
    return [m.group("target").strip() for m in _WIKILINK_RE.finditer(text)]


# =========================================================================
# Resolution helpers (internal)
# =========================================================================


async def _build_basename_index(graph: BaseFileGraph) -> dict[str, list[str]]:
    """Walk once, group paths by basename (file name with extension).

    Used by short-link resolution batch hot paths.
    """
    out: dict[str, list[str]] = {}
    async for path, _ in graph.iter_nodes():
        out.setdefault(Path(path).name, []).append(path)
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


def _has_dir(target: str) -> bool:
    """``True`` for literal-path targets (containing ``/``);
    ``False`` for short links (basename only).
    """
    return "/" in target


def _filter_short_link_candidates(basename: str, paths: list[str]) -> list[str]:
    """Apply folder-note rule to short-link basename matches.

    If any path is a folder-note (parent dir name == file stem) the
    folder-notes win as the only candidates; otherwise all matches
    are returned. Sorted for determinism.
    """
    if not paths:
        return []
    stem = Path(basename).stem
    folder_hits = sorted(p for p in paths if Path(p).parent.name == stem)
    if folder_hits:
        return folder_hits
    return sorted(paths)


# =========================================================================
# Public API — single-shot lookups
# =========================================================================


async def resolve(graph: BaseFileGraph, link: str) -> str | None:
    """Resolve a single wikilink to **one** vault-relative path, or None.

    Applies implicit ``.md`` completion, then dispatches:
      * literal path (contains ``/``) → direct ``get_node`` lookup
      * short link (no ``/``)         → basename match + folder-note rule

    Returns None if dangling or ambiguous (with warning on ambiguity).
    Use ``resolve_links`` for the multi-link expansion semantics.
    """
    target, _ = _split_link(link)
    if not target:
        return None
    target = _complete_path(target)

    if _has_dir(target):
        return target if await graph.get_node(target) else None

    paths = [
        p async for p, _ in graph.iter_nodes() if Path(p).name == target
    ]
    candidates_for = _filter_short_link_candidates(target, paths)
    if len(candidates_for) == 1:
        return candidates_for[0]
    if len(candidates_for) > 1:
        _logger.warning(
            f"Wikilink [[{target}]] is ambiguous, "
            f"candidates: {candidates_for}",
        )
    return None


async def candidates(graph: BaseFileGraph, target: str) -> list[str]:
    """All vault paths a ``[[target]]`` could match.

    Applies implicit ``.md`` completion. For literal paths (with ``/``)
    returns ``[target]`` if it exists else ``[]``. For short links
    returns every node whose basename matches, with folder-note hits
    ordered first.
    """
    target = _complete_path(target)
    if _has_dir(target):
        return [target] if await graph.get_node(target) else []

    stem = Path(target).stem
    folder_hits: list[str] = []
    name_hits: list[str] = []
    async for path, _ in graph.iter_nodes():
        if Path(path).name != target:
            continue
        if Path(path).parent.name == stem:
            folder_hits.append(path)
        else:
            name_hits.append(path)
    if folder_hits:
        return sorted(folder_hits)
    return sorted(name_hits)


async def collisions(graph: BaseFileGraph) -> dict[str, list[str]]:
    """Every basename that resolves to >1 path. Used by maintainer.lint.

    Reflects short-link ambiguity: ``[[X.md]]`` (or ``[[X]]``) hitting
    multiple files in different directories.
    """
    basename_index = await _build_basename_index(graph)
    return {
        name: sorted(paths)
        for name, paths in basename_index.items()
        if len(paths) > 1
    }


async def collisions_for(
    graph: BaseFileGraph, proposed_path: str | Path,
) -> list[str]:
    """Existing paths that would conflict with adding ``proposed_path``.

    ``proposed_path`` is vault-relative (matches ``graph.iter_nodes()``).
    Folder-note rule: when the proposed path is itself a folder-note
    (parent dir name == file stem), only colliding folder-notes are
    returned. Otherwise all paths sharing the basename are returned.
    """
    p = Path(proposed_path)
    name = p.name
    stem = p.stem
    proposed_str = str(p)
    is_folder_note = p.parent.name == stem

    folder_hits: list[str] = []
    name_hits: list[str] = []
    async for path, _ in graph.iter_nodes():
        if path == proposed_str:
            continue
        path_obj = Path(path)
        if path_obj.name != name:
            continue
        if path_obj.parent.name == stem:
            folder_hits.append(path)
        else:
            name_hits.append(path)

    if is_folder_note:
        return sorted(folder_hits)
    return sorted(folder_hits) + sorted(name_hits)


async def extract_anchors(graph: BaseFileGraph, text: str) -> list[str]:
    """Pull ``[[X]]`` from ``text``, resolve each, dedup in source order.

    Uses single-target ``resolve`` semantics — ambiguous short links
    return no anchor. (For multi-link expansion at *write* time, see
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
    graph: BaseFileGraph, links: Iterable[FileLink],
) -> list[FileLink]:
    """Resolve pre-resolution ``FileLink`` records against the graph.

    Each input link has ``path`` already passed through
    ``_complete_path`` (so it has an extension). Returns FileLinks
    with ``path`` rewritten to the vault-relative resolved form
    file_graph stores. ``anchor`` and ``predicate`` pass through
    unchanged.

    Resolution dispatch:
      * literal path (contains ``/``) → direct ``get_node`` lookup;
        keep if exists, drop if dangling.
      * short link (no ``/``)         → basename match + folder-note
        rule. Ambiguity expands into multiple FileLinks (one per
        candidate path); dangling produces zero.

    Output cardinality per input link:

        * literal, target indexed                   → 1 link
        * literal, dangling                         → 0 links
        * short, 1 folder-note hit                  → 1 link
        * short, N folder-note hits                 → N links (one per)
        * short, 0 folder-notes, 1 basename hit     → 1 link
        * short, 0 folder-notes, N basename hits    → N links (one per)
        * short, dangling                           → 0 links

    Build the basename index lazily: only paid if at least one input
    is a short link.
    """
    link_list = list(links)
    if not link_list:
        return []

    basename_index: dict[str, list[str]] | None = None
    out: list[FileLink] = []
    for link in link_list:
        target = link.path
        if not target:
            continue

        if _has_dir(target):
            if await graph.get_node(target) is None:
                continue
            out.append(FileLink(
                path=target,
                anchor=link.anchor,
                predicate=link.predicate,
            ))
            continue

        # Short link — may expand into multiple links.
        if basename_index is None:
            basename_index = await _build_basename_index(graph)
        for chosen in _filter_short_link_candidates(target, basename_index.get(target, [])):
            out.append(FileLink(
                path=chosen,
                anchor=link.anchor,
                predicate=link.predicate,
            ))

    return out


async def text_to_links(
    graph: BaseFileGraph, text: str,
) -> list[FileLink]:
    """One-shot: extract wikilinks from ``text`` and resolve to safe links.

    Equivalent to ``await resolve_links(graph, iter_links(text))``.
    The parser pipeline's single-call entry point.
    """
    return await resolve_links(graph, iter_links(text))
