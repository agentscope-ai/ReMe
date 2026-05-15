"""Link parser — wikilink syntax extraction from text.

A **Link** is a textual reference to a vault file (``[[X]]`` and
typed/anchored variants). This module is the syntax layer:
text → ``FileLink`` records.

``FileLink.path`` here is the **raw target as written** (no
``.md`` completion, no resolution). To turn the raw target into a
vault-relative key, call ``resolve_links`` (or ``text_to_links`` for
one-shot extract+resolve), which delegates to ``path_resolver``.

## Inline forms recognised

    [[X]]                           bare wikilink         → predicate=None
    extends:: [[X]]                 line-level Dataview   → predicate="extends"
    [extends:: [[X]]]               inline-bracketed      → predicate="extends"

Multi-target — every wikilink under one typed context inherits its
predicate (any separator works):

    extends:: [[A]], [[B]]          → 2 links, both "extends"
    [concerns:: [[A]] and [[B]]]    → 2 links, both "concerns"
    extends:: [[A#s1]], [[B#s2]]    → anchors preserved per link

Context precedence: **inline-bracketed > line-level > bare**.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from ..component.file_graph.base_file_graph import BaseFileGraph
from ..schema import FileLink
from . import path_resolver


WIKILINK_RE = re.compile(
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


def iter_links(text: str) -> list[FileLink]:
    """Extract every wikilink in ``text`` as a pre-resolution ``FileLink``.

    ``FileLink.path`` is the raw target as written (no ``.md``
    completion, no graph lookup); ``anchor`` and ``predicate`` are
    parsed from surrounding syntax. Pure function — no graph access.
    Pass through ``resolve_links`` (or ``text_to_links``) to get the
    vault-relative form.
    """
    if not text:
        return []
    inline_spans = _iter_inline_fields(text)
    links: list[FileLink] = []
    for wm in WIKILINK_RE.finditer(text):
        target = wm.group("target").strip()
        anchor_raw = wm.group("anchor")
        anchor = anchor_raw.strip() if anchor_raw else ""
        links.append(FileLink(
            path=target,
            anchor=anchor or None,
            predicate=_predicate_for(text, wm.start(), inline_spans),
        ))
    return links


async def resolve_links(
    graph: BaseFileGraph, links: Iterable[FileLink],
) -> list[FileLink]:
    """Resolve each ``FileLink.path`` via ``path_resolver``.

    Parser-pipeline semantics — short-path ambiguity **expands** into
    one ``FileLink`` per candidate (so the body's wikilink is recorded
    against every plausible target). Dangling links are dropped.

    Operation-time callers that need a single primary key should call
    ``path_resolver.resolve`` directly and let ``PathAmbiguous``
    propagate.
    """
    out: list[FileLink] = []
    for link in links:
        if not link.path:
            continue
        try:
            resolved = await path_resolver.resolve(graph, link.path)
            out.append(FileLink(
                path=resolved,
                anchor=link.anchor,
                predicate=link.predicate,
            ))
        except path_resolver.PathAmbiguous as e:
            for c in e.candidates:
                out.append(FileLink(
                    path=c,
                    anchor=link.anchor,
                    predicate=link.predicate,
                ))
        except path_resolver.PathNotFound:
            continue
    return out


async def text_to_links(graph: BaseFileGraph, text: str) -> list[FileLink]:
    """One-shot ``iter_links`` + ``resolve_links`` for the parser pipeline."""
    return await resolve_links(graph, iter_links(text))
