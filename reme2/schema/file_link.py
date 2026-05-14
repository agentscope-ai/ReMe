"""FileLink — typed wikilink between vault files.

One type, two states (the value of ``path`` distinguishes them):

    * **pre-resolution** — ``path`` holds the raw wikilink target as
      written, e.g. ``"Foo"`` or ``"topics/Bar"``. Produced by
      ``iter_links(text)``: a pure regex pass over body text, no graph
      access.
    * **resolved**       — ``path`` holds the vault-relative path
      file_graph stores, e.g. ``"topics/Foo/Foo.md"``. Produced by
      ``utils.wikilink_resolver.resolve_links(graph, links)`` (or the
      one-shot ``text_to_links(graph, text)``), which expands stem
      ambiguity into one ``FileLink`` per candidate path.

file_graph trusts ``link.path`` directly for adjacency: it only ever
stores resolved links. The pre-resolution form is internal pipeline
plumbing.

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
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field


# -- Regexes (module-private) ---------------------------------------------

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


# -- Schema ---------------------------------------------------------------


class FileLink(BaseModel):
    """Typed wikilink — ``(path, anchor, predicate)``.

    Single type for both states (see module docstring): ``path`` is
    a raw wikilink target before resolution, a vault-relative resolved
    path after.

    Fields:
        path       wikilink target. Raw target text in pre-resolution
                   form (e.g. ``"Foo"``, ``"topics/Bar"``);
                   vault-relative resolved path in stored form
                   (e.g. ``"topics/Foo/Foo.md"``).
        anchor     heading or block anchor (text after ``#`` in the
                   wikilink). Pass-through across resolution.
        predicate  Dataview-style typed-link predicate; ``None`` for bare.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        ...,
        description=(
            "Wikilink target. Pre-resolution: the raw target as written "
            "(e.g. 'Foo'). Resolved: the vault-relative path file_graph "
            "stores. Stem ambiguity is resolved BEFORE construction of "
            "the resolved form by emitting one FileLink per candidate."
        ),
    )
    anchor: str | None = Field(
        default=None,
        description="Heading or block anchor (text after '#'). None if absent.",
    )
    predicate: str | None = Field(
        default=None,
        description="Typed-link predicate (Dataview-style). None for bare [[X]].",
    )


# -- Public extraction utilities ------------------------------------------


def iter_links(text: str) -> list[FileLink]:
    """Extract every wikilink in ``text`` as a pre-resolution ``FileLink``.

    Each emitted ``FileLink`` has ``path`` set to the raw wikilink
    target as written (the file portion only — the ``#anchor`` lives
    in its own field). Predicate is decided by surrounding context
    with precedence **inline-bracketed > line-level > bare**.

    Pure function: no graph access. Pass the result through
    ``utils.wikilink_resolver.resolve_links`` (or one-shot
    ``text_to_links``) to get the resolved form file_graph stores.
    """
    if not text:
        return []
    inline_spans = _iter_inline_fields(text)
    links: list[FileLink] = []
    for wm in _WIKILINK_RE.finditer(text):
        target = wm.group("target").strip()
        anchor_raw = wm.group("anchor")
        anchor = anchor_raw.strip() if anchor_raw else ""
        links.append(
            FileLink(
                path=target,
                anchor=anchor or None,
                predicate=_predicate_for(text, wm.start(), inline_spans),
            ),
        )
    return links


def extract_wikilinks(text: str) -> list[str]:
    """Flat list of wikilink **file targets** in body text (no dedup).

    Returns just the file part of each wikilink (before any ``#anchor``).
    Single regex pass — cheaper than ``iter_links`` when callers don't
    need predicates (e.g. ``extract_anchors`` for query seeding in the
    retriever).
    """
    if not text:
        return []
    return [m.group("target").strip() for m in _WIKILINK_RE.finditer(text)]
