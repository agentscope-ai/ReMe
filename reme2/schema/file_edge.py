"""FileEdge — typed wikilink edge between vault files.

This module is the single source of truth for both the edge **schema**
and the **inline parser** that recovers edges from body text. Edges
live exclusively in body text (frontmatter is not walked); the
predicate vocabulary is **open** — any identifier-shaped token
(`[A-Za-z][A-Za-z0-9_]*`) that the parser sees is preserved verbatim
on `FileEdge.predicate`. Vocabulary curation, if any, is the
maintainer's job, not the schema's.

## Inline forms recognised by `parse_wikilinks`

    [[X]]                           bare wikilink         → predicate=None
    extends:: [[X]]                 line-level Dataview   → predicate="extends"
    [extends:: [[X]]]               inline-bracketed      → predicate="extends"
    extends:: [[A]], [[B]]          multi-target          → 2 edges
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field


# -- Schema ---------------------------------------------------------------


class FileEdge(BaseModel):
    """5-field minimal edge model. No provenance, no confidence."""

    model_config = ConfigDict(extra="forbid")

    target: str = Field(..., description="Raw wikilink target as written in source.")
    predicate: str | None = Field(
        default=None,
        description="Typed-edge predicate (Dataview-style). None for bare [[X]].",
    )
    anchor: str | None = Field(default=None, description="Heading or block anchor (after #).")
    alias: str | None = Field(default=None, description="Display alias (after |).")
    embed: bool = Field(default=False, description="True for `![[X]]` embed prefix.")


# -- Regexes --------------------------------------------------------------

# Bare wikilink. `(?:!)?` is non-capturing so `m.group(1)` stays the
# target — read the embed prefix off `m.group(0).startswith("!")`.
WIKILINK_RE = re.compile(
    r"""
    (?:!)?
    \[\[
        (?P<target>[^\]\|\#\n]+?)
        (?:\#(?P<anchor>[^\]\|\n]+))?
        (?:\|(?P<alias>[^\]\n]+))?
    \]\]
    """,
    re.VERBOSE,
)

# Line-level Dataview field. Anchored MULTILINE; allows leading bullet
# (`-`/`*`/`+`) so `- extends:: [[X]]` works inside Markdown lists.
# Predicate identifier follows Dataview convention: letter, then
# letters / digits / underscore.
DATAVIEW_LINE_RE = re.compile(
    r"^[ \t]*(?:[-*+][ \t]+)?(?P<predicate>[A-Za-z][A-Za-z0-9_]*)\s*::\s*(?P<value>.+?)\s*$",
    re.MULTILINE,
)

# Inline-bracketed field: opens with `[predicate::`. The closing `]` is
# located by `_iter_inline_fields` via depth-counted bracket scan because
# the value may contain `[[wikilink]]` whose inner `[[…]]` brackets are
# part of the value, not field delimiters.
_INLINE_FIELD_OPEN_RE = re.compile(r"\[(?P<predicate>[A-Za-z][A-Za-z0-9_]*)\s*::\s*")


# -- Internal helpers -----------------------------------------------------


def _edge_from_wm(wm: re.Match, *, predicate: str | None) -> FileEdge:
    anchor = wm.group("anchor")
    alias = wm.group("alias")
    return FileEdge(
        target=wm.group("target").strip(),
        anchor=anchor.strip() if anchor else None,
        alias=alias.strip() if alias else None,
        embed=wm.group(0).startswith("!"),
        predicate=predicate,
    )


def _iter_inline_fields(text: str) -> list[tuple[int, int, str, int]]:
    """Find inline-bracketed `[predicate:: …]` fields by depth scan.

    Returns ``(start, end, predicate, value_start)`` tuples — `value_start`
    is the absolute offset where the value begins inside `text`, used to
    project wikilink spans back to absolute positions for dedup.

    Newlines terminate the scan: an inline field that spans a line break
    is treated as malformed and skipped (matches Dataview's parser).
    """
    out: list[tuple[int, int, str, int]] = []
    for m in _INLINE_FIELD_OPEN_RE.finditer(text):
        value_start = m.end()
        depth = 1  # the outer '[' was the regex's first character
        i = value_start
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
                    out.append((m.start(), i + 1, m.group("predicate"), value_start))
                    break
            i += 1
    return out


# -- Public parsing API ---------------------------------------------------


def extract_wikilinks(text: str) -> list[str]:
    """Targets-only list of wikilinks in body text (no dedup).

    Used by callers that need to follow links structurally without caring
    about predicates (e.g. the ingestor's auto-discovery hint).
    """
    if not text:
        return []
    return [m.group("target").strip() for m in WIKILINK_RE.finditer(text)]


def parse_wikilinks(text: str) -> list[FileEdge]:
    """Structured parse of all three edge forms.

    Order is by source position. Wikilinks attributed to an inline-
    bracketed or line-level field are not also reported as bare; the
    same wikilink span is consumed exactly once.
    """
    if not text:
        return []

    consumed: list[tuple[int, int]] = []
    items: list[tuple[int, FileEdge]] = []

    # 1. Inline-bracketed: most specific, handled first.
    for field_start, field_end, predicate, value_start in _iter_inline_fields(text):
        value = text[value_start : field_end - 1]
        for wm in WIKILINK_RE.finditer(value):
            items.append((field_start, _edge_from_wm(wm, predicate=predicate)))
        consumed.append((field_start, field_end))

    # 2. Line-level: skip if entirely inside an inline-bracketed span.
    for m in DATAVIEW_LINE_RE.finditer(text):
        if any(cs <= m.start() and m.end() <= ce for cs, ce in consumed):
            continue
        predicate = m.group("predicate")
        value = m.group("value")
        value_start = m.start("value")
        for wm in WIKILINK_RE.finditer(value):
            wl_start = value_start + wm.start()
            wl_end = value_start + wm.end()
            items.append((wl_start, _edge_from_wm(wm, predicate=predicate)))
            consumed.append((wl_start, wl_end))

    # 3. Bare: anything left over.
    for wm in WIKILINK_RE.finditer(text):
        s, e = wm.span()
        if any(cs <= s and e <= ce for cs, ce in consumed):
            continue
        items.append((s, _edge_from_wm(wm, predicate=None)))

    items.sort(key=lambda pair: pair[0])
    return [edge for _, edge in items]
