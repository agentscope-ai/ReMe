"""Wikilink protocol — Obsidian-compatible bare links + Dataview-style typed edges.

## Bare wikilinks (Obsidian)

    [[Note]]                  short form (resolve by stem)
    [[path/to/Note]]          path form
    [[Note#Section]]          heading anchor
    [[Note#^block-id]]        block anchor
    [[Note|Alias]]            display alias
    ![[Note]]                 embed (transclusion)
    ![[Note#Section|Alias]]   all combined

## Typed wikilinks / property edges (Dataview inline field syntax)

    [predicate:: [[Target]]]      typed link (bracketed, visible)
    (predicate:: [[Target]])      typed link (parenthesized, hidden in render)
    [predicate:: scalar value]    typed scalar (becomes a property, not a graph edge)

`predicate` is identifier-shaped: starts with a letter, then letters /
digits / `_` / `-`. The link inside follows the bare protocol above.

## Output schema

Wikilink edges are returned as `FileEdge` (the persistence model exposed
via `file_store.get_edges(path)`). Typed scalar properties are returned
as `InlineField` — they carry no link target, just `{predicate, value}`.

## Functions

Bare extraction (back-compat — return targets only):
    extract_wikilinks(text)              → list[str]
    extract_wikilinks_from_metadata(d)   → list[str]

Structured parsing (return FileEdge):
    parse_wikilinks(text)                → list[FileEdge]
    parse_wikilinks_from_metadata(d)     → list[FileEdge]
    extract_inline_fields(text)          → list[InlineField]
    extract_typed_edges(text)            → list[FileEdge]   (predicate-bearing only)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ..schema.file_edge import FileEdge


EdgeSource = Literal["regex", "frontmatter", "llm"]


# -- Regexes --------------------------------------------------------------

# Bare wikilink. Group(1) = target, kept as the FIRST capturing group so
# legacy callers using `m.group(1)` (e.g. wikilink rewriters) still work.
# The leading `(?:!)?` is non-capturing — read the embed prefix off
# `m.group(0).startswith("!")` instead.
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

# Dataview inline field: [pred:: value] or (pred:: value). Value may be a
# bare wikilink OR a scalar with no inner brackets. The open/close bracket
# pair isn't enforced by the regex — `_bracket_pair_ok` rejects mismatched
# `[..)`-style fragments.
INLINE_FIELD_RE = re.compile(
    r"""
    (?P<open>[\[\(])
    (?P<predicate>[A-Za-z][\w\-]*)
    \s*::\s*
    (?P<value>
        (?: !? \[\[ [^\]\|\#\n]+? (?:\#[^\]\|\n]+)? (?:\|[^\]\n]+)? \]\] )
        |
        (?: [^\]\)\n]+? )
    )
    (?P<close>[\]\)])
    """,
    re.VERBOSE,
)


# -- Auxiliary schema (typed scalar) --------------------------------------


@dataclass(frozen=True, slots=True)
class InlineField:
    """Typed scalar property — `[predicate:: value]` where value is NOT a wikilink.

    Distinct from `FileEdge`: a scalar property doesn't point at another file,
    so it's not a graph edge — it's an attribute on the source file.
    """

    predicate: str
    value: str


# -- Internal helpers -----------------------------------------------------


def _edge_from_match(
    m: re.Match,
    *,
    predicate: str | None = None,
    source: EdgeSource = "regex",
) -> FileEdge:
    anchor = m.group("anchor")
    alias = m.group("alias")
    return FileEdge(
        target=m.group("target").strip(),
        anchor=anchor.strip() if anchor else None,
        alias=alias.strip() if alias else None,
        embed=m.group(0).startswith("!"),
        predicate=predicate,
        source=source,
    )


def _bracket_pair_ok(open_ch: str, close_ch: str) -> bool:
    return (open_ch == "[" and close_ch == "]") or (open_ch == "(" and close_ch == ")")


# -- Public API: bare extraction (back-compat) ----------------------------


def extract_wikilinks(text: str) -> list[str]:
    """Targets-only list of wikilinks found in text (no dedup).

    Includes wikilinks inside typed wrappers — the watcher's graph
    projection should see every link target, with or without a predicate.
    Use `parse_wikilinks` if you need the predicate too.
    """
    if not text:
        return []
    return [m.group("target").strip() for m in WIKILINK_RE.finditer(text)]


def extract_wikilinks_from_metadata(metadata: dict) -> list[str]:
    """Recursively walk a frontmatter dict, extracting wikilink targets from string leaves."""
    links: list[str] = []

    def _walk(value) -> None:
        if isinstance(value, str):
            links.extend(extract_wikilinks(value))
        elif isinstance(value, dict):
            for v in value.values():
                _walk(v)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                _walk(item)

    _walk(metadata)
    return links


# -- Public API: structured parsing ---------------------------------------


def parse_wikilinks(text: str) -> list[FileEdge]:
    """Structured parse — bare wikilinks AND typed-wrapper predicates.

    Order is by source position. A wikilink inside a typed wrapper
    appears once, with its predicate attached. All edges have
    `source="regex"`.
    """
    if not text:
        return []

    typed_spans: list[tuple[int, int]] = []
    items: list[tuple[int, FileEdge]] = []

    for m in INLINE_FIELD_RE.finditer(text):
        if not _bracket_pair_ok(m.group("open"), m.group("close")):
            continue
        value = m.group("value").strip()
        wm = WIKILINK_RE.fullmatch(value)
        if wm is None:
            continue  # scalar — handled by extract_inline_fields
        items.append((
            m.start(),
            _edge_from_match(wm, predicate=m.group("predicate").strip()),
        ))
        typed_spans.append(m.span())

    for m in WIKILINK_RE.finditer(text):
        s, e = m.span()
        if any(ts <= s and e <= te for ts, te in typed_spans):
            continue  # this bare match is the inner link of a typed wrapper
        items.append((s, _edge_from_match(m)))

    items.sort(key=lambda pair: pair[0])
    return [w for _, w in items]


def extract_inline_fields(text: str) -> list[InlineField]:
    """Typed scalar fields — `[pred:: value]` whose value is NOT a wikilink.

    Link-valued typed fields are returned by `parse_wikilinks` instead,
    since they're graph edges rather than scalar properties.
    """
    if not text:
        return []
    out: list[InlineField] = []
    for m in INLINE_FIELD_RE.finditer(text):
        if not _bracket_pair_ok(m.group("open"), m.group("close")):
            continue
        value = m.group("value").strip()
        if WIKILINK_RE.fullmatch(value) is not None:
            continue
        out.append(InlineField(predicate=m.group("predicate").strip(), value=value))
    return out


def extract_typed_edges(text: str) -> list[FileEdge]:
    """Sugar: just the predicate-bearing wikilinks (typed graph edges)."""
    return [w for w in parse_wikilinks(text) if w.is_typed]


def parse_wikilinks_from_metadata(metadata: dict) -> list[FileEdge]:
    """Recursive frontmatter parse — frontmatter keys act as predicates.

    `author: "[[John]]"` → `FileEdge(target="John", predicate="author", source="frontmatter")`.
    `related: ["[[X]]", "[[Y]]"]` → two `FileEdge(predicate="related", ...)`.
    A bare wikilink in a string value with no parent key keeps `predicate=None`.
    Inline `[pred:: ...]` syntax inside a string value still wins over the
    inherited frontmatter key (predicate carried from the inline parse).
    All edges have `source="frontmatter"`.
    """
    links: list[FileEdge] = []

    def _walk(value, predicate: str | None) -> None:
        if isinstance(value, str):
            for w in parse_wikilinks(value):
                effective = w.predicate if w.predicate is not None else predicate
                links.append(FileEdge(
                    target=w.target,
                    anchor=w.anchor,
                    alias=w.alias,
                    embed=w.embed,
                    predicate=effective,
                    source="frontmatter",
                ))
        elif isinstance(value, dict):
            for k, v in value.items():
                _walk(v, predicate=str(k))
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                _walk(item, predicate=predicate)

    _walk(metadata, predicate=None)
    return links
