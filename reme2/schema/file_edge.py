"""FileEdge — typed wikilink edge between vault files.

Single source of truth for the edge **schema** and the **inline parser**
(`FileEdge.from_text`) that recovers edges from body text. Edges live
exclusively in body text (frontmatter is not walked); the predicate
vocabulary is **open** — any identifier-shaped token
(`[A-Za-z][A-Za-z0-9_]*`) the parser sees is preserved verbatim on
`FileEdge.predicate`. Vocabulary curation, if any, is the maintainer's
job, not the schema's.

## Inline forms recognised by `FileEdge.from_text`

    [[X]]                           bare wikilink         → predicate=None
    extends:: [[X]]                 line-level Dataview   → predicate="extends"
    [extends:: [[X]]]               inline-bracketed      → predicate="extends"

Multi-target — every wikilink under one typed context inherits its
predicate (any separator works, not just commas):

    extends:: [[A]], [[B]]          line-level multi      → 2 edges, both "extends"
    extends:: [[A]] and [[B]]       prose-style multi     → 2 edges, both "extends"
    [concerns:: [[A]], [[B]]]       inline multi          → 2 edges, both "concerns"
    extends:: [[A#s1]], [[B#s2]]    multi w/ anchors      → anchors preserved per link

Context precedence is **inline-bracketed > line-level > bare** —
a wikilink inside a `[predicate:: …]` envelope is typed by that
envelope even if the line happens to start `predicate:: …`.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field


# -- Regexes (module-private) ---------------------------------------------

# Wikilink. The optional `!` embed marker and `|alias` are matched but
# not captured — both are presentational and dropped from the edge.
# `target` is the file part; `anchor` is the optional `#…` suffix; we
# rejoin them into a single `link` string at edge-construction time.
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

# Line-level Dataview field. Anchored MULTILINE; allows leading bullet
# (`-`/`*`/`+`) so `- extends:: [[X]]` works inside Markdown lists.
_DATAVIEW_LINE_RE = re.compile(
    r"^[ \t]*(?:[-*+][ \t]+)?(?P<predicate>[A-Za-z][A-Za-z0-9_]*)\s*::\s*(?P<value>.+?)\s*$",
    re.MULTILINE,
)

# Inline-bracketed field opener: `[predicate::`. The matching `]` is
# located by depth-counted scan because the value may contain
# `[[wikilink]]` whose inner brackets are part of the value.
_INLINE_FIELD_OPEN_RE = re.compile(r"\[(?P<predicate>[A-Za-z][A-Za-z0-9_]*)\s*::\s*")


def _iter_inline_fields(text: str) -> list[tuple[int, int, str]]:
    """Find inline-bracketed `[predicate:: …]` field spans by depth scan.

    Returns a list of ``(start, end, predicate)`` triples. Newlines
    terminate the scan: an inline field that spans a line break is
    treated as malformed (matches Dataview semantics) and skipped.
    """
    out: list[tuple[int, int, str]] = []
    for m in _INLINE_FIELD_OPEN_RE.finditer(text):
        depth = 1  # the outer '[' was the regex's first character
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


# -- Schema ---------------------------------------------------------------


class FileEdge(BaseModel):
    """2-field minimal edge model: ``link`` + ``predicate``.

    ``link`` preserves the wikilink as written, including any ``#anchor``
    suffix (e.g. ``"X"`` or ``"X#sec"``). The presentational ``|alias``
    and ``!`` embed prefix are discarded by the parser. ``path`` and
    ``anchor`` are derived ``@property``s that split ``link`` on the
    first ``#`` for callers that want either part without re-splitting
    the string themselves.
    """

    model_config = ConfigDict(extra="forbid")

    link: str = Field(
        ...,
        description=(
            "Wikilink as written, preserving any '#anchor' suffix "
            "(e.g. 'X' or 'X#sec'). Display alias and embed prefix discarded."
        ),
    )
    predicate: str | None = Field(
        default=None,
        description="Typed-edge predicate (Dataview-style). None for bare [[X]].",
    )

    @property
    def path(self) -> str:
        """File-part of ``link`` (before any ``#anchor``).

        For ``"X"`` returns ``"X"``; for ``"X#sec"`` returns ``"X"``;
        for ``"topics/Foo#bar"`` returns ``"topics/Foo"``. Always
        returns a non-empty string because ``link`` is required and the
        regex never matches an empty target.
        """
        return self.link.split("#", 1)[0].strip()

    @property
    def anchor(self) -> str | None:
        """Heading or block anchor parsed from ``link`` (text after first ``#``).

        Returns ``None`` if the link has no anchor or the anchor is empty.
        """
        if "#" not in self.link:
            return None
        tail = self.link.split("#", 1)[1].strip()
        return tail or None

    @classmethod
    def _from_match(cls, wm: re.Match, *, predicate: str | None) -> FileEdge:
        target = wm.group("target").strip()
        anchor = wm.group("anchor")
        link = f"{target}#{anchor.strip()}" if anchor else target
        return cls(link=link, predicate=predicate)

    @classmethod
    def from_text(cls, text: str) -> list[FileEdge]:
        """Extract all edges from body text in source order.

        Single pass: every wikilink in the text becomes one edge, and
        its ``predicate`` is decided by the surrounding context with
        precedence **inline-bracketed > line-level > bare**:

        * ``[predicate:: [[X]]]``  → ``predicate="predicate"``
        * ``predicate:: [[X]]``    → ``predicate="predicate"``  (line-level Dataview)
        * ``[[X]]``                → ``predicate=None``         (bare)

        No consumed-span bookkeeping, no second sort: ``finditer``
        already yields wikilinks in source order, and per-position
        classification is unambiguous.
        """
        if not text:
            return []

        # Inline-bracketed `[predicate:: ...]` envelopes need a
        # depth-counted scan (regex can't match balanced `[[…]]` inside).
        inline_spans = _iter_inline_fields(text)

        return [
            cls._from_match(wm, predicate=_predicate_for(text, wm.start(), inline_spans))
            for wm in _WIKILINK_RE.finditer(text)
        ]


def _predicate_for(
    text: str,
    pos: int,
    inline_spans: list[tuple[int, int, str]],
) -> str | None:
    """Resolve the predicate governing a wikilink at offset ``pos``.

    Checks the two typed-edge contexts in precedence order; falls
    through to ``None`` (bare) when neither applies.
    """
    # 1. Inline-bracketed envelope `[predicate:: …]` containing pos.
    for field_start, field_end, predicate in inline_spans:
        if field_start <= pos < field_end:
            return predicate

    # 2. Line-level `predicate:: value` whose value range covers pos.
    line_start = text.rfind("\n", 0, pos) + 1
    line_end = text.find("\n", pos)
    if line_end == -1:
        line_end = len(text)
    m = _DATAVIEW_LINE_RE.match(text[line_start:line_end])
    if m and line_start + m.start("value") <= pos:
        return m.group("predicate")

    # 3. Bare wikilink — no predicate.
    return None


# -- Public utility (target-only fast path) -------------------------------


def extract_wikilinks(text: str) -> list[str]:
    """Flat list of wikilink **file targets** in body text (no dedup).

    Returns just the file part of each wikilink (before any ``#anchor``)
    because callers feed the result to `resolve_wikilink`, which matches
    against vault file stems / paths and would not recognise an anchor
    suffix. Single regex pass — cheaper than `FileEdge.from_text` when
    callers don't need predicates (e.g. ingestor's auto-discovery hint,
    memory_io anchor resolution).
    """
    if not text:
        return []
    return [m.group("target").strip() for m in _WIKILINK_RE.finditer(text)]
