"""FileLink unit tests — body-only wikilink extraction.

Covers:
  * three legal inline forms (bare / line-level Dataview / inline-bracketed
    Dataview) and multi-target expansion within each
  * dedup against typed wrappers (no double-emit when a wikilink lives
    inside a typed envelope)
  * open-vocabulary predicate pass-through and the identifier-shape gate
  * the implicit-markdown rule: ``[[Foo]]`` (no extension) emerges as
    ``path="Foo.md"``; ``[[image.png]]`` (has extension) emerges as-is.
    Resolution of short links (basename match + folder-note rule) is
    tested separately against a ``BaseFileGraph``.
  * the explicit decision that frontmatter is no longer walked for links
    (extraction takes body text only)
  * the 3-field schema: ``(path, anchor, predicate)`` — all real fields,
    extra='forbid'

Pre-resolution form: ``iter_links(text)`` returns ``FileLink`` records with
``path`` set to the wikilink target with implicit ``.md`` applied. The
resolver (``utils.wikilink_resolver``) then maps short links to the
vault-relative full path; here we only test the extractor.
"""

import inspect

import pytest
from pydantic import ValidationError

from reme2.schema import FileLink
from reme2.utils.wikilink_resolver import extract_wikilinks, iter_links


# --------------------------------------------------------------------------
# Bare wikilinks — implicit ``.md`` completion at extraction time
# --------------------------------------------------------------------------


def test_bare_wikilink_gets_implicit_md():
    links = iter_links("see [[X]]")
    assert len(links) == 1
    assert links[0].path == "X.md"
    assert links[0].predicate is None
    assert links[0].anchor is None


def test_anchor_split_from_target():
    """``[[X#sec]]`` → path='X.md' (implicit), anchor='sec'."""
    links = iter_links("![[X#sec|Alias]]")
    assert len(links) == 1
    assert links[0].path == "X.md"
    assert links[0].anchor == "sec"


def test_alias_only_dropped():
    links = iter_links("[[X|Alias]]")
    assert len(links) == 1
    assert links[0].path == "X.md"
    assert links[0].anchor is None


def test_embed_only_dropped():
    links = iter_links("![[X]]")
    assert len(links) == 1
    assert links[0].path == "X.md"


def test_explicit_extension_kept_as_is():
    """``[[image.png]]`` already has an extension — no completion."""
    assert iter_links("![[image.png]]")[0].path == "image.png"
    assert iter_links("[[notes.txt]]")[0].path == "notes.txt"


def test_explicit_md_extension_kept_as_is():
    """``[[Foo.md]]`` already has the ``.md`` extension — no double-append."""
    assert iter_links("[[Foo.md]]")[0].path == "Foo.md"


def test_dir_path_no_extension_gets_md():
    """``[[topics/Bar]]`` → ``topics/Bar.md`` (last segment lacks extension)."""
    assert iter_links("[[topics/Bar]]")[0].path == "topics/Bar.md"


def test_dir_path_with_extension_kept():
    """``[[topics/image.png]]`` → kept literal."""
    assert iter_links("[[topics/image.png]]")[0].path == "topics/image.png"


# --------------------------------------------------------------------------
# Line-level Dataview
# --------------------------------------------------------------------------


def test_line_level_field():
    links = iter_links("extends:: [[Source Topic]]")
    assert len(links) == 1
    assert links[0].path == "Source Topic.md"
    assert links[0].predicate == "extends"


def test_line_level_multi_target():
    links = iter_links("concerns:: [[A]], [[B]], [[C]]")
    assert [(link.path, link.predicate) for link in links] == [
        ("A.md", "concerns"),
        ("B.md", "concerns"),
        ("C.md", "concerns"),
    ]


def test_line_level_with_bullet():
    links = iter_links("- extends:: [[X]]\n  * concerns:: [[Y]]")
    assert [(link.path, link.predicate) for link in links] == [
        ("X.md", "extends"),
        ("Y.md", "concerns"),
    ]


# --------------------------------------------------------------------------
# Multi-link cases — many wikilinks under one or several typed contexts
# --------------------------------------------------------------------------


def test_multi_target_with_anchors_preserves_each():
    """Each comma-separated target keeps its own anchor as a separate field."""
    links = iter_links("extends:: [[A#sec1]], [[B#sec2]], [[C]]")
    assert [(link.path, link.anchor, link.predicate) for link in links] == [
        ("A.md", "sec1", "extends"),
        ("B.md", "sec2", "extends"),
        ("C.md", None, "extends"),
    ]


def test_multi_target_non_comma_separator_still_typed():
    """Wikilinks anywhere in the value range (not just comma-separated)
    inherit the line's predicate. Useful for prose-style fields."""
    links = iter_links("extends:: [[A]] and also [[B]]")
    assert [(link.path, link.predicate) for link in links] == [
        ("A.md", "extends"),
        ("B.md", "extends"),
    ]


def test_multi_dataview_lines_each_multi_target():
    """Multiple Dataview lines each with multi-target → all links typed
    by their respective line's predicate."""
    links = iter_links(
        "extends:: [[A]], [[B]]\nrelates:: [[C]], [[D]]",
    )
    assert [(link.path, link.predicate) for link in links] == [
        ("A.md", "extends"),
        ("B.md", "extends"),
        ("C.md", "relates"),
        ("D.md", "relates"),
    ]


def test_inline_bracketed_then_bare_on_same_line():
    """Inline-bracketed governs only the wikilinks inside its brackets;
    a trailing bare wikilink on the same line stays bare."""
    links = iter_links("[ext:: [[A]]] then [[B]]")
    assert [(link.path, link.predicate) for link in links] == [
        ("A.md", "ext"),
        ("B.md", None),
    ]


def test_mid_line_dataview_like_not_typed():
    """``predicate::`` only counts at line start (modulo bullet) —
    a ``predicate::`` mid-line is just prose, so its wikilinks are bare."""
    links = iter_links("[ext:: [[A]]] and concerns:: [[B]]")
    assert [(link.path, link.predicate) for link in links] == [
        ("A.md", "ext"),
        ("B.md", None),  # `concerns::` mid-line is not Dataview
    ]


# --------------------------------------------------------------------------
# Inline-bracketed Dataview
# --------------------------------------------------------------------------


def test_inline_bracketed():
    links = iter_links("This [extends:: [[Y]]] something else.")
    assert len(links) == 1
    assert links[0].path == "Y.md"
    assert links[0].predicate == "extends"


def test_inline_bracketed_multi_target():
    links = iter_links("[concerns:: [[A]], [[B]]]")
    assert [(link.path, link.predicate) for link in links] == [
        ("A.md", "concerns"),
        ("B.md", "concerns"),
    ]


def test_inline_bracketed_skips_cross_line():
    # A `[predicate:: ...]` that spans a newline is malformed → the inner
    # wikilink falls back to bare; the unmatched `[` does not eat tail text.
    links = iter_links("[extends:: [[X]]\nbad]")
    assert len(links) == 1
    assert links[0].path == "X.md"
    assert links[0].predicate is None


# --------------------------------------------------------------------------
# Dedup: a wikilink inside a typed wrapper should not double-emit
# --------------------------------------------------------------------------


def test_inline_bracketed_does_not_double_emit():
    links = iter_links("see [extends:: [[X]]] again.")
    assert len(links) == 1
    assert links[0].predicate == "extends"


def test_line_level_value_does_not_double_emit():
    links = iter_links("extends:: [[X]]")
    assert len(links) == 1


def test_typed_and_bare_coexist_for_same_target():
    links = iter_links("extends:: [[X]]\nFree text mentioning [[X]] again.")
    paths_preds = sorted(((link.path, link.predicate or "") for link in links))
    assert paths_preds == [("X.md", ""), ("X.md", "extends")]


# --------------------------------------------------------------------------
# Open-vocabulary predicates — any identifier-shaped token passes through
# --------------------------------------------------------------------------


def test_arbitrary_predicate_preserved():
    links = iter_links("anything_goes:: [[X]]")
    assert len(links) == 1
    assert links[0].path == "X.md"
    assert links[0].predicate == "anything_goes"


def test_inline_arbitrary_predicate_preserved():
    links = iter_links("[wat:: [[X]]]")
    assert len(links) == 1
    assert links[0].predicate == "wat"


def test_predicate_must_be_identifier_shaped():
    # A leading digit fails the regex `[A-Za-z][A-Za-z0-9_]*` so the line is
    # not recognised as a Dataview field — the wikilink falls back to bare.
    links = iter_links("123bad:: [[X]]")
    assert len(links) == 1
    assert links[0].path == "X.md"
    assert links[0].predicate is None


def test_file_link_accepts_any_predicate_string():
    link = FileLink(path="X.md", predicate="totally_made_up")
    assert link.predicate == "totally_made_up"


# --------------------------------------------------------------------------
# Frontmatter is NOT walked for links — explicit regression
# --------------------------------------------------------------------------


def test_iter_links_takes_body_text_only():
    """``iter_links`` operates on body text — no frontmatter walking."""
    sig = inspect.signature(iter_links)
    assert list(sig.parameters.keys()) == ["text"], "iter_links should accept body text only — frontmatter walk removed"


def test_no_frontmatter_walker_exported():
    from reme2.schema import file_link as fl

    for removed in (
        "parse_wikilinks_from_metadata",
        "extract_wikilinks_from_metadata",
        "extract_inline_fields",
        "extract_typed_edges",
        "InlineField",
    ):
        assert not hasattr(fl, removed), f"{removed} should have been removed when YAML links were dropped"


# --------------------------------------------------------------------------
# FileLink schema
# --------------------------------------------------------------------------


def test_file_link_extra_forbid():
    with pytest.raises(ValidationError):
        FileLink(path="X.md", target="X.md")  # type: ignore[call-arg]


def test_file_link_field_set():
    """Stored fields are ``(path, anchor, predicate)`` — no others."""
    link = FileLink(path="X.md")
    dumped = link.model_dump()
    assert set(dumped.keys()) == {"path", "anchor", "predicate"}
    assert dumped == {"path": "X.md", "anchor": None, "predicate": None}


def test_file_link_dump_excludes_none_when_asked():
    link = FileLink(path="X.md")
    assert link.model_dump(exclude_none=True) == {"path": "X.md"}


def test_file_link_full_construction():
    link = FileLink(path="topics/Foo.md", anchor="sec", predicate="extends")
    assert link.path == "topics/Foo.md"
    assert link.anchor == "sec"
    assert link.predicate == "extends"


def test_file_link_path_required():
    with pytest.raises(ValidationError):
        FileLink()  # type: ignore[call-arg]


# --------------------------------------------------------------------------
# Anchor extraction edge cases
# --------------------------------------------------------------------------


def test_anchor_extraction_edge_cases():
    """``[[X]]`` → no anchor; ``[[X#sec]]`` → anchor='sec'.
    The wikilink regex requires the anchor capture to be one or more
    chars, so a literal ``[[X#]]`` doesn't match the regex at all (the
    trailing ``#`` makes it invalid syntax). Whitespace-only anchors
    are treated as no anchor (stripped to empty → None)."""
    assert iter_links("[[X]]")[0].anchor is None
    assert iter_links("[[X#sec]]")[0].anchor == "sec"
    # `[[X#]]` is not a valid wikilink — anchor group requires 1+ chars.
    assert iter_links("[[X#]]") == []
    # `[[X# ]]` matches but strips to empty → anchor=None.
    assert iter_links("[[X# ]]")[0].anchor is None


def test_anchor_inside_pipe_alias_still_extracted():
    """Anchor is captured before the alias pipe."""
    links = iter_links("[[topics/Foo#sec|Display]]")
    assert links[0].path == "topics/Foo.md"
    assert links[0].anchor == "sec"


# --------------------------------------------------------------------------
# Back-compat: extract_wikilinks returns flat target list (used by retriever).
# Note: extract_wikilinks does NOT apply implicit-md completion — callers
# (resolve, extract_anchors) apply it themselves at resolve time.
# --------------------------------------------------------------------------


def test_extract_wikilinks_flat_targets():
    targets = extract_wikilinks("see [[X]] and extends:: [[Y]] and [extends:: [[Z]]]")
    assert targets == ["X", "Y", "Z"]


def test_extract_wikilinks_strips_anchor():
    """``extract_wikilinks`` returns just the file part — anchor stripped."""
    targets = extract_wikilinks("see [[X#sec]] and ![[Y#a|alias]]")
    assert targets == ["X", "Y"]


def test_extract_wikilinks_does_not_complete_md():
    """Raw form — no implicit ``.md`` (that's a resolution-stage concern)."""
    targets = extract_wikilinks("[[Foo]] [[image.png]] [[topics/Bar]]")
    assert targets == ["Foo", "image.png", "topics/Bar"]


# --------------------------------------------------------------------------
# Source ordering stability
# --------------------------------------------------------------------------


def test_links_sorted_by_source_position():
    body = "intro [[First]] then\nextends:: [[Second]]\ntail [[Third]]\n"
    links = iter_links(body)
    assert [link.path for link in links] == ["First.md", "Second.md", "Third.md"]
