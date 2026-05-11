"""Wikilink protocol unit tests — body-only edge extraction.

Covers the three legal inline forms (bare / line-level Dataview /
inline-bracketed Dataview), multi-target expansion, dedup against
typed wrappers, open-vocabulary predicate pass-through, and the
explicit decision that frontmatter is no longer walked for links.
"""

from reme2.schema.file_edge import FileEdge, extract_wikilinks, parse_wikilinks


# --------------------------------------------------------------------------
# Bare wikilinks
# --------------------------------------------------------------------------


def test_bare_wikilink():
    edges = parse_wikilinks("see [[X]]")
    assert len(edges) == 1
    assert edges[0].target == "X"
    assert edges[0].predicate is None
    assert edges[0].embed is False


def test_bare_with_anchor_alias_embed():
    edges = parse_wikilinks("![[X#sec|Alias]]")
    assert len(edges) == 1
    assert edges[0].target == "X"
    assert edges[0].anchor == "sec"
    assert edges[0].alias == "Alias"
    assert edges[0].embed is True


# --------------------------------------------------------------------------
# Line-level Dataview
# --------------------------------------------------------------------------


def test_line_level_field():
    edges = parse_wikilinks("extends:: [[Source Topic]]")
    assert len(edges) == 1
    assert edges[0].target == "Source Topic"
    assert edges[0].predicate == "extends"


def test_line_level_multi_target():
    edges = parse_wikilinks("concerns:: [[A]], [[B]], [[C]]")
    assert [(e.target, e.predicate) for e in edges] == [
        ("A", "concerns"),
        ("B", "concerns"),
        ("C", "concerns"),
    ]


def test_line_level_with_bullet():
    edges = parse_wikilinks("- extends:: [[X]]\n  * concerns:: [[Y]]")
    assert [(e.target, e.predicate) for e in edges] == [
        ("X", "extends"),
        ("Y", "concerns"),
    ]


# --------------------------------------------------------------------------
# Inline-bracketed Dataview
# --------------------------------------------------------------------------


def test_inline_bracketed():
    edges = parse_wikilinks("This [extends:: [[Y]]] something else.")
    assert len(edges) == 1
    assert edges[0].target == "Y"
    assert edges[0].predicate == "extends"


def test_inline_bracketed_multi_target():
    edges = parse_wikilinks("[concerns:: [[A]], [[B]]]")
    assert [(e.target, e.predicate) for e in edges] == [
        ("A", "concerns"),
        ("B", "concerns"),
    ]


def test_inline_bracketed_skips_cross_line():
    # A `[predicate:: ...]` that spans a newline is malformed → the inner
    # wikilink falls back to bare; the unmatched `[` does not eat tail text.
    edges = parse_wikilinks("[extends:: [[X]]\nbad]")
    assert len(edges) == 1
    assert edges[0].target == "X"
    assert edges[0].predicate is None


# --------------------------------------------------------------------------
# Dedup: a wikilink inside a typed wrapper should not double-emit
# --------------------------------------------------------------------------


def test_inline_bracketed_does_not_double_emit():
    edges = parse_wikilinks("see [extends:: [[X]]] again.")
    assert len(edges) == 1
    assert edges[0].predicate == "extends"


def test_line_level_value_does_not_double_emit():
    edges = parse_wikilinks("extends:: [[X]]")
    assert len(edges) == 1


def test_typed_and_bare_coexist_for_same_target():
    edges = parse_wikilinks("extends:: [[X]]\nFree text mentioning [[X]] again.")
    targets_preds = sorted(((e.target, e.predicate or "") for e in edges))
    assert targets_preds == [("X", ""), ("X", "extends")]


# --------------------------------------------------------------------------
# Open-vocabulary predicates — any identifier-shaped token passes through
# --------------------------------------------------------------------------


def test_arbitrary_predicate_preserved():
    edges = parse_wikilinks("anything_goes:: [[X]]")
    assert len(edges) == 1
    assert edges[0].target == "X"
    assert edges[0].predicate == "anything_goes"


def test_inline_arbitrary_predicate_preserved():
    edges = parse_wikilinks("[wat:: [[X]]]")
    assert len(edges) == 1
    assert edges[0].predicate == "wat"


def test_predicate_must_be_identifier_shaped():
    # A leading digit fails the regex `[A-Za-z][A-Za-z0-9_]*` so the line is
    # not recognised as a Dataview field — the wikilink falls back to bare.
    edges = parse_wikilinks("123bad:: [[X]]")
    assert len(edges) == 1
    assert edges[0].target == "X"
    assert edges[0].predicate is None


def test_file_edge_accepts_any_predicate_string():
    e = FileEdge(target="X", predicate="totally_made_up")
    assert e.predicate == "totally_made_up"


# --------------------------------------------------------------------------
# Frontmatter is NOT walked for links — explicit regression
# --------------------------------------------------------------------------


def test_frontmatter_links_block_is_ignored():
    # Even if a YAML-shaped string sits at the top of body, parse_wikilinks
    # only operates on body. We pass body text directly here, so this test
    # asserts the API surface no longer accepts a metadata dict.
    import inspect

    sig = inspect.signature(parse_wikilinks)
    assert list(sig.parameters.keys()) == ["text"], (
        "parse_wikilinks should accept body text only — frontmatter walk removed"
    )


def test_no_frontmatter_walker_exported():
    from reme2.schema import file_edge as wl

    for removed in (
        "parse_wikilinks_from_metadata",
        "extract_wikilinks_from_metadata",
        "extract_inline_fields",
        "extract_typed_edges",
        "InlineField",
    ):
        assert not hasattr(wl, removed), (
            f"{removed} should have been removed when YAML edges were dropped"
        )


# --------------------------------------------------------------------------
# FileEdge schema
# --------------------------------------------------------------------------


def test_file_edge_extra_forbid():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        FileEdge(target="X", source="regex")  # type: ignore[call-arg]


def test_file_edge_minimal_field_set():
    e = FileEdge(target="X")
    dumped = e.model_dump()
    assert set(dumped.keys()) == {"target", "predicate", "anchor", "alias", "embed"}


# --------------------------------------------------------------------------
# Back-compat: extract_wikilinks returns flat target list (used by ingestor)
# --------------------------------------------------------------------------


def test_extract_wikilinks_flat_targets():
    targets = extract_wikilinks("see [[X]] and extends:: [[Y]] and [extends:: [[Z]]]")
    assert targets == ["X", "Y", "Z"]


# --------------------------------------------------------------------------
# Source ordering stability
# --------------------------------------------------------------------------


def test_edges_sorted_by_source_position():
    body = (
        "intro [[First]] then\n"
        "extends:: [[Second]]\n"
        "tail [[Third]]\n"
    )
    edges = parse_wikilinks(body)
    assert [e.target for e in edges] == ["First", "Second", "Third"]
