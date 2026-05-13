"""FileEdge unit tests — body-only edge extraction.

Covers the three legal inline forms (bare / line-level Dataview /
inline-bracketed Dataview), multi-target expansion, dedup against
typed wrappers, open-vocabulary predicate pass-through, the explicit
decision that frontmatter is no longer walked for links, and the
2-field schema (`link` + `predicate`) with `anchor` as a derived
property and `alias` / `embed` discarded at parse time.
"""

from reme2.schema.file_edge import FileEdge, extract_wikilinks


# --------------------------------------------------------------------------
# Bare wikilinks
# --------------------------------------------------------------------------


def test_bare_wikilink():
    edges = FileEdge.from_text("see [[X]]")
    assert len(edges) == 1
    assert edges[0].link == "X"
    assert edges[0].predicate is None
    assert edges[0].anchor is None


def test_anchor_alias_embed_collapse_into_link():
    """`![[X#sec|Alias]]` → link='X#sec' (alias and embed dropped)."""
    edges = FileEdge.from_text("![[X#sec|Alias]]")
    assert len(edges) == 1
    assert edges[0].link == "X#sec"
    # anchor is a derived property parsed from link.
    assert edges[0].anchor == "sec"


def test_alias_only_drops_to_link():
    edges = FileEdge.from_text("[[X|Alias]]")
    assert len(edges) == 1
    assert edges[0].link == "X"
    assert edges[0].anchor is None


def test_embed_only_drops_to_link():
    edges = FileEdge.from_text("![[X]]")
    assert len(edges) == 1
    assert edges[0].link == "X"


# --------------------------------------------------------------------------
# Line-level Dataview
# --------------------------------------------------------------------------


def test_line_level_field():
    edges = FileEdge.from_text("extends:: [[Source Topic]]")
    assert len(edges) == 1
    assert edges[0].link == "Source Topic"
    assert edges[0].predicate == "extends"


def test_line_level_multi_target():
    edges = FileEdge.from_text("concerns:: [[A]], [[B]], [[C]]")
    assert [(e.link, e.predicate) for e in edges] == [
        ("A", "concerns"),
        ("B", "concerns"),
        ("C", "concerns"),
    ]


def test_line_level_with_bullet():
    edges = FileEdge.from_text("- extends:: [[X]]\n  * concerns:: [[Y]]")
    assert [(e.link, e.predicate) for e in edges] == [
        ("X", "extends"),
        ("Y", "concerns"),
    ]


# --------------------------------------------------------------------------
# Multi-edge cases — many wikilinks under one or several typed contexts
# --------------------------------------------------------------------------


def test_multi_target_with_anchors_preserves_each():
    """Each comma-separated target keeps its own anchor in `link`."""
    edges = FileEdge.from_text("extends:: [[A#sec1]], [[B#sec2]], [[C]]")
    assert [(e.link, e.anchor, e.predicate) for e in edges] == [
        ("A#sec1", "sec1", "extends"),
        ("B#sec2", "sec2", "extends"),
        ("C", None, "extends"),
    ]


def test_multi_target_non_comma_separator_still_typed():
    """Wikilinks anywhere in the value range (not just comma-separated)
    inherit the line's predicate. Useful for prose-style fields."""
    edges = FileEdge.from_text("extends:: [[A]] and also [[B]]")
    assert [(e.link, e.predicate) for e in edges] == [
        ("A", "extends"),
        ("B", "extends"),
    ]


def test_multi_dataview_lines_each_multi_target():
    """Multiple Dataview lines each with multi-target → all edges typed
    by their respective line's predicate."""
    edges = FileEdge.from_text(
        "extends:: [[A]], [[B]]\nrelates:: [[C]], [[D]]"
    )
    assert [(e.link, e.predicate) for e in edges] == [
        ("A", "extends"),
        ("B", "extends"),
        ("C", "relates"),
        ("D", "relates"),
    ]


def test_inline_bracketed_then_bare_on_same_line():
    """Inline-bracketed governs only the wikilinks inside its brackets;
    a trailing bare wikilink on the same line stays bare."""
    edges = FileEdge.from_text("[ext:: [[A]]] then [[B]]")
    assert [(e.link, e.predicate) for e in edges] == [
        ("A", "ext"),
        ("B", None),
    ]


def test_mid_line_dataview_like_not_typed():
    """``predicate::`` only counts at line start (modulo bullet) —
    a `predicate::` mid-line is just prose, so its wikilinks are bare."""
    edges = FileEdge.from_text("[ext:: [[A]]] and concerns:: [[B]]")
    assert [(e.link, e.predicate) for e in edges] == [
        ("A", "ext"),
        ("B", None),  # `concerns::` mid-line is not Dataview
    ]


# --------------------------------------------------------------------------
# Inline-bracketed Dataview
# --------------------------------------------------------------------------


def test_inline_bracketed():
    edges = FileEdge.from_text("This [extends:: [[Y]]] something else.")
    assert len(edges) == 1
    assert edges[0].link == "Y"
    assert edges[0].predicate == "extends"


def test_inline_bracketed_multi_target():
    edges = FileEdge.from_text("[concerns:: [[A]], [[B]]]")
    assert [(e.link, e.predicate) for e in edges] == [
        ("A", "concerns"),
        ("B", "concerns"),
    ]


def test_inline_bracketed_skips_cross_line():
    # A `[predicate:: ...]` that spans a newline is malformed → the inner
    # wikilink falls back to bare; the unmatched `[` does not eat tail text.
    edges = FileEdge.from_text("[extends:: [[X]]\nbad]")
    assert len(edges) == 1
    assert edges[0].link == "X"
    assert edges[0].predicate is None


# --------------------------------------------------------------------------
# Dedup: a wikilink inside a typed wrapper should not double-emit
# --------------------------------------------------------------------------


def test_inline_bracketed_does_not_double_emit():
    edges = FileEdge.from_text("see [extends:: [[X]]] again.")
    assert len(edges) == 1
    assert edges[0].predicate == "extends"


def test_line_level_value_does_not_double_emit():
    edges = FileEdge.from_text("extends:: [[X]]")
    assert len(edges) == 1


def test_typed_and_bare_coexist_for_same_target():
    edges = FileEdge.from_text("extends:: [[X]]\nFree text mentioning [[X]] again.")
    links_preds = sorted(((e.link, e.predicate or "") for e in edges))
    assert links_preds == [("X", ""), ("X", "extends")]


# --------------------------------------------------------------------------
# Open-vocabulary predicates — any identifier-shaped token passes through
# --------------------------------------------------------------------------


def test_arbitrary_predicate_preserved():
    edges = FileEdge.from_text("anything_goes:: [[X]]")
    assert len(edges) == 1
    assert edges[0].link == "X"
    assert edges[0].predicate == "anything_goes"


def test_inline_arbitrary_predicate_preserved():
    edges = FileEdge.from_text("[wat:: [[X]]]")
    assert len(edges) == 1
    assert edges[0].predicate == "wat"


def test_predicate_must_be_identifier_shaped():
    # A leading digit fails the regex `[A-Za-z][A-Za-z0-9_]*` so the line is
    # not recognised as a Dataview field — the wikilink falls back to bare.
    edges = FileEdge.from_text("123bad:: [[X]]")
    assert len(edges) == 1
    assert edges[0].link == "X"
    assert edges[0].predicate is None


def test_file_edge_accepts_any_predicate_string():
    e = FileEdge(link="X", predicate="totally_made_up")
    assert e.predicate == "totally_made_up"


# --------------------------------------------------------------------------
# Frontmatter is NOT walked for links — explicit regression
# --------------------------------------------------------------------------


def test_frontmatter_links_block_is_ignored():
    # Even if a YAML-shaped string sits at the top of body, FileEdge.from_text
    # only operates on body. We pass body text directly here, so this test
    # asserts the API surface no longer accepts a metadata dict.
    import inspect

    sig = inspect.signature(FileEdge.from_text)
    assert list(sig.parameters.keys()) == ["text"], (
        "FileEdge.from_text should accept body text only — frontmatter walk removed"
    )


def test_no_frontmatter_walker_exported():
    from reme2.schema import file_edge as fe

    for removed in (
        "parse_wikilinks_from_metadata",
        "extract_wikilinks_from_metadata",
        "extract_inline_fields",
        "extract_typed_edges",
        "InlineField",
    ):
        assert not hasattr(fe, removed), (
            f"{removed} should have been removed when YAML edges were dropped"
        )


# --------------------------------------------------------------------------
# FileEdge schema
# --------------------------------------------------------------------------


def test_file_edge_extra_forbid():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        FileEdge(link="X", target="X")  # type: ignore[call-arg]


def test_file_edge_minimal_field_set():
    """Stored fields are just `link` and `predicate`. `path` and `anchor`
    are `@property`s (not stored fields) so they shouldn't appear in
    `model_dump()`. `target` / `alias` / `embed` were dropped entirely."""
    e = FileEdge(link="X")
    dumped = e.model_dump()
    assert set(dumped.keys()) == {"link", "predicate"}
    for removed in ("target", "alias", "embed", "anchor", "path"):
        assert removed not in dumped


def test_anchor_property_parses_from_link():
    """`anchor` is derived from `link`, not stored separately."""
    assert FileEdge(link="X").anchor is None
    assert FileEdge(link="X#sec").anchor == "sec"
    assert FileEdge(link="X#sec#more").anchor == "sec#more"
    # Empty anchor is treated as no anchor.
    assert FileEdge(link="X#").anchor is None
    assert FileEdge(link="X# ").anchor is None


def test_path_property_parses_from_link():
    """`path` is derived from `link` — file part before any `#anchor`."""
    assert FileEdge(link="X").path == "X"
    assert FileEdge(link="X#sec").path == "X"
    assert FileEdge(link="topics/Foo").path == "topics/Foo"
    assert FileEdge(link="topics/Foo#bar").path == "topics/Foo"
    # Multiple '#' — only first splits; the rest live in anchor.
    assert FileEdge(link="X#a#b").path == "X"
    assert FileEdge(link="X#a#b").anchor == "a#b"
    # Empty anchor → path is still the full prefix.
    assert FileEdge(link="X#").path == "X"


def test_path_anchor_roundtrip_via_link():
    """Reconstructing `link` from `path` + `anchor` yields the original."""
    for link in ("X", "X#sec", "topics/Foo", "topics/Foo#bar", "X#a#b"):
        e = FileEdge(link=link)
        rebuilt = e.path if not e.anchor else f"{e.path}#{e.anchor}"
        assert rebuilt == link, f"{link!r} → {rebuilt!r}"


def test_anchor_is_not_constructor_arg():
    """Since `anchor` and `path` are properties, passing them to the
    constructor should fail (extra='forbid')."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        FileEdge(link="X", anchor="sec")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        FileEdge(link="X", path="X")  # type: ignore[call-arg]


# --------------------------------------------------------------------------
# Back-compat: extract_wikilinks returns flat target list (used by ingestor)
# --------------------------------------------------------------------------


def test_extract_wikilinks_flat_targets():
    targets = extract_wikilinks("see [[X]] and extends:: [[Y]] and [extends:: [[Z]]]")
    assert targets == ["X", "Y", "Z"]


def test_extract_wikilinks_strips_anchor():
    """`extract_wikilinks` returns just the file part — anchor stripped
    so callers can feed it to `resolve_wikilink`."""
    targets = extract_wikilinks("see [[X#sec]] and ![[Y#a|alias]]")
    assert targets == ["X", "Y"]


# --------------------------------------------------------------------------
# Source ordering stability
# --------------------------------------------------------------------------


def test_edges_sorted_by_source_position():
    body = (
        "intro [[First]] then\n"
        "extends:: [[Second]]\n"
        "tail [[Third]]\n"
    )
    edges = FileEdge.from_text(body)
    assert [e.link for e in edges] == ["First", "Second", "Third"]
