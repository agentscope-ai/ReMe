"""Markdown AST chunker tests — full-skeleton TOC + inlined content.

Each chunk renders the **complete heading skeleton of the document**
(every heading, top-to-bottom) with the chunk's content inlined under
the section that owns it. Sections that don't own this chunk's content
appear as bare headings — every chunk gives the reader a full document
map.

Covers:
* Tree build: heading-stack folding, section ranges, body wrap-up.
* Whole-fit: small docs / sections emit as a single chunk.
* Skeleton completeness: every chunk lists every doc heading.
* Owner positioning: content sits under the right section's heading.
* Body run packing: bodies under one section share the same owner slot.
* Subsection recursion: each subsection chunks under the same skeleton
  with its own owner slot.
* Leaf split: lists / tables / code fences / paragraphs split internally
  with their structural header (table separator, code fence, list bullet)
  preserved per piece — and the full doc skeleton wraps each piece.
"""

from reme2.component.file_parser.linked_file_parser import (
    LinkedFileParser,
    MdNode,
)


def _parser(chunk_chars: int, embed_toc: bool = True) -> LinkedFileParser:
    """Construct a parser without invoking BaseComponent.__init__ (no app context)."""
    p = LinkedFileParser.__new__(LinkedFileParser)
    p.encoding = "utf-8"
    p.chunk_chars = chunk_chars
    p.embed_toc = embed_toc
    return p


def _all_headings(text: str) -> list[str]:
    """All markdown heading lines in `text`, in order."""
    return [ln.strip() for ln in text.split("\n") if ln.lstrip().startswith("#")]


# --------------------------------------------------------------------------
# Tree build
# --------------------------------------------------------------------------


def test_tree_groups_under_headings():
    txt = (
        "# Top\n"
        "para1\n"
        "\n"
        "## Sub A\n"
        "para2\n"
        "\n"
        "### Deeper\n"
        "para3\n"
        "\n"
        "## Sub B\n"
        "para4\n"
    )
    from mistletoe.block_token import Document
    from mistletoe.markdown_renderer import MarkdownRenderer

    p = _parser(2000)
    with MarkdownRenderer() as r:
        tree = p._build_tree(Document(txt), r)

    assert tree.kind == "root"
    assert len(tree.children) == 1
    h1 = tree.children[0]
    assert h1.kind == "section" and h1.heading == "Top" and h1.level == 1
    kinds = [c.kind for c in h1.children]
    assert kinds == ["body", "section", "section"]
    sub_a, sub_b = h1.children[1], h1.children[2]
    assert sub_a.heading == "Sub A" and sub_a.level == 2
    assert sub_b.heading == "Sub B" and sub_b.level == 2
    assert [c.kind for c in sub_a.children] == ["body", "section"]
    deeper = sub_a.children[1]
    assert deeper.heading == "Deeper" and deeper.level == 3


def test_tree_handles_body_before_first_heading():
    txt = "intro paragraph\n\n# H1\nbody\n"
    from mistletoe.block_token import Document
    from mistletoe.markdown_renderer import MarkdownRenderer

    p = _parser(2000)
    with MarkdownRenderer() as r:
        tree = p._build_tree(Document(txt), r)
    assert [c.kind for c in tree.children] == ["body", "section"]


def test_tree_section_pop_on_equal_level():
    txt = "# Top\n## A\nx\n## B\ny\n"
    from mistletoe.block_token import Document
    from mistletoe.markdown_renderer import MarkdownRenderer

    p = _parser(2000)
    with MarkdownRenderer() as r:
        tree = p._build_tree(Document(txt), r)
    h1 = tree.children[0]
    assert [c.heading for c in h1.children if c.kind == "section"] == ["A", "B"]


# --------------------------------------------------------------------------
# Whole-fit emit
# --------------------------------------------------------------------------


def test_small_doc_emits_one_chunk():
    txt = "# Top\nhello world\n"
    chunks = _parser(500)._chunk(txt, "/x.md")
    assert len(chunks) == 1
    text = chunks[0].text
    assert "# Top" in text and "hello world" in text


# --------------------------------------------------------------------------
# Full-skeleton TOC: every chunk shows every heading
# --------------------------------------------------------------------------


def test_every_chunk_lists_every_doc_heading():
    """No matter which slice is being chunked, the chunk text must
    contain every heading in the document — that's what 'complete TOC
    structure' means."""
    txt = (
        "# Doc\n"
        "intro paragraph here\n"
        "\n"
        "## Section A\n"
        "para A content here long\n"
        "\n"
        "### Subsection\n"
        "deeper content here long\n"
        "\n"
        "## Section B\n"
        "para B content long\n"
    )
    chunks = _parser(80)._chunk(txt, "/x.md")
    assert len(chunks) >= 3
    expected_headings = {"# Doc", "## Section A", "### Subsection", "## Section B"}
    for c in chunks:
        present = set(_all_headings(c.text))
        assert expected_headings <= present, (
            f"chunk missing headings {expected_headings - present}: {c.text!r}"
        )


def test_owner_section_holds_chunk_content():
    """The chunk's content sits directly under its owner heading — not
    under any other section's heading."""
    txt = (
        "# Doc\n"
        "\n"
        "## A\n"
        "alpha alpha alpha alpha here\n"
        "\n"
        "## B\n"
        "bravo bravo bravo bravo here\n"
    )
    chunks = _parser(60)._chunk(txt, "/x.md")
    a_chunk = next(c for c in chunks if "alpha" in c.text)
    b_chunk = next(c for c in chunks if "bravo" in c.text)
    # In A's chunk, "alpha" must appear AFTER "## A" and BEFORE "## B"
    a_idx = a_chunk.text.find("alpha")
    a_a_idx = a_chunk.text.find("## A")
    a_b_idx = a_chunk.text.find("## B")
    assert a_a_idx < a_idx < a_b_idx
    # In B's chunk, "bravo" must appear AFTER "## B"
    b_idx = b_chunk.text.find("bravo")
    b_b_idx = b_chunk.text.find("## B")
    assert b_b_idx < b_idx
    # And the OTHER section's body must NOT appear in this chunk.
    assert "alpha" not in b_chunk.text
    assert "bravo" not in a_chunk.text


def test_skeleton_unchanged_across_chunks():
    """Strip out body content — every chunk should yield the same
    sequence of heading lines (the doc's skeleton)."""
    txt = (
        "# Top\n"
        "\n"
        "## A\n"
        "aaa aaa aaa long\n"
        "\n"
        "## B\n"
        "bbb bbb bbb long\n"
        "\n"
        "## C\n"
        "ccc ccc ccc long\n"
    )
    chunks = _parser(60)._chunk(txt, "/x.md")
    skeletons = [_all_headings(c.text) for c in chunks]
    expected = ["# Top", "## A", "## B", "## C"]
    for sk in skeletons:
        assert sk == expected


def test_subsection_skeleton_preserved():
    """Subsection headings still appear in EVERY chunk's skeleton, not
    just the chunk that owns the subsection's content."""
    txt = (
        "# Doc\n"
        "\n"
        "## A\n"
        "para A long content here\n"
        "\n"
        "## B\n"
        "\n"
        "### B1\n"
        "deep content here long\n"
        "\n"
        "## C\n"
        "para C long content here\n"
    )
    chunks = _parser(60)._chunk(txt, "/x.md")
    a_chunk = next(c for c in chunks if "para A" in c.text)
    # B1 heading must appear in A's chunk too — full skeleton preserved.
    assert "### B1" in a_chunk.text
    assert "## C" in a_chunk.text
    # And in B1's chunk, A's heading must appear before B's.
    b1_chunk = next(c for c in chunks if "deep content" in c.text)
    assert "## A" in b1_chunk.text
    assert "## B" in b1_chunk.text
    assert "### B1" in b1_chunk.text
    assert "## C" in b1_chunk.text


# --------------------------------------------------------------------------
# Body run packing
# --------------------------------------------------------------------------


def test_body_run_greedy_packs_under_one_owner():
    """Multiple bodies under one section pack into one chunk; each
    chunk still carries the full skeleton."""
    txt = (
        "# Top\n"
        "\n"
        "## Single\n"
        "para1 content here\n"
        "\n"
        "para2 different content\n"
        "\n"
        "para3 last paragraph\n"
    )
    chunks = _parser(80)._chunk(txt, "/x.md")
    for c in chunks:
        # Skeleton has both top and single.
        assert "# Top" in c.text
        assert "## Single" in c.text
    joined = "\n".join(c.text for c in chunks)
    for tag in ("para1", "para2", "para3"):
        assert tag in joined


# --------------------------------------------------------------------------
# Leaf splits: full skeleton wraps each piece
# --------------------------------------------------------------------------


def test_table_split_keeps_skeleton():
    txt = (
        "# Doc\n"
        "\n"
        "## Tables\n"
        "\n"
        "| name | value |\n"
        "|------|-------|\n"
        "| r1   | a     |\n"
        "| r2   | b     |\n"
        "| r3   | c     |\n"
        "| r4   | d     |\n"
        "\n"
        "## Other\n"
        "other content\n"
    )
    chunks = _parser(80)._chunk(txt, "/x.md")
    table_chunks = [c for c in chunks if "| name | value |" in c.text]
    assert len(table_chunks) >= 2
    for c in table_chunks:
        # Header repeats per piece.
        assert "| name | value |" in c.text
        assert "----" in c.text
        # Skeleton complete: # Doc, ## Tables, ## Other all present.
        assert "# Doc" in c.text
        assert "## Tables" in c.text
        assert "## Other" in c.text


def test_code_fence_split_keeps_skeleton():
    txt = (
        "# Doc\n"
        "\n"
        "## Code\n"
        "\n"
        "```python\n"
        "def line1():\n"
        "    pass\n"
        "\n"
        "def line2():\n"
        "    pass\n"
        "\n"
        "def line3():\n"
        "    pass\n"
        "\n"
        "def line4():\n"
        "    pass\n"
        "```\n"
        "\n"
        "## After\n"
        "after content\n"
    )
    chunks = _parser(100)._chunk(txt, "/x.md")
    code_chunks = [c for c in chunks if "```python" in c.text]
    assert len(code_chunks) >= 2
    for c in code_chunks:
        # Fence opener + closer repeat per piece.
        assert "```python" in c.text
        # Skeleton: # Doc, ## Code, ## After.
        assert "# Doc" in c.text
        assert "## Code" in c.text
        assert "## After" in c.text


def test_list_split_keeps_skeleton():
    txt = (
        "# Doc\n"
        "\n"
        "## Items\n"
        "\n"
        "- item one with some text\n"
        "- item two with text\n"
        "- item three text\n"
        "- item four text\n"
        "- item five text\n"
        "- item six text\n"
        "\n"
        "## After\n"
        "after content\n"
    )
    chunks = _parser(110)._chunk(txt, "/x.md")
    list_chunks = [c for c in chunks if "- item" in c.text]
    assert len(list_chunks) >= 2
    for c in list_chunks:
        assert "# Doc" in c.text
        assert "## Items" in c.text
        assert "## After" in c.text
    joined = "\n".join(c.text for c in list_chunks)
    for tag in ("one", "two", "three", "four", "five", "six"):
        assert tag in joined


def test_paragraph_line_split_keeps_skeleton():
    txt = (
        "# P\n"
        "\n"
        "## Section\n"
        "alpha line one with extra padding text here\n"
        "beta line two with extra padding text here\n"
        "gamma line three with extra padding text here\n"
        "delta line four with extra padding text here\n"
        "\n"
        "## After\n"
        "after content\n"
    )
    chunks = _parser(110)._chunk(txt, "/x.md")
    para_chunks = [c for c in chunks if any(t in c.text for t in ("alpha", "beta", "gamma", "delta"))]
    assert len(para_chunks) >= 2
    for c in para_chunks:
        assert "# P" in c.text
        assert "## Section" in c.text
        assert "## After" in c.text


# --------------------------------------------------------------------------
# embed_toc toggle + content-only budget
# --------------------------------------------------------------------------


def test_embed_toc_off_strips_skeleton():
    """With embed_toc=False, chunks contain only their own content —
    no full-doc heading skeleton wrapping them."""
    txt = (
        "# Doc\n"
        "\n"
        "## A\n"
        "para A long content here\n"
        "\n"
        "## B\n"
        "para B long content here\n"
    )
    chunks = _parser(60, embed_toc=False)._chunk(txt, "/x.md")
    a = next(c for c in chunks if "para A" in c.text)
    b = next(c for c in chunks if "para B" in c.text)
    # Neither chunk should carry the OTHER section's heading.
    assert "## B" not in a.text
    assert "## A" not in b.text
    # And neither should carry the doc title (no full-doc TOC).
    assert "# Doc" not in a.text
    assert "# Doc" not in b.text


def test_embed_toc_off_content_only():
    """A whole-doc chunk under embed_toc=False is just the rendered
    content — its own section headings remain (they're part of the
    content), but no extra TOC wrap is added."""
    txt = "# Top\nhello world content here\n"
    chunks = _parser(500, embed_toc=False)._chunk(txt, "/x.md")
    assert len(chunks) == 1
    text = chunks[0].text.strip()
    # The doc's own heading IS the content of the root chunk.
    assert text.startswith("# Top")
    assert "hello world content here" in text
    # But chunk size matches just the rendered doc — no extra prefix
    # would have been added that isn't in the source.
    assert text == "# Top\n\nhello world content here"


def test_embed_toc_default_is_on():
    """Default behavior keeps TOC embedding on."""
    p = _parser(500)
    assert p.embed_toc is True
    chunks = p._chunk("# Top\nbody content\n", "/x.md")
    assert "# Top" in chunks[0].text


def test_chunk_chars_constrains_content_only():
    """chunk_chars limits CONTENT size; the TOC skeleton is additive
    and may push final chunk text well beyond chunk_chars."""
    # Doc with a deep heading skeleton (~70 chars) and short body.
    txt = (
        "# H1 long heading title here\n"
        "## H2 long subheading title\n"
        "### H3 deep heading title\n"
        "x\n"  # body of H3
    )
    chunks = _parser(50)._chunk(txt, "/x.md")
    assert len(chunks) == 1
    final = chunks[0].text
    # Final chunk text > chunk_chars because TOC was added on top.
    assert len(final) > 50
    # All headings present (full skeleton).
    assert "# H1" in final and "## H2" in final and "### H3" in final
    # Body present.
    assert "x" in final


def test_body_run_budget_excludes_toc():
    """A body run greedy-pack uses chunk_chars purely for body content;
    the TOC skeleton overhead doesn't squeeze the budget."""
    # Two bodies whose joined size = 50 chars; with TOC skeleton,
    # old behavior (TOC counted) might split, but new should fit.
    txt = (
        "# Top\n"
        "## Sub long heading title here\n"
        "abcdefghij abcdefghij abcdefghij\n"   # 32 chars body
        "\n"
        "klmnopqrst klmnopqrst klmnopqrst\n"   # 32 chars body
    )
    # 80 chars budget covers the joined body (32+2+32=66) but is
    # smaller than body+TOC under old (counting) semantics (~120).
    chunks = _parser(80)._chunk(txt, "/x.md")
    # Both bodies in one chunk because content-only budget allows it.
    assert len(chunks) == 1
    assert "abcdefghij" in chunks[0].text
    assert "klmnopqrst" in chunks[0].text


def test_finalize_off_returns_content_unchanged():
    """White-box: _finalize with embed_toc=False is the identity for content."""
    from mistletoe.block_token import Document
    from mistletoe.markdown_renderer import MarkdownRenderer

    p = _parser(500, embed_toc=False)
    txt = "# A\n## B\nbody\n"
    with MarkdownRenderer() as r:
        tree = p._build_tree(Document(txt), r)
    h1 = tree.children[0]
    b = h1.children[0]  # ## B
    out = p._finalize(tree, b, "RAW", owns_subtree=False)
    assert out == "RAW"


def test_finalize_on_wraps_with_toc():
    """White-box: _finalize with embed_toc=True wraps content with skeleton."""
    from mistletoe.block_token import Document
    from mistletoe.markdown_renderer import MarkdownRenderer

    p = _parser(500, embed_toc=True)
    txt = "# A\n## B\nbody\n"
    with MarkdownRenderer() as r:
        tree = p._build_tree(Document(txt), r)
    h1 = tree.children[0]
    b = h1.children[0]
    out = p._finalize(tree, b, "RAW", owns_subtree=False)
    assert "# A" in out and "## B" in out and "RAW" in out


# --------------------------------------------------------------------------
# Empty / whitespace-only
# --------------------------------------------------------------------------


def test_empty_text_yields_no_chunks():
    assert _parser(500)._chunk("", "/x.md") == []
    assert _parser(500)._chunk("   \n\n  ", "/x.md") == []


# --------------------------------------------------------------------------
# Leaf-block split parts: [Part X/N] markers
# --------------------------------------------------------------------------


def test_table_split_marks_parts():
    """Table split into N pieces gets [Part X/N] prefix on each."""
    txt = (
        "# Doc\n"
        "\n"
        "## T\n"
        "\n"
        "| a | b |\n"
        "|---|---|\n"
        "| 1 | x |\n"
        "| 2 | y |\n"
        "| 3 | z |\n"
        "| 4 | u |\n"
        "| 5 | v |\n"
        "| 6 | w |\n"
    )
    chunks = _parser(80)._chunk(txt, "/x.md")
    table_chunks = [c for c in chunks if "| a   | b" in c.text]
    assert len(table_chunks) >= 2
    n = len(table_chunks)
    for i, c in enumerate(table_chunks, 1):
        assert f"[Part {i}/{n}]" in c.text
    # The marker sits BEFORE the table header.
    for c in table_chunks:
        marker_pos = c.text.find("[Part")
        header_pos = c.text.find("| a")
        assert 0 <= marker_pos < header_pos


def test_code_split_marks_parts():
    txt = (
        "# Doc\n"
        "\n"
        "## C\n"
        "\n"
        "```python\n"
        "def f1(): pass\n"
        "def f2(): pass\n"
        "def f3(): pass\n"
        "def f4(): pass\n"
        "def f5(): pass\n"
        "def f6(): pass\n"
        "```\n"
    )
    chunks = _parser(70)._chunk(txt, "/x.md")
    code_chunks = [c for c in chunks if "```python" in c.text]
    assert len(code_chunks) >= 2
    n = len(code_chunks)
    for i, c in enumerate(code_chunks, 1):
        assert f"[Part {i}/{n}]" in c.text
        # Marker before the fence opener.
        assert c.text.find("[Part") < c.text.find("```python")


def test_list_split_marks_parts():
    txt = (
        "# Doc\n"
        "\n"
        "## L\n"
        "\n"
        "- item one with extra padding text content here\n"
        "- item two with extra padding text content here\n"
        "- item three with extra padding text content here\n"
        "- item four with extra padding text content here\n"
        "- item five with extra padding text content here\n"
        "- item six with extra padding text content here\n"
    )
    chunks = _parser(120)._chunk(txt, "/x.md")
    list_chunks = [c for c in chunks if "- item" in c.text]
    assert len(list_chunks) >= 2
    n = len(list_chunks)
    for i, c in enumerate(list_chunks, 1):
        assert f"[Part {i}/{n}]" in c.text


def test_paragraph_split_marks_parts():
    txt = (
        "# Doc\n"
        "\n"
        "## P\n"
        "alpha line one with extra padding text here\n"
        "beta line two with extra padding text here\n"
        "gamma line three with extra padding text here\n"
        "delta line four with extra padding text here\n"
    )
    chunks = _parser(100)._chunk(txt, "/x.md")
    para_chunks = [
        c for c in chunks
        if any(t in c.text for t in ("alpha", "beta", "gamma", "delta"))
    ]
    assert len(para_chunks) >= 2
    n = len(para_chunks)
    for i, c in enumerate(para_chunks, 1):
        assert f"[Part {i}/{n}]" in c.text


def test_single_piece_leaf_has_no_part_marker():
    """When a leaf block fits in one piece, no [Part] prefix is added."""
    txt = (
        "# Doc\n"
        "\n"
        "## T\n"
        "\n"
        "| a | b |\n"
        "|---|---|\n"
        "| 1 | 2 |\n"
    )
    chunks = _parser(500)._chunk(txt, "/x.md")
    assert len(chunks) == 1
    assert "[Part" not in chunks[0].text


def test_part_marker_absent_for_body_run_packing():
    """Body run packing (separate blocks under one section) doesn't
    use [Part] markers — that's reserved for splitting ONE leaf block."""
    txt = (
        "# Doc\n"
        "\n"
        "## S\n"
        "para1 first paragraph content\n"
        "\n"
        "para2 second paragraph content\n"
        "\n"
        "para3 third paragraph content\n"
    )
    # Force greedy-pack across separate paragraphs.
    chunks = _parser(60)._chunk(txt, "/x.md")
    for c in chunks:
        assert "[Part" not in c.text


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def test_chunk_line_ranges_track_source():
    txt = (
        "# Top\n"      # line 1
        "intro\n"      # line 2
        "\n"
        "## Sub\n"    # line 4
        "body\n"       # line 5
    )
    chunks = _parser(500)._chunk(txt, "/x.md")
    assert len(chunks) == 1
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 5


def test_chunk_id_changes_with_content():
    a = _parser(500)._chunk("# X\nhello\n", "/x.md")[0]
    b = _parser(500)._chunk("# X\nhello\n", "/x.md")[0]
    c = _parser(500)._chunk("# X\nhellp\n", "/x.md")[0]
    assert a.id == b.id
    assert a.id != c.id


# --------------------------------------------------------------------------
# Render helper directly
# --------------------------------------------------------------------------


def test_render_with_full_toc_inlines_at_owner():
    """White-box: build a tree, call _render_with_full_toc directly and
    verify the slot-fill behavior."""
    from mistletoe.block_token import Document
    from mistletoe.markdown_renderer import MarkdownRenderer

    p = _parser(2000)
    txt = "# A\n\n## B\nbody B\n\n## C\nbody C\n"
    with MarkdownRenderer() as r:
        tree = p._build_tree(Document(txt), r)
    h1 = tree.children[0]
    b = h1.children[0]  # ## B
    rendered = p._render_with_full_toc(tree, b, "INSERTED", owns_subtree=False)
    # Skeleton lists # A, ## B, ## C; INSERTED sits under ## B but BEFORE ## C
    assert "# A" in rendered
    assert "## B" in rendered
    assert "## C" in rendered
    b_idx = rendered.find("## B")
    inserted_idx = rendered.find("INSERTED")
    c_idx = rendered.find("## C")
    assert b_idx < inserted_idx < c_idx


def test_render_with_full_toc_root_owner():
    """Root owner means content sits BEFORE the first heading."""
    from mistletoe.block_token import Document
    from mistletoe.markdown_renderer import MarkdownRenderer

    p = _parser(2000)
    txt = "# A\n\nbody A\n"
    with MarkdownRenderer() as r:
        tree = p._build_tree(Document(txt), r)
    rendered = p._render_with_full_toc(tree, tree, "PRELUDE", owns_subtree=False)
    assert rendered.startswith("PRELUDE")
    assert "# A" in rendered


# --------------------------------------------------------------------------
# MdNode dataclass
# --------------------------------------------------------------------------


def test_mdnode_defaults():
    n = MdNode(kind="body")
    assert n.heading is None and n.level == 0
    assert n.children == [] and n.block is None
    assert n.text == "" and n.start_line == 0
