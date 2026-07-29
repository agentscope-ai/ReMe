"""Unit tests for session-chunk merging via ``merge_session_chunk_intervals``.

Session chunks (``*.jsonl`` under the dialog dir) whose line ranges overlap,
contain one another, or are adjacent are merged into their union by the caller
(``merge_session_chunk_intervals``) before rendering with
``render_chunk_entries`` + ``join_chunk_entries``. Bodies here are plain
(non-``Msg``) text lines, which pass through stripped and get a ``L<n>:``
line-number prefix — letting these tests assert the union content, line order,
and file-line numbering directly.
"""

from reme.schema import FileChunk
from reme.steps.index._source_format import join_chunk_entries, merge_session_chunk_intervals, render_chunk_entries

_DIALOG_DIR = "session"


def _render(chunks: list[FileChunk], dialog_dir: str, **kwargs) -> str:
    """Merge session chunks, render entries, then join — mirroring the search steps."""
    return join_chunk_entries(
        render_chunk_entries(merge_session_chunk_intervals(chunks, dialog_dir), dialog_dir, **kwargs),
    )


def _chunk(start: int, end: int, text: str, score: float = 1.0, path: str = "session/s1.jsonl") -> FileChunk:
    return FileChunk(path=path, start_line=start, end_line=end, text=text, scores={"score": score})


def test_overlapping_session_chunks_merge_into_union_without_duplicates():
    """Overlapping ranges collapse to one passage; the shared line is shown once, in order."""
    a = _chunk(1, 3, "m1\nm2\nm3\n")
    b = _chunk(3, 5, "m3\nm4\nm5\n")

    answer = _render([a, b], _DIALOG_DIR, include_source=False)

    assert answer == "L1: m1\nL2: m2\nL3: m3\nL4: m4\nL5: m5"


def test_contained_session_chunk_is_absorbed_by_the_larger_range():
    """When one range fully contains another, only the union (the larger) is shown."""
    big = _chunk(1, 5, "m1\nm2\nm3\nm4\nm5\n")
    small = _chunk(2, 4, "m2\nm3\nm4\n")

    answer = _render([big, small], _DIALOG_DIR, include_source=False)

    assert answer == "L1: m1\nL2: m2\nL3: m3\nL4: m4\nL5: m5"


def test_adjacent_session_chunks_merge_end_plus_one_equals_next_start():
    """Gap-free consecutive chunks (prev.end + 1 == next.start) merge into one union."""
    a = _chunk(1, 3, "m1\nm2\nm3\n")
    b = _chunk(4, 6, "m4\nm5\nm6\n")

    answer = _render([a, b], _DIALOG_DIR, include_source=False)

    assert answer == "L1: m1\nL2: m2\nL3: m3\nL4: m4\nL5: m5\nL6: m6"


def test_session_chunks_with_a_gap_are_not_merged():
    """A missing line between ranges (start > end + 1) keeps the chunks separate."""
    a = _chunk(1, 3, "m1\nm2\nm3\n")
    b = _chunk(5, 6, "m5\nm6\n")  # line 4 missing -> not adjacent

    answer = _render([a, b], _DIALOG_DIR, include_source=False)

    assert answer == "L1: m1\nL2: m2\nL3: m3\n\nL5: m5\nL6: m6"


def test_session_chunks_from_different_files_are_not_merged():
    """Overlapping ranges in different session files must stay separate."""
    a = _chunk(1, 3, "A1\nA2\nA3\n", path="session/s1.jsonl")
    b = _chunk(2, 4, "B2\nB3\nB4\n", path="session/s2.jsonl")

    answer = _render([a, b], _DIALOG_DIR, include_source=False)

    assert answer == "L1: A1\nL2: A2\nL3: A3\n\nL2: B2\nL3: B3\nL4: B4"


def test_non_session_chunks_are_never_merged():
    """Non-transcript chunks (not ``*.jsonl`` under the dialog dir) pass through untouched."""
    a = _chunk(1, 3, "M1\nM2\nM3\n", path="daily/a.md")
    b = _chunk(2, 4, "M2\nM3\nM4\n", path="daily/a.md")

    answer = _render([a, b], _DIALOG_DIR, include_source=False)

    assert answer == "M1\nM2\nM3\n\nM2\nM3\nM4"


def test_merge_preserves_line_order_regardless_of_input_rank_order():
    """A later, higher-ranked chunk does not reorder union content; lines stay chronological."""
    later = _chunk(3, 5, "m3\nm4\nm5\n", score=9.0)
    earlier = _chunk(1, 3, "m1\nm2\nm3\n", score=1.0)

    # Higher-scored later-range chunk is listed first (as a ranker would).
    answer = _render([later, earlier], _DIALOG_DIR, include_source=False)

    assert answer == "L1: m1\nL2: m2\nL3: m3\nL4: m4\nL5: m5"


def test_merged_header_spans_the_union_range_and_keeps_best_score():
    """With source headers, the merged entry reports the union range and the top score."""
    a = _chunk(1, 3, "m1\nm2\nm3\n", score=2.0)
    b = _chunk(3, 5, "m3\nm4\nm5\n", score=7.0)

    answer = _render([a, b], _DIALOG_DIR, include_source=True)

    assert answer.count("==========") == 2  # exactly one header (open + close markers)
    assert "session/s1.jsonl:1-5" in answer
    assert "score=7.0000" in answer


def test_separate_intervals_in_same_file_stay_separate():
    """Two disjoint interval clusters in one file yield two merged units, ordered by line."""
    a = _chunk(1, 2, "m1\nm2\n")
    b = _chunk(3, 4, "m3\nm4\n")  # adjacent to a -> merges with a into 1-4
    c = _chunk(10, 11, "m10\nm11\n")  # far away -> separate

    answer = _render([a, b, c], _DIALOG_DIR, include_source=False)

    assert answer == "L1: m1\nL2: m2\nL3: m3\nL4: m4\n\nL10: m10\nL11: m11"


def test_same_file_units_stay_adjacent_and_sorted_even_when_interleaved_by_rank():
    """Two disjoint units of one session file are grouped together and ordered by
    ``start_line``, even when a different file is ranked between them and the
    lower interval was ranked last."""
    s1_high = _chunk(10, 12, "S1x\nS1y\nS1z\n", score=9.0, path="session/s1.jsonl")
    s2_mid = _chunk(1, 3, "S2a\nS2b\nS2c\n", score=5.0, path="session/s2.jsonl")
    s1_low = _chunk(1, 3, "S1a\nS1b\nS1c\n", score=1.0, path="session/s1.jsonl")

    # Rank order (as a ranker would emit, by score desc): s1[10-12], s2[1-3], s1[1-3].
    answer = _render([s1_high, s2_mid, s1_low], _DIALOG_DIR, include_source=False)

    # s1's two units are adjacent and sorted by start_line (1-3 before 10-12),
    # placed at s1's earliest rank (0), so the whole s1 block precedes s2.
    assert answer == "L1: S1a\nL2: S1b\nL3: S1c\n\nL10: S1x\nL11: S1y\nL12: S1z\n\nL1: S2a\nL2: S2b\nL3: S2c"


def test_same_file_units_adjacency_with_source_headers():
    """Header view: same-file units are contiguous and ascending; other files follow."""
    s1_high = _chunk(10, 12, "S1x\nS1y\nS1z\n", score=9.0, path="session/s1.jsonl")
    s2_mid = _chunk(1, 3, "S2a\nS2b\nS2c\n", score=5.0, path="session/s2.jsonl")
    s1_low = _chunk(1, 3, "S1a\nS1b\nS1c\n", score=1.0, path="session/s1.jsonl")

    answer = _render([s1_high, s2_mid, s1_low], _DIALOG_DIR, include_source=True)

    headers = [line for line in answer.splitlines() if line.startswith("==========")]
    assert headers[0].split(" [")[0] == "========== session/s1.jsonl:1-3"
    assert headers[1].split(" [")[0] == "========== session/s1.jsonl:10-12"
    assert headers[2].split(" [")[0] == "========== session/s2.jsonl:1-3"
