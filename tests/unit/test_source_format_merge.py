"""Unit tests for session-chunk merging in ``format_chunks_answer``.

Session chunks (``*.jsonl`` under the dialog dir) whose line ranges overlap,
contain one another, or are adjacent are merged into their union before
rendering. Bodies here are plain (non-``Msg``) text, so ``render_chunk_body``
falls back to the raw chunk text — letting these tests assert the union
content and line order directly.
"""

from reme.schema import FileChunk
from reme.steps.index._source_format import format_chunks_answer

_DIALOG_DIR = "session"


def _chunk(start: int, end: int, text: str, score: float = 1.0, path: str = "session/s1.jsonl") -> FileChunk:
    return FileChunk(path=path, start_line=start, end_line=end, text=text, scores={"score": score})


def test_overlapping_session_chunks_merge_into_union_without_duplicates():
    """Overlapping ranges collapse to one passage; the shared line is shown once, in order."""
    a = _chunk(1, 3, "L1\nL2\nL3\n")
    b = _chunk(3, 5, "L3\nL4\nL5\n")

    answer = format_chunks_answer([a, b], _DIALOG_DIR, include_source=False)

    assert answer == "L1\nL2\nL3\nL4\nL5\n"


def test_contained_session_chunk_is_absorbed_by_the_larger_range():
    """When one range fully contains another, only the union (the larger) is shown."""
    big = _chunk(1, 5, "L1\nL2\nL3\nL4\nL5\n")
    small = _chunk(2, 4, "L2\nL3\nL4\n")

    answer = format_chunks_answer([big, small], _DIALOG_DIR, include_source=False)

    assert answer == "L1\nL2\nL3\nL4\nL5\n"


def test_adjacent_session_chunks_merge_end_plus_one_equals_next_start():
    """Gap-free consecutive chunks (prev.end + 1 == next.start) merge into one union."""
    a = _chunk(1, 3, "L1\nL2\nL3\n")
    b = _chunk(4, 6, "L4\nL5\nL6\n")

    answer = format_chunks_answer([a, b], _DIALOG_DIR, include_source=False)

    assert answer == "L1\nL2\nL3\nL4\nL5\nL6\n"


def test_session_chunks_with_a_gap_are_not_merged():
    """A missing line between ranges (start > end + 1) keeps the chunks separate."""
    a = _chunk(1, 3, "L1\nL2\nL3\n")
    b = _chunk(5, 6, "L5\nL6\n")  # line 4 missing -> not adjacent

    answer = format_chunks_answer([a, b], _DIALOG_DIR, include_source=False)

    assert answer == "L1\nL2\nL3\n\n\nL5\nL6\n"


def test_session_chunks_from_different_files_are_not_merged():
    """Overlapping ranges in different session files must stay separate."""
    a = _chunk(1, 3, "A1\nA2\nA3\n", path="session/s1.jsonl")
    b = _chunk(2, 4, "B2\nB3\nB4\n", path="session/s2.jsonl")

    answer = format_chunks_answer([a, b], _DIALOG_DIR, include_source=False)

    assert answer == "A1\nA2\nA3\n\n\nB2\nB3\nB4\n"


def test_non_session_chunks_are_never_merged():
    """Non-transcript chunks (not ``*.jsonl`` under the dialog dir) pass through untouched."""
    a = _chunk(1, 3, "M1\nM2\nM3\n", path="daily/a.md")
    b = _chunk(2, 4, "M2\nM3\nM4\n", path="daily/a.md")

    answer = format_chunks_answer([a, b], _DIALOG_DIR, include_source=False)

    assert answer == "M1\nM2\nM3\n\n\nM2\nM3\nM4\n"


def test_merge_preserves_line_order_regardless_of_input_rank_order():
    """A later, higher-ranked chunk does not reorder union content; lines stay chronological."""
    later = _chunk(3, 5, "L3\nL4\nL5\n", score=9.0)
    earlier = _chunk(1, 3, "L1\nL2\nL3\n", score=1.0)

    # Higher-scored later-range chunk is listed first (as a ranker would).
    answer = format_chunks_answer([later, earlier], _DIALOG_DIR, include_source=False)

    assert answer == "L1\nL2\nL3\nL4\nL5\n"


def test_merged_header_spans_the_union_range_and_keeps_best_score():
    """With source headers, the merged entry reports the union range and the top score."""
    a = _chunk(1, 3, "L1\nL2\nL3\n", score=2.0)
    b = _chunk(3, 5, "L3\nL4\nL5\n", score=7.0)

    answer = format_chunks_answer([a, b], _DIALOG_DIR, include_source=True)

    assert answer.count("==========") == 2  # exactly one header (open + close markers)
    assert "session/s1.jsonl:1-5" in answer
    assert "score=7.0000" in answer


def test_separate_intervals_in_same_file_stay_separate():
    """Two disjoint interval clusters in one file yield two merged units, ordered by line."""
    a = _chunk(1, 2, "L1\nL2\n")
    b = _chunk(3, 4, "L3\nL4\n")  # adjacent to a -> merges with a into 1-4
    c = _chunk(10, 11, "L10\nL11\n")  # far away -> separate

    answer = format_chunks_answer([a, b, c], _DIALOG_DIR, include_source=False)

    assert answer == "L1\nL2\nL3\nL4\n\n\nL10\nL11\n"


def test_same_file_units_stay_adjacent_and_sorted_even_when_interleaved_by_rank():
    """Two disjoint units of one session file are grouped together and ordered by
    ``start_line``, even when a different file is ranked between them and the
    lower interval was ranked last."""
    s1_high = _chunk(10, 12, "S1x\nS1y\nS1z\n", score=9.0, path="session/s1.jsonl")
    s2_mid = _chunk(1, 3, "S2a\nS2b\nS2c\n", score=5.0, path="session/s2.jsonl")
    s1_low = _chunk(1, 3, "S1a\nS1b\nS1c\n", score=1.0, path="session/s1.jsonl")

    # Rank order (as a ranker would emit, by score desc): s1[10-12], s2[1-3], s1[1-3].
    answer = format_chunks_answer([s1_high, s2_mid, s1_low], _DIALOG_DIR, include_source=False)

    # s1's two units are adjacent and sorted by start_line (1-3 before 10-12),
    # placed at s1's earliest rank (0), so the whole s1 block precedes s2.
    assert answer == "S1a\nS1b\nS1c\n\n\nS1x\nS1y\nS1z\n\n\nS2a\nS2b\nS2c\n"


def test_same_file_units_adjacency_with_source_headers():
    """Header view: same-file units are contiguous and ascending; other files follow."""
    s1_high = _chunk(10, 12, "S1x\nS1y\nS1z\n", score=9.0, path="session/s1.jsonl")
    s2_mid = _chunk(1, 3, "S2a\nS2b\nS2c\n", score=5.0, path="session/s2.jsonl")
    s1_low = _chunk(1, 3, "S1a\nS1b\nS1c\n", score=1.0, path="session/s1.jsonl")

    answer = format_chunks_answer([s1_high, s2_mid, s1_low], _DIALOG_DIR, include_source=True)

    headers = [line for line in answer.splitlines() if line.startswith("==========")]
    assert headers[0].split(" [")[0] == "========== session/s1.jsonl:1-3"
    assert headers[1].split(" [")[0] == "========== session/s1.jsonl:10-12"
    assert headers[2].split(" [")[0] == "========== session/s2.jsonl:1-3"
