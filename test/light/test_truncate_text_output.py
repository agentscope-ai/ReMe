# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring
"""Unit tests for _truncate_fresh, _retruncate, and truncate_text_output.

Assumptions that mirror the production code:
- Every single line is shorter than DEFAULT_MAX_BYTES.
- Lines that exceed DEFAULT_MAX_BYTES are explicitly ignored / not fully read.
- _truncate_fresh always includes a continuation hint in the notice, so
  _retruncate can assume the hint is always present.

Run:
    cd tests/light && python test_truncate_text_output.py
"""

import re


from reme.memory.file_based.utils.file_utils import (
    TRUNCATION_NOTICE_MARKER,
    _truncate_fresh,
    _retruncate,
    truncate_text_output,
)

# ── helpers ──────────────────────────────────────────────────────────────────

LINE_BYTES = 20  # every test line is exactly this many bytes
ENC = "utf-8"


def make_line(i: int) -> str:
    """Return a line that is exactly LINE_BYTES bytes in UTF-8.

    Format: "L{i}" padded with underscores, terminated with "\\n".
    Works for i up to 999.
    """
    prefix = f"L{i}"
    return prefix + "_" * (LINE_BYTES - 1 - len(prefix)) + "\n"


def make_text(n: int, start: int = 1) -> str:
    """Build n lines starting from line number `start`."""
    return "".join(make_line(i) for i in range(start, start + n))


def content_of(result: str) -> str:
    """Return the portion before the truncation marker."""
    return result.split(TRUNCATION_NOTICE_MARKER)[0]


def notice_of(result: str) -> str:
    """Return the portion after the truncation marker, or '' if absent."""
    parts = result.split(TRUNCATION_NOTICE_MARKER, 1)
    return parts[1] if len(parts) > 1 else ""


def parse_next_line(result: str) -> int:
    """Extract 'start_line=X' from the notice, or -1 if absent."""
    m = re.search(r"start_line=(\d+) to read more", notice_of(result))
    return int(m.group(1)) if m else -1


def parse_covers_bytes(result: str) -> int:
    """Extract 'covers the next X bytes' from the notice."""
    m = re.search(r"covers the next (\d+) bytes", notice_of(result))
    return int(m.group(1)) if m else -1


def assert_eq(a, b, msg=""):
    assert a == b, f"{msg}: expected {b!r}, got {a!r}"


def assert_in(needle, haystack, msg=""):
    assert needle in haystack, f"{msg}: {needle!r} not found in {haystack!r}"


def assert_not_in(needle, haystack, msg=""):
    assert needle not in haystack, f"{msg}: {needle!r} unexpectedly found in {haystack!r}"


# ── _truncate_fresh tests ─────────────────────────────────────────────────────


def test_fresh_no_truncation_when_under_limit():
    text = make_text(3)  # 60 bytes
    result = _truncate_fresh(text, start_line=1, total_lines=3, max_bytes=100, file_path=None, encoding=ENC)
    assert_eq(result, text, "under limit: no change")
    assert_not_in(TRUNCATION_NOTICE_MARKER, result)


def test_fresh_no_truncation_at_exact_limit():
    text = make_text(3)  # 60 bytes
    result = _truncate_fresh(text, start_line=1, total_lines=3, max_bytes=60, file_path=None, encoding=ENC)
    assert_eq(result, text, "at exact limit: no change")
    assert_not_in(TRUNCATION_NOTICE_MARKER, result)


def test_fresh_mid_file_correct_next_line():
    # 10 lines × 20 bytes = 200 bytes; max_bytes=95
    # 95 bytes → 4 complete lines (80 bytes) + 15 bytes into line 5
    # newline_count=4, next_line=1+4=5, 5<=10 → read_from=5
    text = make_text(10)
    result = _truncate_fresh(text, start_line=1, total_lines=10, max_bytes=95, file_path=None, encoding=ENC)

    assert_in(TRUNCATION_NOTICE_MARKER, result)
    assert_eq(parse_next_line(result), 5, "next_line after mid-file cut")
    assert_eq(parse_covers_bytes(result), 95)
    assert_in("L4", content_of(result), "line 4 fully included")
    assert_not_in("L5\n", content_of(result), "line 5 not fully included")


def test_fresh_notice_contains_total_lines_and_start_line():
    text = make_text(10, start=3)
    result = _truncate_fresh(text, start_line=3, total_lines=12, max_bytes=95, file_path="/tmp/foo.txt", encoding=ENC)
    notice = notice_of(result)
    assert_in("contains 12 lines in total", notice)
    assert_in("starts at line 3", notice)
    assert_in("file_path=/tmp/foo.txt", notice)


def test_fresh_next_line_equals_total_lines_reads_from_last():
    # 5 lines × 20 = 100 bytes; max_bytes=85
    # 85 bytes → 4 complete lines (80 bytes) + 5 bytes into line 5
    # newline_count=4, next_line=1+4=5 = total_lines → read_from=5
    text = make_text(5)
    result = _truncate_fresh(text, start_line=1, total_lines=5, max_bytes=85, file_path=None, encoding=ENC)

    assert_in(TRUNCATION_NOTICE_MARKER, result)
    assert_eq(parse_next_line(result), 5, "should re-read the last line")


def test_fresh_next_line_overshoots_total_lines():
    # max_bytes=115 → 5 complete lines (100 bytes) + 15 bytes into line 6
    # newline_count=5, next_line=1+5=6 > total_lines(5), start_line(1) < 5
    # → read_from = total_lines = 5
    text = make_text(10)
    result = _truncate_fresh(text, start_line=1, total_lines=5, max_bytes=115, file_path=None, encoding=ENC)

    assert_in(TRUNCATION_NOTICE_MARKER, result)
    assert_eq(parse_next_line(result), 5, "overshoot: fall back to total_lines")


def test_fresh_long_first_line_skips_to_next_line():
    # Line 1 is 200 bytes (> max_bytes=100), no '\n' in truncated result.
    # newline_count=0, next_line=1+max(1,0)=2, 2<=5 → read_from=2
    long_line = "A" * 199 + "\n"  # 200 bytes
    text = long_line + make_text(4, start=2)
    result = _truncate_fresh(text, start_line=1, total_lines=5, max_bytes=100, file_path=None, encoding=ENC)

    assert_in(TRUNCATION_NOTICE_MARKER, result)
    assert_eq(parse_next_line(result), 2, "skip to line 2 after long line 1")
    assert_not_in("A" * 101, content_of(result))


def test_fresh_long_middle_line_skips_to_next_line():
    # Lines 1-2 normal (40 bytes); line 3 is 200 bytes; lines 4-5 normal.
    # max_bytes=100: fits lines 1-2 (40 bytes) + 60 bytes of line 3 (no '\n')
    # newline_count=2, next_line=1+2=3, 3<=5 → read_from=3
    text = make_text(2, start=1) + "B" * 199 + "\n" + make_text(2, start=4)
    result = _truncate_fresh(text, start_line=1, total_lines=5, max_bytes=100, file_path=None, encoding=ENC)

    assert_in(TRUNCATION_NOTICE_MARKER, result)
    assert_eq(parse_next_line(result), 3, "restart from long line 3")


def test_fresh_long_last_line_start_equals_total_no_notice():
    # start_line == total_lines, single line too long → unhandled case, no notice.
    # newline_count=0, next_line=5+1=6 > 5, start_line(5)==total_lines(5)
    long_line = "C" * 199 + "\n"  # 200 bytes
    result = _truncate_fresh(long_line, start_line=5, total_lines=5, max_bytes=100, file_path=None, encoding=ENC)

    assert_not_in(TRUNCATION_NOTICE_MARKER, result)
    assert_eq(len(result), 100, "only max_bytes bytes returned")


def test_fresh_long_last_line_arrived_from_previous_chunk_no_notice():
    long_line = "D" * 199 + "\n"
    result = _truncate_fresh(long_line, start_line=5, total_lines=5, max_bytes=80, file_path=None, encoding=ENC)

    assert_not_in(TRUNCATION_NOTICE_MARKER, result)


# ── _retruncate tests ─────────────────────────────────────────────────────────


def _first_pass(n_lines: int = 50, max_bytes: int = 490) -> str:
    """Produce a first-truncated text via _truncate_fresh."""
    return _truncate_fresh(
        make_text(n_lines),
        start_line=1,
        total_lines=n_lines,
        max_bytes=max_bytes,
        file_path="/file.txt",
        encoding=ENC,
    )


def test_retruncate_within_slack_returns_unchanged():
    # content ≈ 490 bytes; re-truncate with max_bytes=400.
    # 490 <= 400+100=500 → slack hit, return unchanged.
    pass1 = _first_pass(n_lines=50, max_bytes=490)
    result = _retruncate(pass1, max_bytes=400, encoding=ENC)
    assert_eq(result, pass1, "within slack: unchanged")


def test_retruncate_updates_byte_count_in_notice():
    pass1 = _first_pass(n_lines=50, max_bytes=490)
    result = _retruncate(pass1, max_bytes=195, encoding=ENC)

    assert_in(TRUNCATION_NOTICE_MARKER, result)
    assert_eq(parse_covers_bytes(result), 195, "notice byte count updated")


def test_retruncate_updates_next_line():
    # pass1 content ≈ 490 bytes (24 complete lines), start_line=1.
    # re-truncate at 195 bytes → 9 complete lines, next_line=1+9=10.
    pass1 = _first_pass(n_lines=50, max_bytes=490)
    result = _retruncate(pass1, max_bytes=195, encoding=ENC)

    assert_eq(parse_next_line(result), 10, "next_line updated to 10")


def test_retruncate_content_is_smaller():
    pass1 = _first_pass(n_lines=50, max_bytes=490)
    result = _retruncate(pass1, max_bytes=195, encoding=ENC)

    assert len(content_of(result)) < len(content_of(pass1)), "content should shrink"


def test_retruncate_missing_starts_at_line_returns_unchanged():
    # Notice missing "starts at line X" → return unchanged.
    # Use large content (600 bytes) to bypass the 100-byte slack.
    large_content = make_text(30)  # 600 bytes
    broken = (
        large_content + TRUNCATION_NOTICE_MARKER + "\nThe output above was truncated."
        "\nThe full content is saved to the file and contains 30 lines in total."
        "\nThis excerpt covers the next 600 bytes."  # no "starts at line X"
        "\nIf the current content is not enough, call `read_file` with file_path=/f.txt start_line=5 to read more."
    )
    result = _retruncate(broken, max_bytes=10, encoding=ENC)
    assert_eq(result, broken, "missing 'starts at line': unchanged")


def test_retruncate_missing_covers_next_bytes_returns_unchanged():
    # Notice has malformed "covers ??? bytes" → return unchanged.
    large_content = "X" * 500 + "\n"
    broken = (
        large_content + TRUNCATION_NOTICE_MARKER + "\nThe output above was truncated."
        "\nThe full content is saved to the file and contains 10 lines in total."
        "\nThis excerpt starts at line 1 and covers ??? bytes."
        "\nIf the current content is not enough, call `read_file` with file_path=/f.txt start_line=5 to read more."
    )
    result = _retruncate(broken, max_bytes=10, encoding=ENC)
    assert_eq(result, broken, "missing 'covers the next N bytes': unchanged")


# ── truncate_text_output dispatch / guard tests ───────────────────────────────


def test_dispatch_empty_string():
    result = truncate_text_output("", start_line=1, total_lines=0, max_bytes=10)
    assert_eq(result, "", "empty string bypassed")


def test_dispatch_max_bytes_zero():
    text = make_text(5)
    result = truncate_text_output(text, max_bytes=0)
    assert_eq(result, text, "max_bytes=0 bypassed")


def test_dispatch_routes_to_fresh_when_no_marker():
    text = make_text(10)
    result = truncate_text_output(text, start_line=1, total_lines=10, max_bytes=95)
    assert_in(TRUNCATION_NOTICE_MARKER, result)
    assert_eq(parse_next_line(result), 5)


def test_dispatch_routes_to_retruncate_when_marker_present():
    pass1 = _first_pass(n_lines=50, max_bytes=490)
    pass2 = truncate_text_output(pass1, max_bytes=195)
    assert_in(TRUNCATION_NOTICE_MARKER, pass2)
    assert_eq(parse_covers_bytes(pass2), 195)


# ── multi-pass integration tests ──────────────────────────────────────────────


def test_three_pass_decreasing_truncation():
    """Three successive truncations with shrinking max_bytes."""
    n_lines = 50
    text = make_text(n_lines)  # 1000 bytes

    pass1 = truncate_text_output(text, start_line=1, total_lines=n_lines, max_bytes=490, file_path="/f.txt")
    assert TRUNCATION_NOTICE_MARKER in pass1
    assert parse_covers_bytes(pass1) == 490
    assert parse_next_line(pass1) > 1

    # 490 > 195+100=295 → re-truncation proceeds
    pass2 = truncate_text_output(pass1, max_bytes=195)
    assert TRUNCATION_NOTICE_MARKER in pass2
    assert parse_covers_bytes(pass2) == 195
    assert parse_next_line(pass2) < parse_next_line(pass1), "next_line regresses"
    assert len(content_of(pass2)) < len(content_of(pass1))

    # content_of(pass2) ≈ 195 bytes; 195 > 90+100=190 → re-truncation proceeds
    pass3 = truncate_text_output(pass2, max_bytes=90)
    assert TRUNCATION_NOTICE_MARKER in pass3
    assert parse_covers_bytes(pass3) == 90
    assert parse_next_line(pass3) < parse_next_line(pass2), "next_line regresses further"
    assert len(content_of(pass3)) < len(content_of(pass2))


def test_three_pass_next_lines_are_consistent():
    """next_line values should monotonically decrease with each re-truncation."""
    n_lines = 50
    text = make_text(n_lines)

    pass1 = truncate_text_output(text, start_line=1, total_lines=n_lines, max_bytes=490, file_path="/f.txt")
    pass2 = truncate_text_output(pass1, max_bytes=195)
    pass3 = truncate_text_output(pass2, max_bytes=90)

    n1 = parse_next_line(pass1)
    n2 = parse_next_line(pass2)
    n3 = parse_next_line(pass3)

    assert n1 > n2 > n3 > 1, f"Expected n1 > n2 > n3 > 1, got {n1} > {n2} > {n3}"


# ── runner ────────────────────────────────────────────────────────────────────


def run_all():
    tests = [
        # _truncate_fresh
        test_fresh_no_truncation_when_under_limit,
        test_fresh_no_truncation_at_exact_limit,
        test_fresh_mid_file_correct_next_line,
        test_fresh_notice_contains_total_lines_and_start_line,
        test_fresh_next_line_equals_total_lines_reads_from_last,
        test_fresh_next_line_overshoots_total_lines,
        test_fresh_long_first_line_skips_to_next_line,
        test_fresh_long_middle_line_skips_to_next_line,
        test_fresh_long_last_line_start_equals_total_no_notice,
        test_fresh_long_last_line_arrived_from_previous_chunk_no_notice,
        # _retruncate
        test_retruncate_within_slack_returns_unchanged,
        test_retruncate_updates_byte_count_in_notice,
        test_retruncate_updates_next_line,
        test_retruncate_content_is_smaller,
        test_retruncate_missing_starts_at_line_returns_unchanged,
        test_retruncate_missing_covers_next_bytes_returns_unchanged,
        # truncate_text_output dispatch / guard
        test_dispatch_empty_string,
        test_dispatch_max_bytes_zero,
        test_dispatch_routes_to_fresh_when_no_marker,
        test_dispatch_routes_to_retruncate_when_marker_present,
        # multi-pass integration
        test_three_pass_decreasing_truncation,
        test_three_pass_next_lines_are_consistent,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests.")
    return failed == 0


if __name__ == "__main__":
    import sys

    ok = run_all()
    sys.exit(0 if ok else 1)
