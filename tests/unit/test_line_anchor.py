"""Tests for GitHub-style line anchor parsing."""

import pytest

from reme.utils import format_line_anchor, parse_line_anchor


def test_parse_supported_line_anchors():
    assert parse_line_anchor("L9") == [(9, 9)]
    assert parse_line_anchor("L9-L10") == [(9, 10)]
    assert parse_line_anchor("L9-L10,L15-L20") == [(9, 10), (15, 20)]


def test_parse_merges_overlapping_and_adjacent_ranges():
    ranges = parse_line_anchor("L9-L12,L11-L15,L16")
    assert ranges == [(9, 16)]
    assert format_line_anchor(ranges) == "L9-L16"


def test_non_line_heading_is_unchanged():
    assert parse_line_anchor("Introduction") is None


@pytest.mark.parametrize("anchor", ["L0", "L10-L9", "L9-Lx", "L9,"])
def test_invalid_line_anchor_rejected(anchor):
    with pytest.raises(ValueError):
        parse_line_anchor(anchor)
