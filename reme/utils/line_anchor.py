"""Parse GitHub-style 1-based line anchors used by local file links."""

import re

_LINE_ANCHOR_RE = re.compile(r"L[0-9]+(?:-L[0-9]+)?(?:,L[0-9]+(?:-L[0-9]+)?)*")


def parse_line_anchor(anchor: str | None) -> list[tuple[int, int]] | None:
    """Return normalized inclusive ranges, or ``None`` for a non-line anchor.

    Supported forms are ``L9``, ``L9-L10`` and
    ``L9-L10,L15-L20``. Overlapping and adjacent ranges are merged.
    Anchors beginning with ``L<digit>`` are treated as line anchors and raise
    ``ValueError`` when malformed, zero-based, or reversed.
    """
    if not anchor or not re.match(r"L[0-9]", anchor):
        return None
    if not _LINE_ANCHOR_RE.fullmatch(anchor):
        raise ValueError(f"invalid line anchor: #{anchor}")

    ranges: list[tuple[int, int]] = []
    for item in anchor.split(","):
        start_text, separator, end_text = item.partition("-L")
        start = int(start_text[1:])
        end = int(end_text) if separator else start
        if start < 1 or end < 1:
            raise ValueError("line numbers must be at least 1")
        if start > end:
            raise ValueError(f"line range start ({start}) exceeds end ({end})")
        ranges.append((start, end))

    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def format_line_anchor(ranges: list[tuple[int, int]]) -> str:
    """Render normalized ranges without the leading ``#``."""
    return ",".join(f"L{start}" if start == end else f"L{start}-L{end}" for start, end in ranges)
