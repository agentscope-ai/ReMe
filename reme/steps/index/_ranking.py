"""Ranking parameters shared by index search steps."""

from typing import Final

RRF_K: Final = 60
_MAX_CANDIDATES: Final = 200


def candidate_count(limit: int, multiplier: float) -> int:
    """Bound the number of candidates fetched before filtering and ranking."""
    return min(_MAX_CANDIDATES, max(1, int(limit * multiplier)))
