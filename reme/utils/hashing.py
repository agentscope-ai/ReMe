"""Stable text digests."""

import hashlib


def hash_text(text: str, encoding: str = "utf-8") -> str:
    """Return SHA-256 hex digest of text."""
    return hashlib.sha256(text.encode(encoding)).hexdigest()
