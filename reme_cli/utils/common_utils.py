import hashlib


def hash_text(text: str, encoding: str = "utf-8") -> str:
    """Generate SHA-256 hash of text content.

    Args:
        text: Input text to hash
        encoding: Encoding of the text (default: "utf-8")

    Returns:
        Hexadecimal representation of the SHA-256 hash
    """
    return hashlib.sha256(text.encode(encoding)).hexdigest()
