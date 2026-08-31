"""Token count estimation."""


def estimate_token_count(
    text: str,
    estimate_divisor: float = 4,
    encoding: str = "utf-8",
) -> int:
    """Estimate the number of tokens in *text* by byte length."""
    return int(len(text.encode(encoding)) / estimate_divisor + 0.5)
