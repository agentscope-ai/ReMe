"""Utility functions for truncating long text strings."""

from .std_logger import get_logger

logger = get_logger()

TRUNCATION_MARKER_START = "<<<TRUNCATED>>>"
TRUNCATION_MARKER_END = "<<<END_TRUNCATED>>>"


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max length, keeping head and tail portions.

    Args:
        text: The text to truncate
        max_length: Maximum allowed length

    Returns:
        Truncated text with unique markers indicating truncation
    """
    text = str(text) if text else ""
    if not text:
        return text

    if len(text) <= max_length:
        return text

    half_length = max_length // 2
    truncated_chars = len(text) - max_length
    logger.debug(
        "Text truncated: original %d chars, kept head %d + tail %d, removed %d chars.",
        len(text),
        half_length,
        half_length,
        truncated_chars,
    )
    return (
        f"{text[:half_length]}\n\n{TRUNCATION_MARKER_START} "
        f"({truncated_chars} characters omitted) "
        f"{TRUNCATION_MARKER_END}\n\n{text[-half_length:]}"
    )


def truncate_text_head(text: str, max_length: int) -> str:
    """Truncate text from the beginning, keeping only the head portion.

    Args:
        text: The text to truncate
        max_length: Maximum allowed length

    Returns:
        Truncated text with marker indicating truncation at the end
    """
    text = str(text) if text else ""
    if not text:
        return text

    if len(text) <= max_length:
        return text

    truncated_chars = len(text) - max_length
    logger.debug(
        "Text truncated from head: original %d chars, kept %d, removed %d chars from tail.",
        len(text),
        max_length,
        truncated_chars,
    )
    return f"{text[:max_length]}{TRUNCATION_MARKER_START}"


def is_truncated(text: str) -> bool:
    """Check if the text has been truncated (contains truncation marker).

    Args:
        text: The text to check

    Returns:
        bool: True if text contains truncation marker, False otherwise
    """
    if not text:
        return False
    return TRUNCATION_MARKER_START in text
