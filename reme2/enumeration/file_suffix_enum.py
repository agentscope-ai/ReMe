"""File suffix enumeration module.

Defines the file suffixes that can be handled by file parsers.
"""

from enum import Enum


class FileSuffixEnum(str, Enum):
    """Enumeration of supported file suffixes.

    Each value represents a file extension that a file parser can handle.
    """

    MD = ".md"

    MARKDOWN = ".markdown"

    TXT = ".txt"
