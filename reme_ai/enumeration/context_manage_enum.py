"""Context management enumeration module.

This module provides enumerations for context management strategies in the ReMe system.
"""

from enum import Enum


class ContextManageEnum(str, Enum):
    """
    An enumeration representing context management strategies.

    Members:
        - COMPACT: Represents the compact context management strategy.
        - COMPRESS: Represents the compress context management strategy.
        - AUTO: Represents the automatic context management strategy.
    """

    COMPACT = "compact"
    COMPRESS = "compress"
    AUTO = "auto"
