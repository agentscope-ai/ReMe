"""Chunk enumeration module."""

from enum import Enum


class ChunkEnum(str, Enum):
    """Enumeration of possible chunk categories for stream processing."""

    THINK = "think"

    CONTENT = "content"

    TOOL_CALL = "tool_call"

    TOOL_RESULT = "tool_result"

    USAGE = "usage"

    ERROR = "error"

    DONE = "done"
