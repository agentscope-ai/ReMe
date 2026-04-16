"""Chunk enumeration module.

Defines the types of data chunks used in streaming responses.
"""

from enum import Enum


class ChunkEnum(str, Enum):
    """Enumeration of possible chunk categories for stream processing.

    This enum defines the various types of chunks that can be transmitted
    during a streaming response from an LLM or agent system.
    """

    THINK = "think"

    CONTENT = "content"

    TOOL_CALL = "tool_call"

    TOOL_RESULT = "tool_result"

    USAGE = "usage"

    ERROR = "error"

    DONE = "done"
