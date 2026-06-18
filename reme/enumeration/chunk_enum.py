"""Chunk enumeration module."""

from enum import Enum


class ChunkEnum(str, Enum):
    """Enumeration of possible chunk categories for stream processing.

    Covers both AgentScope and Claude Code SDK streaming protocols:

    AgentScope events -> ChunkEnum mapping:
        ReplyStartEvent          -> REPLY_START
        ReplyEndEvent            -> REPLY_END
        TextBlockStart/Delta/End -> CONTENT
        ThinkingBlockStart/Delta/End -> THINK
        DataBlockStart/Delta/End -> DATA
        ToolCallStart/Delta/End  -> TOOL_CALL
        ToolResultStart/TextDelta/DataDelta/End -> TOOL_RESULT
        ModelCallEndEvent        -> USAGE
        ExceedMaxItersEvent      -> ERROR

    Claude Code SDK events -> ChunkEnum mapping:
        message_start   -> REPLY_START
        message_stop    -> REPLY_END
        content_block_start/delta/stop (text)      -> CONTENT
        content_block_start/delta/stop (thinking)  -> THINK
        content_block_start/delta/stop (tool_use)  -> TOOL_CALL
        ToolResultBlock -> TOOL_RESULT
        ResultMessage   -> USAGE + DONE
        ResultMessage.is_error -> ERROR
    """

    # Lifecycle markers
    REPLY_START = "reply_start"
    REPLY_END = "reply_end"

    # Content types
    THINK = "think"
    CONTENT = "content"
    DATA = "data"

    # Tool interaction
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

    # Metadata & terminal
    USAGE = "usage"
    ERROR = "error"
    DONE = "done"
