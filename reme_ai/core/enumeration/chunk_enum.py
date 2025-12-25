from enum import Enum


class ChunkEnum(str, Enum):
    THINK = "think"

    ANSWER = "answer"

    TOOL = "tool"

    USAGE = "usage"

    ERROR = "error"

    DONE = "done"
