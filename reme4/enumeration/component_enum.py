"""Component enumeration module."""

from enum import Enum


class ComponentEnum(str, Enum):
    """Enumeration of component types for dependency injection and registration."""

    BASE = "base"

    LLM = "llm"

    EMBEDDING = "embedding"

    EMBEDDING_STORE = "embedding_store"

    FILE_PARSER = "file_parser"

    FILE_STORE = "file_store"

    FILE_GRAPH = "file_graph"

    FILE_CATALOG = "file_catalog"

    KEYWORD_INDEX = "keyword_index"

    SERVICE = "service"

    CLIENT = "client"

    STEP = "step"

    JOB = "job"

    TOKENIZER = "tokenizer"
