"""Component enumeration module."""

from enum import Enum


class ComponentEnum(str, Enum):
    """Enumeration of component types for dependency injection and registration."""

    BASE = "base"

    AS_LLM = "as_llm"

    AS_LLM_FORMATTER = "as_llm_formatter"

    AS_TOKEN_COUNTER = "as_token_counter"

    EMBEDDING_MODEL = "embedding_model"

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
