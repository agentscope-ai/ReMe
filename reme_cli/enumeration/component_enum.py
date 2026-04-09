
from enum import Enum


class ComponentEnum(str, Enum):

    BASE = "base"

    EMBEDDING_MODEL = "embedding_model"

    AS_LLM = "as_llm"

    AS_LLM_FORMATTER = "as_llm_formatter"

    FILE_STORE = "file_store"

    FILE_WATCHER = "file_watcher"
