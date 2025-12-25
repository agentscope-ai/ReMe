from enum import Enum


class RegistryEnum(str, Enum):
    LLM = "llm"

    EMBEDDING_MODEL = "embedding_model"

    VECTOR_STORE = "vector_store"

    OP = "op"

    FLOW = "flow"

    SERVICE = "service"

    TOKEN_COUNTER = "token_counter"
