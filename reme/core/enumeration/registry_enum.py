"""Defines the registry categories for core components of the system."""

from enum import Enum


class RegistryEnum(str, Enum):
    """Enumeration of component types registered within the application lifecycle."""

    # Large Language Model interfaces
    LLM = "llm"

    # Models used for generating vector embeddings
    EMBEDDING_MODEL = "embedding_model"

    # Databases or storage systems for vector search
    VECTOR_STORE = "vector_store"

    # Databases or storage systems for long-term file storage
    FILE_STORE = "file_store"

    # Atomic operations or functional units
    OP = "op"

    # Orchestrated sequences of operations or workflows
    FLOW = "flow"

    # External APIs or shared internal services
    SERVICE = "service"

    # Utilities for tracking and limiting token consumption
    TOKEN_COUNTER = "token_counter"
