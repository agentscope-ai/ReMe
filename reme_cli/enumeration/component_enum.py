"""Component enumeration module.

Defines the types of components that can be registered and used in the application.
"""

from enum import Enum


class ComponentEnum(str, Enum):
    """Enumeration of component types for dependency injection and registration.

    This enum defines the various component categories that can be registered
    in the application's component container. Each component type represents
    a specific role or functionality within the system.
    """

    BASE = "base"

    AS_LLM = "as_llm"

    AS_LLM_FORMATTER = "as_llm_formatter"

    EMBEDDING_MODEL = "embedding_model"

    FILE_STORE = "file_store"

    FILE_WATCHER = "file_watcher"

    SERVICE = "service"

    STEP = "step"

    JOB = "job"
