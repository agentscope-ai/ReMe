"""Case conversion utility for PascalCase, camelCase, and snake_case.

Provides bidirectional conversion between snake_case and PascalCase,
with special handling for common acronyms (LLM, API, URL, etc.).
"""

import re

# Acronyms that should remain uppercase in Pascal/camelCase
_ACRONYMS = frozenset({"LLM", "API", "URL", "HTTP", "JSON", "XML", "AI", "MCP"})
_ACRONYM_MAP = {word.lower(): word for word in _ACRONYMS}


def camel_to_snake(content: str) -> str:
    """Convert PascalCase or camelCase to snake_case.

    Handles acronyms correctly by normalizing them before conversion.
    For example, "OpenAILLMClient" becomes "open_ai_llm_client".

    Args:
        content: A string in PascalCase or camelCase format.

    Returns:
        The converted snake_case string.
    """
    # Normalize acronyms to title case (e.g., LLM -> Llm) to assist regex splitting
    for word in _ACRONYMS:
        content = content.replace(word, word.capitalize())

    # Insert underscores between case transitions and convert to lowercase
    return re.sub(r"(?<!^)(?=[A-Z])", "_", content).lower()


def snake_to_camel(content: str) -> str:
    """Convert snake_case to PascalCase.

    Preserves defined acronyms in uppercase form.
    For example, "open_ai_llm_client" becomes "OpenAILLMClient".

    Args:
        content: A string in snake_case format.

    Returns:
        The converted PascalCase string with acronyms preserved.
    """
    return "".join(_ACRONYM_MAP.get(part.lower(), part.capitalize()) for part in content.split("_") if part)
