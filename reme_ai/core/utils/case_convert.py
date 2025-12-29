"""Case converter for string naming conventions."""

import re


# Special words that need custom handling during conversion
_SPECIAL_WORDS = ["LLM", "API", "URL", "HTTP", "JSON", "XML", "AI", "MCP"]
_SPECIAL_WORD_MAP = {word.lower(): word for word in _SPECIAL_WORDS}


def camel_to_snake(content: str) -> str:
    """Convert camelCase to snake_case.
    
    Args:
        content: String in camelCase format.
        
    Returns:
        String in snake_case format.
        
    Examples:
        >>> camel_to_snake("OpenAILLMClient")
        'open_ai_llm_client'
        >>> camel_to_snake("HTTPAPIClient")
        'http_api_client'
    """
    # Replace special words with capitalized versions
    result = content
    for word in _SPECIAL_WORDS:
        # Replace uppercase special words with capitalized version
        result = result.replace(word, word.capitalize())

    # Insert underscores before capital letters
    snake_str = re.sub(r"(?<!^)(?=[A-Z])", "_", result).lower()
    return snake_str


def snake_to_camel(content: str) -> str:
    """Convert snake_case to camelCase.
    
    Args:
        content: String in snake_case format.
        
    Returns:
        String in camelCase format.
        
    Examples:
        >>> snake_to_camel("open_ai_llm_client")
        'OpenAILLMClient'
        >>> snake_to_camel("http_api_client")
        'HTTPAPIClient'
    """
    # Split by underscore and capitalize each part
    parts = content.split("_")
    camel_str = "".join(x.capitalize() for x in parts)

    # Restore special words to their original form
    for lower_word, original_word in _SPECIAL_WORD_MAP.items():
        capitalized = lower_word.capitalize()
        camel_str = camel_str.replace(capitalized, original_word)

    return camel_str
