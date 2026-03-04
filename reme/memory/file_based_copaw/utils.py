"""Utility functions for working with text."""

import logging

from agentscope.token import HuggingFaceTokenCounter

logger = logging.getLogger(__name__)

# Unique marker for truncated text
TRUNCATION_MARKER_START = "<<<TRUNCATED>>>"
TRUNCATION_MARKER_END = "<<<END_TRUNCATED>>>"


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max length, keeping head and tail portions.

    Args:
        text: The text to truncate
        max_length: Maximum allowed length

    Returns:
        Truncated text with unique markers indicating truncation
    """
    text = str(text) if text else ""
    if not text:
        return text

    if len(text) <= max_length:
        return text

    half_length = max_length // 2
    truncated_chars = len(text) - max_length
    logger.debug(
        "Text truncated: original %d chars, kept head %d + tail %d, removed %d chars.",
        len(text),
        half_length,
        half_length,
        truncated_chars,
    )
    return (
        f"{text[:half_length]}\n\n{TRUNCATION_MARKER_START} "
        f"({truncated_chars} characters omitted) "
        f"{TRUNCATION_MARKER_END}\n\n{text[-half_length:]}"
    )


def is_truncated(text: str) -> bool:
    """Check if the text has been truncated (contains truncation markers).

    Args:
        text: The text to check

    Returns:
        bool: True if text contains truncation markers, False otherwise
    """
    if not text:
        return False
    return TRUNCATION_MARKER_START in text and TRUNCATION_MARKER_END in text


def _extract_text_from_messages(messages: list[dict]) -> str:
    """Extract text content from messages and concatenate into a string.

    Handles various message formats:
    - Simple string content: {"role": "user", "content": "hello"}
    - List content with text blocks:
      {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    - List content with tool_result blocks:
      {"role": "user", "content": [{"type": "tool_result", "output": "..."}]}

    Args:
        messages: List of message dictionaries in chat format.

    Returns:
        str: Concatenated text content from all messages.
    """
    parts = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "")
                    if block_type == "tool_result":
                        output = block.get("output", "")
                        if isinstance(output, str) and output:
                            parts.append(output)
                        elif isinstance(output, list):
                            for sub in output:
                                if isinstance(sub, dict):
                                    sub_text = sub.get("text") or sub.get("content", "")
                                    if sub_text:
                                        parts.append(str(sub_text))
                    else:
                        text = block.get("text") or block.get("content", "")
                        if text:
                            parts.append(str(text))
                elif isinstance(block, str):
                    parts.append(block)
    return "\n".join(parts)


def safe_count_message_tokens(
    token_counter: HuggingFaceTokenCounter,
    messages: list[dict],
) -> int:
    """Safely count tokens in messages with fallback estimation.

    This is a wrapper around count_message_tokens that catches exceptions
    and falls back to a character-based estimation (len // 4) if the
    tokenizer fails.

    Args:
        token_counter: Token counter instance.
        messages: List of message dictionaries in chat format.

    Returns:
        int: The estimated number of tokens in the messages.
    """
    try:
        text = _extract_text_from_messages(messages)
        token_ids = token_counter.tokenizer.encode(text)
        token_count = len(token_ids)
        return token_count

    except Exception as e:
        # Fallback to character-based estimation
        text = _extract_text_from_messages(messages)
        estimated_tokens = len(text) // 4
        logger.warning(
            "Failed to count tokens: %s, using estimated_tokens=%d",
            e,
            estimated_tokens,
        )
        return estimated_tokens


def safe_count_str_tokens(
    token_counter: HuggingFaceTokenCounter,
    text: str,
) -> int:
    """Safely count tokens in a string with fallback estimation.

    Uses the tokenizer to count tokens in the given text. If the tokenizer
    fails, falls back to a character-based estimation (len // 4).

    Args:
        token_counter: Token counter instance.
        text: The string to count tokens for.

    Returns:
        int: The estimated number of tokens in the string.
    """
    try:
        token_ids = token_counter.tokenizer.encode(text)
        token_count = len(token_ids)
        return token_count
    except Exception as e:
        # Fallback to character-based estimation
        estimated_tokens = len(text) // 4
        logger.warning(
            "Failed to count string tokens: %s, using estimated_tokens=%d",
            e,
            estimated_tokens,
        )
        return estimated_tokens


def _get_block_tokens(  # pylint: disable=too-many-return-statements
    block: dict,
    block_type: str,
    token_counter: HuggingFaceTokenCounter,
) -> tuple[int, str]:
    """Get token count and content string for different block types.

    Args:
        block: The content block dict
        block_type: The type of the block

    Returns:
        Tuple of (token count, content string)
    """
    if block_type == "text":
        text = block.get("text", "")
        return (safe_count_str_tokens(token_counter, text), text) if text else (0, "")

    if block_type == "thinking":
        thinking = block.get("thinking", "")
        return (safe_count_str_tokens(token_counter, thinking), thinking) if thinking else (0, "")

    if block_type == "tool_use":
        # Count input dict and raw_input string
        input_dict = block.get("input", {})
        raw_input = block.get("raw_input", "")
        input_str = str(input_dict) if input_dict else ""
        total = input_str + raw_input
        return (safe_count_str_tokens(token_counter, total), total) if total else (0, "")

    if block_type == "tool_result":
        output = block.get("output")
        if isinstance(output, str):
            return (safe_count_str_tokens(token_counter, output), output) if output else (0, "")

        if isinstance(output, list):
            # Recursively count tokens in nested blocks
            total_tokens = 0
            total_str = ""
            for item in output:
                if isinstance(item, dict):
                    item_type = item.get("type", "unknown")
                    item_tokens, item_str = _get_block_tokens(item, item_type, token_counter)
                    total_tokens += item_tokens
                    total_str += item_str
            return total_tokens, total_str
        return 0, ""

    if block_type in ("image", "audio", "video"):
        # For media blocks, count the URL or indicate base64 size
        source = block.get("source", {})
        if source.get("type") == "url":
            url = source.get("url", "")
            return safe_count_str_tokens(token_counter, url), url
        if source.get("type") == "base64":
            # Base64 data can be large, return approximate token count
            data = source.get("data", "")
            return (len(data) // 4, "[base64]") if data else (0, "")
        return 0, ""

    return 0, ""
