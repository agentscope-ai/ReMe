"""Helpers for chunking history messages before vector insertion."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ....core.enumeration import Role
from ....core.schema import Message
from ....core.utils import format_messages


_TOKEN_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[A-Za-z0-9]+(?:[-_'][A-Za-z0-9]+)*|[^\s]",
)


@dataclass(slots=True)
class HistoryChunk:
    """A chunk of conversation history ready to be stored as one vector node."""

    index: int
    start_message_index: int
    end_message_index: int
    content: str
    token_count: int


def normalize_messages(messages: list[Message | dict]) -> list[Message]:
    """Convert raw dict messages to Message objects."""
    return [Message(**message) if isinstance(message, dict) else message for message in messages]


def estimate_mixed_tokens(text: str) -> int:
    """Estimate tokens while treating CJK chars and English words differently."""
    if not text:
        return 0
    return len(_TOKEN_PATTERN.findall(text))


def split_history_messages(
    messages: list[Message],
    *,
    chunk_strategy: str = "hybrid",
    turn_block_size: int = 3,
    max_chunk_tokens: int = 800,
) -> list[HistoryChunk]:
    """Split messages by dialogue turns and/or approximate token count.

    ``turn_block_size`` counts user-started dialogue rounds. ``max_chunk_tokens``
    uses a lightweight mixed Chinese/English tokenizer to keep chunks retrieval-sized
    without requiring an external tokenizer dependency.
    """
    if not messages:
        return []

    chunk_strategy = chunk_strategy.lower()
    if chunk_strategy not in {"turn", "token", "hybrid"}:
        raise ValueError("chunk_strategy must be one of: turn, token, hybrid")

    turn_block_size = max(1, turn_block_size)
    max_chunk_tokens = max(1, max_chunk_tokens)

    units = _build_turn_units(messages) if chunk_strategy in {"turn", "hybrid"} else _build_message_units(messages)
    chunks: list[HistoryChunk] = []
    current_units: list[tuple[int, int, list[Message]]] = []
    current_turns = 0

    def current_text() -> str:
        current_messages = [message for _, _, unit_messages in current_units for message in unit_messages]
        return format_messages(current_messages)

    def flush() -> None:
        nonlocal current_units, current_turns
        if not current_units:
            return
        text = current_text().strip()
        if text:
            chunks.append(
                HistoryChunk(
                    index=len(chunks),
                    start_message_index=current_units[0][0],
                    end_message_index=current_units[-1][1],
                    content=text,
                    token_count=estimate_mixed_tokens(text),
                ),
            )
        current_units = []
        current_turns = 0

    for unit in units:
        unit_text = format_messages(unit[2])
        unit_tokens = estimate_mixed_tokens(unit_text)
        next_turns = current_turns + 1
        should_split_by_turn = chunk_strategy in {"turn", "hybrid"} and next_turns > turn_block_size
        should_split_by_token = (
            chunk_strategy in {"token", "hybrid"}
            and current_units
            and estimate_mixed_tokens(current_text()) + unit_tokens > max_chunk_tokens
        )

        if should_split_by_turn or should_split_by_token:
            flush()

        current_units.append(unit)
        current_turns += 1

        if chunk_strategy == "token" and unit_tokens >= max_chunk_tokens:
            flush()

    flush()
    return chunks


def _build_message_units(messages: list[Message]) -> list[tuple[int, int, list[Message]]]:
    """Treat each message as a split unit for token-only chunking."""
    return [(index, index, [message]) for index, message in enumerate(messages)]


def _build_turn_units(messages: list[Message]) -> list[tuple[int, int, list[Message]]]:
    """Group messages into user-started dialogue turns."""
    units: list[tuple[int, int, list[Message]]] = []
    current_start = 0
    current_messages: list[Message] = []

    for index, message in enumerate(messages):
        role = message.role.value if isinstance(message.role, Role) else str(message.role)
        if role == Role.USER.value and current_messages:
            units.append((current_start, index - 1, current_messages))
            current_start = index
            current_messages = []
        elif not current_messages:
            current_start = index
        current_messages.append(message)

    if current_messages:
        units.append((current_start, len(messages) - 1, current_messages))

    return units
