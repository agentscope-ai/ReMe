"""Memory type enumeration for the three-layer memory architecture."""

from enum import Enum


class MemoryType(str, Enum):
    """
    Three-layer memory architecture for agent memory management.

    Layer 1 - High-level Abstraction Memory:
        - IDENTITY: Self-cognition (identity, personality, current state)
        - PERSONAL: Person-specific memory (preferences and context about specific individuals)
        - PROCEDURAL: Procedural memory (how-to knowledge, e.g., 4 steps to write financial reports)
        - TOOL: Tool memory (tool usage patterns, success rates, token consumption, latency)

    Layer 2 - Summary Memory (Compressed): Summarized digest of raw message history
    Layer 3 - History Memory (Raw): Raw message history
    """

    IDENTITY = "identity"
    PERSONAL = "personal"
    PROCEDURAL = "procedural"
    TOOL = "tool"
    
    SUMMARY = "summary"
    
    HISTORY = "history"