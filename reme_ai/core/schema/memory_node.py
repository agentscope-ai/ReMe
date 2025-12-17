"""Memory schema module for the ReMe AI system.

This module defines the MemoryNode class for storing and retrieving
memories in the ReMe system.
"""

import datetime
import hashlib
from typing import Any, Self
import json
from flowllm.core.schema import VectorNode
from pydantic import BaseModel, Field, model_validator

from ..enumeration import MemoryType

# Length of the memory ID (first N characters of SHA-256 hash)
MEMORY_ID_LENGTH: int = 16


def _get_current_timestamp() -> str:
    """Get current timestamp in YYYY-MM-DD HH:MM:SS format.

    Example:
        >>> _get_current_timestamp()
        '2024-01-15 14:30:45'

    Returns:
        str: Current timestamp string in format 'YYYY-MM-DD HH:MM:SS'.
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class MemoryNode(BaseModel):
    """Memory node for storing memories in the ReMe system.

    Attributes:
        workspace_id: Workspace identifier this memory belongs to.
        memory_id: Unique identifier, auto-generated from content hash.
        memory_type: Type of memory (e.g., SUMMARY, PERSONAL).
        memory_target: Target or topic this memory relates to.
        when_to_use: Condition description for vector retrieval.
        content: Actual memory content.
        ref_memory_id: Reference to related raw history memory.
        time_created: Creation timestamp.
        time_modified: Last modification timestamp.
        author: Author or source of this memory.
        score: Relevance or importance score.
        metadata: Additional metadata for extensibility.
    """

    workspace_id: str = Field(default="", description="Workspace identifier")
    memory_id: str = Field(default="", description="Unique memory identifier")

    memory_type: MemoryType = Field(default=..., description="Type of memory")
    memory_target: str = Field(default="", description="Target or topic of the memory")
    when_to_use: str = Field(default="", description="Condition description for vector retrieval")
    content: str = Field(default="", description="Actual memory content")
    ref_memory_id: str = Field(default="", description="Reference to related raw history memory ID")

    time_created: str = Field(default_factory=_get_current_timestamp, description="Creation timestamp")
    time_modified: str = Field(default_factory=_get_current_timestamp, description="Last modification timestamp")
    author: str = Field(default="", description="Author or source of the memory")
    score: float = Field(default=0, description="Relevance or importance score")

    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def _update_modified_time(self) -> Self:
        """Update time_modified to current timestamp.

        Returns:
            Self: Returns self for method chaining.
        """
        self.time_modified = _get_current_timestamp()
        return self

    def _update_memory_id(self) -> Self:
        """Generate memory_id from SHA-256 hash of content.

        Takes the first MEMORY_ID_LENGTH characters of the hash.

        Returns:
            Self: Returns self for method chaining.
        """
        if not self.content:
            return self

        hash_obj = hashlib.sha256(self.content.encode('utf-8'))
        hex_dig = hash_obj.hexdigest()
        self.memory_id = hex_dig[:MEMORY_ID_LENGTH]
        return self

    @model_validator(mode='after')
    def _update_after_init(self) -> Self:
        """Post-initialization validator.

        Auto-generates memory_id from content if not provided.

        Returns:
            Self: Returns self for method chaining.
        """
        if not self.memory_id:
            self._update_memory_id()
        return self

    def __setattr__(self, name: str, value):
        """Auto-update timestamps and memory_id when content or when_to_use changes.

        Args:
            name: Attribute name being set.
            value: New value for the attribute.
        """
        should_update: bool = name in ("when_to_use", "content") and getattr(self, name, None) != value
        super().__setattr__(name, value)
        if should_update:
            self._update_modified_time()
            if name == "content":
                self._update_memory_id()

    def to_vector_node(self) -> VectorNode:
        """Convert to VectorNode for vector storage.

        When when_to_use is set, use it as vector content and store content in metadata.
        When when_to_use is empty, use content as vector content directly.

        Returns:
            VectorNode: Vector node representation of this memory.
        """
        # Build base metadata (shared fields)
        metadata: dict[str, Any] = {
            "memory_type": self.memory_type.value,
            "memory_target": self.memory_target,
            "ref_memory_id": self.ref_memory_id,
            "time_created": self.time_created,
            "time_modified": self.time_modified,
            "author": self.author,
            "score": self.score,
            **self.metadata,
        }

        if self.when_to_use:
            # Use when_to_use for vector embedding, store content in metadata
            vector_content = self.when_to_use
            metadata["content"] = self.content
        else:
            # Use content directly for vector embedding
            vector_content = self.content

        return VectorNode(
            unique_id=self.memory_id,
            workspace_id=self.workspace_id,
            content=vector_content,
            metadata=metadata,
        )

    def format_memory(self) -> str:
        """Format memory as human-readable string.

        Returns:
            str: Formatted string with when_to_use, content, and ref_memory_id.
        """
        parts: list[str] = [
            f"memory_id={self.memory_id}"
            f"modified_time={self.time_modified}"
        ]

        if self.when_to_use:
            parts.append(self.when_to_use)

        if self.content:
            parts.append(self.content)

        if self.metadata:
            parts.append(f"metadata={json.dumps(self.metadata, ensure_ascii=False)}")

        if self.ref_memory_id:
            parts.append(f"history_memory.ref_memory_id={self.ref_memory_id}")

        return " ".join(parts)

    @classmethod
    def from_vector_node(cls, node: VectorNode) -> Self:
        """Reconstruct MemoryNode from VectorNode.

        Reverses the to_vector_node conversion:
        - If metadata contains 'content': node.content -> when_to_use, metadata['content'] -> content
        - Otherwise: node.content -> content, when_to_use remains empty

        Args:
            node: VectorNode containing memory data.

        Returns:
            Self: Reconstructed MemoryNode instance.

        Raises:
            ValueError: If memory_type in metadata is invalid.
        """
        metadata = node.metadata.copy()
        memory_type_str = metadata.pop("memory_type", None)

        try:
            memory_type: MemoryType = MemoryType(memory_type_str)
        except ValueError as e:
            raise ValueError(
                f"Invalid memory_type '{memory_type_str}' in VectorNode metadata. "
                f"Valid types are: {[t.value for t in MemoryType]}"
            ) from e

        # Restore when_to_use and content based on metadata structure
        if "content" in metadata:
            # Original had when_to_use set
            when_to_use = node.content
            content = metadata.pop("content", "")
        else:
            # Original had empty when_to_use
            when_to_use = ""
            content = node.content

        return cls(
            workspace_id=node.workspace_id,
            memory_id=node.unique_id,
            memory_type=memory_type,
            memory_target=metadata.pop("memory_target", ""),
            when_to_use=when_to_use,
            content=content,
            ref_memory_id=metadata.pop("ref_memory_id", ""),
            time_created=metadata.pop("time_created", ""),
            time_modified=metadata.pop("time_modified", ""),
            author=metadata.pop("author", ""),
            score=metadata.pop("score", 0),
            metadata=metadata,
        )
