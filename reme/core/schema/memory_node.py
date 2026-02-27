"""Memory schema module for the ReMe AI system.

This module defines the MemoryNode class for storing and retrieving
memories in the ReMe system.
"""

import datetime
import hashlib
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .vector_node import VectorNode
from ..enumeration import MemoryType


def get_now_time() -> str:
    """Get current timestamp in YYYY-MM-DD HH:MM:SS format.

    Returns:
        str: Current timestamp string in format 'YYYY-MM-DD HH:MM:SS'.
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Length of the memory ID (first N characters of SHA-256 hash)
MEMORY_ID_LENGTH: int = 16


class MemoryNode(BaseModel):
    """Memory node for storing memories in the ReMe system.

    Attributes:
        memory_id: Unique identifier, auto-generated from content hash.
        memory_type: Type of memory (e.g., SUMMARY, PERSONAL).
        memory_target: Target or topic this memory relates to.
        when_to_use: Condition description for vector retrieval.
        content: Actual memory content.
        message_time: Time of the message that generated this memory.
        ref_memory_id: Reference to related raw history memory.
        time_created: Creation timestamp.
        time_modified: Last modification timestamp.
        author: Author or source of this memory.
        score: Relevance or importance score.
        vector: Vector embedding of the memory content.
        metadata: Additional metadata for extensibility.
    """

    memory_id: str = Field(default="", description="Unique memory identifier")
    memory_type: MemoryType = Field(default=..., description="Type of memory")
    memory_target: str = Field(default="", description="Target or topic of the memory")
    when_to_use: str = Field(default="", description="Condition description for vector retrieval")
    content: str = Field(default="", description="Actual memory content")
    message_time: str = Field(default="", description="Time of the message that generated this memory")
    ref_memory_id: str = Field(default="", description="Reference to related raw history memory ID")

    time_created: str = Field(default_factory=get_now_time, description="Creation timestamp")
    time_modified: str = Field(default_factory=get_now_time, description="Last modification timestamp")
    author: str = Field(default="", description="Author or source of the memory")
    score: float = Field(default=0, description="Relevance or importance score")

    vector: list[float] | None = Field(default=None, description="Vector embedding of the memory content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def _update_modified_time(self) -> "MemoryNode":
        """Update time_modified to current timestamp.

        Returns:
            Self: Returns self for method chaining.
        """
        self.time_modified = get_now_time()
        return self

    def _update_memory_id(self) -> "MemoryNode":
        """Generate memory_id from SHA-256 hash of content.

        Takes the first MEMORY_ID_LENGTH characters of the hash.

        Returns:
            Self: Returns self for method chaining.
        """
        if not self.content:
            return self

        hash_obj = hashlib.sha256(self.content.encode("utf-8"))
        hex_dig = hash_obj.hexdigest()
        self.memory_id = hex_dig[:MEMORY_ID_LENGTH]
        return self

    @model_validator(mode="after")
    def _update_after_init(self) -> "MemoryNode":
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
        safe_metadata: dict[str, str | bool | int | float] = {}
        for key, value in self.metadata.items():
            if isinstance(value, (str, bool, int, float)):
                safe_metadata[key] = value
            elif isinstance(value, (list, tuple, set)):
                safe_metadata[key] = ",".join(str(v) for v in value)
            else:
                safe_metadata[key] = str(value)

        # Build base metadata (shared fields)
        metadata: dict[str, Any] = {
            "memory_type": self.memory_type.value,
            "memory_target": self.memory_target,
            "message_time": self.message_time,
            "ref_memory_id": self.ref_memory_id,
            "time_created": self.time_created,
            "time_modified": self.time_modified,
            "author": self.author,
            "score": self.score,
            **safe_metadata,
        }

        if self.when_to_use:
            # Use when_to_use for vector embedding, store content in metadata
            vector_content = self.when_to_use
            metadata["content"] = self.content
        else:
            # Use content directly for vector embedding
            vector_content = self.content

        return VectorNode(
            vector_id=self.memory_id,
            content=vector_content,
            vector=self.vector,
            metadata=metadata,
        )

    def format(
        self,
        include_memory_id: bool = True,
        include_when_to_use: bool = True,
        include_content: bool = True,
        include_message_time: bool = True,
        ref_memory_id_key: str = "",
    ) -> str:
        """Format memory node as string with configurable fields."""
        line = ""

        if include_memory_id and self.memory_id:
            line += f"memory_id={self.memory_id} "

        if include_message_time and self.message_time:
            line += f"[{self.message_time}] "

        if include_when_to_use and self.when_to_use:
            line += f"{self.when_to_use} "

        if include_content and self.content:
            line += self.content.strip()

        if ref_memory_id_key and self.ref_memory_id:
            line += f" {ref_memory_id_key}={self.ref_memory_id}"

        return line.strip()

    @classmethod
    def from_vector_node(cls, node: VectorNode) -> "MemoryNode":
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
                f"Valid types are: {[t.value for t in MemoryType]}",
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
            memory_id=node.vector_id,
            memory_type=memory_type,
            memory_target=metadata.pop("memory_target", ""),
            when_to_use=when_to_use,
            content=content,
            message_time=metadata.pop("message_time", ""),
            ref_memory_id=metadata.pop("ref_memory_id", ""),
            time_created=metadata.pop("time_created", ""),
            time_modified=metadata.pop("time_modified", ""),
            author=metadata.pop("author", ""),
            score=metadata.pop("score", 0),
            vector=node.vector,
            metadata=metadata,
        )
