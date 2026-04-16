"""Base node schema module.

This module defines the BaseNode model, which serves as the foundational
data structure for nodes in the knowledge graph or document processing pipeline.
"""

from uuid import uuid4

from pydantic import BaseModel, Field


class BaseNode(BaseModel):
    """Base node model for graph and document structures.

    This model represents a single node in the knowledge graph or
    a chunk in the document processing pipeline. It contains text content,
    optional embeddings, and associated metadata.

    Attributes:
        id: Unique identifier for the node, auto-generated if not provided.
        text: Text content of the node.
        embedding: Optional vector embedding of the text content.
        metadata: Additional metadata associated with the node.
    """

    id: str = Field(default_factory=lambda: uuid4().hex, description="Unique node identifier")
    text: str = Field(default="", description="Text content of the node")
    embedding: list[float] | None = Field(default=None, description="Vector embedding of text")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
