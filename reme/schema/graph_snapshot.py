"""Public response schemas for a complete indexed wikilink graph snapshot."""

from typing import Literal

from pydantic import BaseModel, Field


class GraphSnapshotNode(BaseModel):
    """One virtual category or indexed Markdown file in a graph snapshot."""

    id: str = Field(description="Stable node identifier; documents use their workspace-relative path")
    path: str = Field(description="Workspace-relative file or virtual category path")
    name: str = Field(default="", description="Document name from frontmatter")
    description: str = Field(default="", description="Document description from frontmatter")
    indexed: bool = Field(description="Whether the target has an indexed FileNode")
    virtual: bool = Field(default=False, description="Whether this is a generated category node")


class GraphSnapshotEdge(BaseModel):
    """One directed category or wikilink edge in a graph snapshot."""

    source: str = Field(description="Source node identifier")
    target: str = Field(description="Target node identifier")
    target_anchor: str | None = Field(default=None, description="Optional heading, block, or line anchor")


class GraphSnapshot(BaseModel):
    """A category-rooted snapshot of digest wikilinks and their daily-note leaves."""

    version: Literal[1] = 1
    nodes: list[GraphSnapshotNode]
    edges: list[GraphSnapshotEdge]
