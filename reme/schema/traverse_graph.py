"""Public response schemas for wikilink graph traversal."""

from typing import Literal

from pydantic import BaseModel, Field


class TraverseGraphNode(BaseModel):
    """One indexed file or unresolved wikilink target in a traversal result."""

    id: str = Field(description="Stable node identifier; equal to the workspace-relative path")
    path: str = Field(description="Workspace-relative file path")
    name: str = Field(default="", description="Document name from frontmatter")
    description: str = Field(default="", description="Document description from frontmatter")
    depth: int = Field(ge=0, description="Shortest hop distance from any seed")
    indexed: bool = Field(description="Whether the target has an indexed FileNode")


class TraverseGraphEdge(BaseModel):
    """One directed wikilink, preserving its original source and target."""

    source: str = Field(description="Workspace-relative source file path")
    target: str = Field(description="Workspace-relative target file path")
    target_anchor: str | None = Field(default=None, description="Optional heading, block, or line anchor")
    depth: int = Field(ge=1, description="Traversal depth at which this edge was first reached")


class TraverseGraph(BaseModel):
    """A bounded wikilink graph rooted at one or more workspace paths."""

    version: Literal[1] = 1
    seeds: list[str] = Field(description="Normalized workspace-relative traversal roots")
    depth: int = Field(ge=0, description="Requested hop limit")
    direction: Literal["forward", "backward", "both"]
    nodes: list[TraverseGraphNode]
    edges: list[TraverseGraphEdge]
