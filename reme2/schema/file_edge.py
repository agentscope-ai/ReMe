"""FileEdge — typed wikilink edge between vault files.

Replaces the prior `link: list[str]` model on FileMetadata. An edge
captures both the bare wikilink shape (`target` + optional anchor /
alias / embed prefix) AND the typed-edge predicate that turns a plain
`[[X]]` reference into a graph relation.

Sources:
    "regex"       — extracted from inline syntax (`[[X]]`, `[pred:: [[X]]]`)
    "frontmatter" — inferred from a frontmatter key acting as predicate
                    (e.g. `author: "[[John]]"` → predicate="author")
    "llm"         — produced by an IE pipeline; should set `confidence`

The `target` field stays raw (e.g. `"X"` or `"topics/X"`) — resolution
to an absolute path is done by the file_store via `resolve_wikilink`.
"""

from typing import Literal

from pydantic import BaseModel, Field


class FileEdge(BaseModel):
    target: str = Field(..., description="Raw wikilink target as written in source.")
    predicate: str | None = Field(
        default=None,
        description="Typed-edge predicate (Dataview-style). None for plain wikilinks.",
    )
    anchor: str | None = Field(default=None, description="Heading or block anchor (after #).")
    alias: str | None = Field(default=None, description="Display alias (after |).")
    embed: bool = Field(default=False, description="True for `![[X]]` embed prefix.")
    source: Literal["regex", "frontmatter", "llm"] = Field(
        default="regex",
        description="Provenance of this edge.",
    )
    confidence: float | None = Field(
        default=None,
        description="LLM confidence (0..1). None for regex/frontmatter.",
    )

    @property
    def is_typed(self) -> bool:
        return self.predicate is not None
