"""File chunk — an embedding node tied to a line range in a file."""

from pydantic import Field

from .emb_node import EmbNode


class FileChunk(EmbNode):
    """A chunk of a file with positional info and per-stage retrieval scores."""

    path: str = Field(default="", description="Path relative to the vault")
    start_line: int = Field(default=0, description="Inclusive start line (0-based)")
    end_line: int = Field(default=0, description="Exclusive end line")
    scores: dict[str, float] = Field(default_factory=dict, description="Retrieval scores keyed by stage")

    @property
    def score(self) -> float:
        """Final aggregated score; 0.0 if not yet computed."""
        return self.scores.get("score", 0.0)

    def set_hash_id(self):
        """Replace ``id`` with a deterministic hash of (path, range, text)."""
        from ..utils import hash_text

        self.id = hash_text(" ".join([self.path, str(self.start_line), str(self.end_line), self.text]))
        return self
