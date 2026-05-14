from pydantic import Field

from .emb_node import EmbNode


class FileChunk(EmbNode):
    path: str = Field(default="")
    start_line: int = Field(default=0)
    end_line: int = Field(default=0)
    scores: dict[str, float] = Field(default_factory=dict)

    @property
    def score(self) -> float:
        return self.scores.get("score", 0.0)

    def set_hash_id(self):
        from ..utils import hash_text
        self.id = hash_text(" ".join([self.path, str(self.start_line), str(self.end_line), self.text]))
        return self