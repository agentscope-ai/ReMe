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

    @property
    def hash(self) -> str:
        return "_".join([self.id, self.path, str(self.start_line), str(self.end_line)])
