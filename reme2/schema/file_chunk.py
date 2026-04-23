from pydantic import Field

from .base_node import BaseNode


class FileChunk(BaseNode):
    """文件内容分块，包含位置和评分元数据。"""

    path: str = Field(...)
    start_line: int = Field(...)
    end_line: int = Field(...)
    hash: str = Field(...)
    scores: dict[str, float] = Field(default_factory=dict)

    @property
    def score(self) -> float:
        return self.scores.get("score", 0.0)

    @property
    def unique_key(self) -> str:
        return f"{self.path}:{self.start_line}:{self.end_line}"
