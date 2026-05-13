from pydantic import BaseModel, ConfigDict, Field

from .file_edge import FileEdge


class FileFrontMatter(BaseModel):
    model_config = ConfigDict(extra="allow")

    title: str = Field(default="")
    description: str = Field(default="")
    tags: list[str] | None = Field(default=None)

    @property
    def metadata(self) -> dict:
        return dict(self.__pydantic_extra__ or {})


class FileNode(BaseModel):
    path: str = Field(default=...)
    st_mtime: float = Field(default=...)
    edges: list[FileEdge] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    front_matter: FileFrontMatter = Field(default_factory=FileFrontMatter)
