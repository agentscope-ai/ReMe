from pydantic import BaseModel, Field, ConfigDict


class FileEdge(BaseModel):
    link: str = Field(default=...)
    predicate: str | None = Field(default=None)

    @property
    def link_path(self) -> str:
        return self.link.split("#", 1)[0]

    @property
    def link_anchor(self) -> str:
        link_split = self.link.split("#", 1)
        return link_split[1] if len(link_split) > 1 else ""


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
    front_matter: FileFrontMatter = Field(default_factory=FileFrontMatter)
