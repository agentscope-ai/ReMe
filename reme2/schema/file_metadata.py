from pydantic import BaseModel, Field


class FileMetadata(BaseModel):
    file: str = Field(...)
    path: str = Field(...)
    st_mtime: float = Field(...)
    link: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
