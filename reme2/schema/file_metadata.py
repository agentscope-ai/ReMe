from pydantic import BaseModel, Field


class FileMetadata(BaseModel):
    file: str = Field(...)
    path: str = Field(...)
    st_mtime: float = Field(...)
    metadata: dict = Field(default_factory=dict)
