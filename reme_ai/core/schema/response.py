from pydantic import Field, BaseModel


class Response(BaseModel):
    answer: str | dict | list = Field(default="")
    success: bool = Field(default=True)
    metadata: dict = Field(default_factory=dict)
