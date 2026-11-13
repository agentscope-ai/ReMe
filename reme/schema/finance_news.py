"""One source news item persisted as a UTF-8 JSONL record."""

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class FinanceNewsItem(BaseModel):
    """Stable source ID, publication time, and complete cleaned news text."""

    model_config = ConfigDict(extra="forbid")

    news_id: str = Field(min_length=1)
    event_time: AwareDatetime
    source: Literal["财联社"] = "财联社"
    title: str
    content: str
