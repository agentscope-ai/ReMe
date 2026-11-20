"""Source records saved by the scheduled finance collectors."""

from datetime import date
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class FinanceNewsItem(BaseModel):
    """One complete CLS news item, identified independently of its collection batch."""

    model_config = ConfigDict(extra="forbid")

    news_id: str = Field(pattern=r"^[1-9]\d*$")
    published_at: AwareDatetime
    source: Literal["财联社"] = "财联社"
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    url: str


class StockQuote(BaseModel):
    """A regular-session quote with explicit market time and observation time."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    symbol: str
    name: str = Field(min_length=1)
    market: Literal["HK", "US"]
    currency: Literal["HKD", "USD"]
    trading_date: date
    price: float = Field(gt=0)
    previous_close: float = Field(gt=0)
    change_percent: float
    quoted_at: AwareDatetime
    collected_at: AwareDatetime
    quote_age_seconds: float = Field(ge=0)
    source: Literal["tencent"] = "tencent"


class CryptoQuote(BaseModel):
    """BTC/USDT spot statistics over an explicit rolling 24-hour window."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    symbol: Literal["BTCUSDT"] = "BTCUSDT"
    market: Literal["CRYPTO"] = "CRYPTO"
    currency: Literal["USDT"] = "USDT"
    price: float = Field(gt=0)
    open_24h: float = Field(gt=0)
    change_24h_percent: float
    window_start: AwareDatetime
    window_end: AwareDatetime
    collected_at: AwareDatetime
    source: Literal["binance"] = "binance"

    @model_validator(mode="after")
    def validate_window(self):
        """Keep the statistics window chronological without inventing a last-trade time."""
        if self.window_start >= self.window_end:
            raise ValueError("window_start must precede window_end")
        return self
