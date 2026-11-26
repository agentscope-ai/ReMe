"""Collect spot gold, WTI crude oil futures, or the Nasdaq Composite from Tencent."""

from datetime import datetime
import re
from zoneinfo import ZoneInfo

import httpx

from ._finance import QuoteCollectorStep
from ....core.registry import R
from ....schema.finance import MarketQuote

_URL = "https://qt.gtimg.cn/"
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_NEW_YORK = ZoneInfo("America/New_York")
_INSTRUMENTS = {
    "gold": ("hf_XAU", "XAUUSD", "XAU/USD spot gold in USD per troy ounce"),
    "oil": ("hf_CL", "CL", "WTI crude oil futures in USD per barrel"),
    "nasdaq": ("us.IXIC", "IXIC", "Nasdaq Composite in index points"),
}


@R.register("tencent_quotes_step")
class TencentQuotesStep(QuoteCollectorStep):
    """Collect one independently scheduled instrument from Tencent."""

    quote_model = MarketQuote

    def __init__(self, instrument: str, **kwargs):
        super().__init__(**kwargs)
        self.ticker, self.symbol, label = _INSTRUMENTS[instrument]
        self.source = f"{instrument}_quotes"
        self.filename = f"{instrument}.jsonl"
        self.description = (
            f"{label} from Tencent. Read {self.filename}; change_percent is relative to previous close. "
            "Use quoted_at and collected_at for freshness."
        )

    def fetch_quote(self, client: httpx.Client) -> MarketQuote:
        data = self.request(client, _URL, params={"q": self.ticker})
        received_at = self.received_at()
        match = re.fullmatch(rf'v_{re.escape(self.ticker)}="([^"\r\n]*)";', data.decode("gb18030").strip())
        if not match:
            raise ValueError(f"Tencent returned no quote for {self.ticker}")

        if self.symbol == "IXIC":
            fields = match[1].split("~")
            if len(fields) < 31 or fields[2] != ".IXIC":
                raise ValueError("Tencent returned an incomplete Nasdaq Composite quote")
            price, previous_close = float(fields[3]), float(fields[4])
            quoted_at = datetime.fromisoformat(fields[30]).replace(tzinfo=_NEW_YORK)
        else:
            fields = match[1].split(",")
            if len(fields) < 13:
                raise ValueError(f"Tencent returned an incomplete quote for {self.ticker}")
            price, previous_close = float(fields[0]), float(fields[7])
            quoted_at = datetime.fromisoformat(f"{fields[12]}T{fields[6]}").replace(tzinfo=_SHANGHAI)

        if previous_close == 0:
            raise ValueError(f"Invalid previous close for {self.ticker}")
        return MarketQuote(
            symbol=self.symbol,
            price=price,
            previous_close=previous_close,
            change_percent=round((price - previous_close) / abs(previous_close) * 100, 4),
            quoted_at=quoted_at,
            collected_at=received_at,
        )

    @staticmethod
    def changed(quote, previous):
        return quote.quoted_at >= previous.quoted_at and (quote.price, quote.previous_close) != (
            previous.price,
            previous.previous_close,
        )
