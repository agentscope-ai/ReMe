"""Collect BTC/USDT spot quotes continuously through Binance's public market-data endpoint."""

from datetime import datetime, timezone
import json

import httpx

from .base_quotes import QuoteCollectorStep
from .....core.registry import R
from .....schema.finance import CryptoQuote

_URL = "https://data-api.binance.vision/api/v3/ticker/24hr"


@R.register("btc_quotes_step")
class BtcQuotesStep(QuoteCollectorStep):
    """Publish changed spot statistics around the clock, including weekends."""

    source = "btc_quotes"
    filename = "btc.jsonl"
    quote_model = CryptoQuote
    description = (
        "BTC/USDT spot statistics from Binance. Read btc.jsonl; change_24h_percent is the rolling 24-hour change, "
        "and prices are in USDT. Use window_end and collected_at for freshness."
    )

    def fetch_quote(self, client: httpx.Client) -> CryptoQuote:
        payload = json.loads(self.request(client, _URL, params={"symbol": "BTCUSDT"}))
        if not isinstance(payload, dict) or payload.get("symbol") != "BTCUSDT":
            raise ValueError("Binance returned missing or invalid BTCUSDT statistics")
        opening = float(payload["openPrice"])
        if opening <= 0:
            raise ValueError("Invalid BTCUSDT 24-hour opening price")
        price = float(payload["lastPrice"])
        return CryptoQuote(
            price=price,
            open_24h=opening,
            change_24h_percent=round((price / opening - 1) * 100, 4),
            window_start=datetime.fromtimestamp(int(payload["openTime"]) / 1000, timezone.utc),
            window_end=datetime.fromtimestamp(int(payload["closeTime"]) / 1000, timezone.utc),
            collected_at=self.received_at(),
        )

    @staticmethod
    def changed(quote, previous):
        return quote.window_end >= previous.window_end and (quote.price, quote.open_24h) != (
            previous.price,
            previous.open_24h,
        )
