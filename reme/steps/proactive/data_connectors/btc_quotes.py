"""Collect BTC/USDT spot quotes continuously through Binance's public market-data endpoint."""

from datetime import datetime, timezone
import json
from pathlib import Path

import httpx

from ._finance import FinanceCollectorStep
from ....components import R
from ....schema.finance import CryptoQuote

_URL = "https://data-api.binance.vision/api/v3/ticker/24hr"


@R.register("btc_quotes_step")
class BtcQuotesStep(FinanceCollectorStep):
    """Collect once per minute, including weekends, without stock-market calendar filtering."""

    source = "btc_quotes"
    description = (
        "BTC/USDT spot statistics from Binance. Read btc.jsonl; change_24h_percent is the rolling 24-hour change, "
        "not a stock's change from previous close. Prices are in USDT. Use window_end and collected_at for freshness."
    )

    def collect(self, directory: Path) -> list[dict]:
        if self._already_collected():
            return []
        with httpx.Client(timeout=httpx.Timeout(self.request_timeout, connect=5)) as client:
            payload = json.loads(self.request(client, _URL, params={"symbol": "BTCUSDT"}))
        if not isinstance(payload, dict) or payload.get("symbol") != "BTCUSDT":
            raise ValueError("Binance returned missing or invalid BTCUSDT statistics")
        opening = float(payload["openPrice"])
        if opening <= 0:
            raise ValueError("Invalid BTCUSDT 24-hour opening price")
        price = float(payload["lastPrice"])
        quote = CryptoQuote(
            price=price,
            open_24h=opening,
            change_24h_percent=round((price / opening - 1) * 100, 4),
            window_start=datetime.fromtimestamp(int(payload["openTime"]) / 1000, timezone.utc),
            window_end=datetime.fromtimestamp(int(payload["closeTime"]) / 1000, timezone.utc),
            collected_at=datetime.now(self.collected_at.tzinfo),
        )
        return self.write_records(directory, "btc.jsonl", [quote])
