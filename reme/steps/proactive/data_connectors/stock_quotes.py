"""Collect changed quotes during the configured HK and US market sessions."""

from datetime import datetime
from functools import lru_cache
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import httpx

from ._finance import FinanceCollectorStep
from ....core.registry import R
from ....schema import ResourceMetadata
from ....schema.finance import StockQuote

_SYMBOLS = {
    "hk01810": ("01810.HK", "HK"),
    "hk09988": ("09988.HK", "HK"),
    "hk00700": ("00700.HK", "HK"),
    "usAAPL": ("AAPL", "US"),
    "usAMZN": ("AMZN", "US"),
    "usMETA": ("META", "US"),
}
_MARKETS = {"HK": ("XHKG", "Asia/Hong_Kong", "HKD"), "US": ("XNYS", "America/New_York", "USD")}
_URL = "https://qt.gtimg.cn/"


@lru_cache(maxsize=8)
def _calendar(name, year):
    try:
        import exchange_calendars
    except ImportError as exc:
        raise RuntimeError("Stock quotes require exchange-calendars; install reme-ai[finance]") from exc
    return exchange_calendars.get_calendar(name, start=f"{year}-01-01", end=f"{year}-12-31")


def _active_session(market, now):
    """Accept only regular trading hours, excluding breaks and the closing boundary."""
    name, timezone, _ = _MARKETS[market]
    day = now.astimezone(ZoneInfo(timezone)).date()
    calendar = _calendar(name, day.year)
    if not calendar.is_session(day.isoformat()):
        return None
    opening = calendar.session_open(day.isoformat()).to_pydatetime()
    closing = calendar.session_close(day.isoformat()).to_pydatetime()
    if not opening <= now < closing:
        return None
    pause = calendar.session_break_start(day.isoformat())
    resume = calendar.session_break_end(day.isoformat())
    if pause <= now < resume:
        return None
    return day


@R.register("stock_quotes_step")
class StockQuotesStep(FinanceCollectorStep):
    """Request active symbols together and publish only changed quotes."""

    source = "stock_quotes"
    description = (
        "Watchlist daily price changes from Tencent. Read quotes.jsonl; change_percent is relative to previous close. "
        "Quotes may be delayed: use quoted_at and quote_age_seconds, not collection time, to judge freshness."
    )

    def delivery_ready(self, path: Path, metadata: ResourceMetadata) -> bool:
        quotes = []
        for item in metadata.files:
            payload = path.parent / item.path
            self.require_path(payload)
            quotes.extend(StockQuote.model_validate(value) for value in self.files.iter_jsonl(payload))
        now = datetime.now(self.collected_at.tzinfo)
        return bool(quotes) and all(_active_session(quote.market, now) is not None for quote in quotes)

    def _quote(self, key, fields, session, received_at):
        symbol, market = _SYMBOLS[key]
        if fields[2].split(".")[0] != key[2:]:
            raise ValueError(f"Tencent returned the wrong symbol for {symbol}")
        _, timezone, currency = _MARKETS[market]
        pattern = "%Y/%m/%d %H:%M:%S" if market == "HK" else "%Y-%m-%d %H:%M:%S"
        quoted = datetime.strptime(fields[30], pattern).replace(tzinfo=ZoneInfo(timezone))
        if quoted.date() != session:
            return None
        price, previous = float(fields[3]), float(fields[4])
        if previous <= 0:
            raise ValueError(f"Invalid previous close for {symbol}")
        return StockQuote(
            symbol=symbol,
            name=fields[1],
            market=market,
            currency=currency,
            trading_date=session,
            price=price,
            previous_close=previous,
            change_percent=round((price / previous - 1) * 100, 4),
            quoted_at=quoted,
            collected_at=received_at,
            quote_age_seconds=(received_at - quoted).total_seconds(),
        )

    def collect(self, directory: Path) -> list[dict]:
        sessions = {market: _active_session(market, self.collected_at) for market in _MARKETS}
        keys = [key for key, (_, market) in _SYMBOLS.items() if sessions[market] is not None]
        if not keys or self._already_collected():
            return []
        with httpx.Client(timeout=httpx.Timeout(self.request_timeout, connect=5)) as client:
            body = self.request(client, _URL, params={"q": ",".join(keys)}).decode("gb18030")
        received_at = datetime.now(self.collected_at.tzinfo)
        values = {}
        for key, value in re.findall(r'v_([A-Za-z0-9]+)="([^"\r\n]*)";', body):
            if key in values:
                raise ValueError(f"Duplicate Tencent quote: {key}")
            values[key] = value.split("~")
        if any(key not in values or len(values[key]) < 33 for key in keys):
            raise ValueError("Tencent returned missing or incomplete quotes")
        quotes = [self._quote(key, values[key], sessions[_SYMBOLS[key][1]], received_at) for key in keys]
        latest = {}
        for quote in self.saved_records("quotes.jsonl", StockQuote):
            previous = latest.get(quote.symbol)
            if previous is None or quote.collected_at > previous.collected_at:
                latest[quote.symbol] = quote
        changed = []
        for quote in quotes:
            if quote is None:
                continue
            previous = latest.get(quote.symbol)
            if previous is not None and (
                quote.quoted_at < previous.quoted_at
                or (quote.trading_date, quote.price, quote.previous_close)
                == (previous.trading_date, previous.price, previous.previous_close)
            ):
                continue
            changed.append(quote)
        return self.write_records(directory, "quotes.jsonl", changed)
