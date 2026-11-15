"""Finance schedules collect into isolated files and use the real append_event Job."""

# pylint: disable=protected-access,missing-function-docstring,redefined-outer-name

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from threading import Event as ThreadEvent
from unittest.mock import AsyncMock, Mock
from zoneinfo import ZoneInfo

import httpx
import pytest

from reme.components import ApplicationContext, LocalAgentRuntime, R, RuntimeContext
from reme.components.agent_runtime._internal.files import AgentFiles
from reme.components.agent_wrapper import BaseAgentWrapper
from reme.components.agent_inbox import AgentEventInbox
from reme.components.job import BaseJob
from reme.config import resolve_app_config
from reme.enumeration import ComponentEnum
from reme.schema import CryptoQuote, FinanceNewsItem, Response, StockQuote
from reme.steps.proactive.data_connectors import BtcQuotesStep, FinanceNewsStep, StockQuotesStep
from reme.steps.proactive.data_connectors import base_connector, btc_quotes, stock_quotes
from reme.steps.proactive.data_connectors.stock_quotes import _active_session

_SHANGHAI = ZoneInfo("Asia/Shanghai")


@pytest.fixture
def clock(monkeypatch):
    value = Mock(wraps=datetime)
    value.now.return_value = datetime(2026, 11, 16, 10, 0, tzinfo=_SHANGHAI)
    monkeypatch.setattr(base_connector, "datetime", value)
    monkeypatch.setattr(btc_quotes, "datetime", value)
    monkeypatch.setattr(stock_quotes, "datetime", value)
    return value


@pytest.fixture
def mock_http(monkeypatch):
    client = httpx.Client
    requests = []

    def install(handler):
        def respond(request):
            requests.append(request)
            return handler(request)

        monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs))
        return requests

    return install


@asynccontextmanager
async def collector_runtime(tmp_path):
    context = ApplicationContext(workspace_dir=str(tmp_path), reme_agent_dir="reme_agents", language="zh")
    inbox = AgentEventInbox(app_context=context)
    runtime = LocalAgentRuntime(
        app_context=context,
        agents=[{"name": "finance", "trigger_type": "event", "description": "偏好 AI 相关的财经新闻"}],
    )
    context.components = {
        "agent_inbox": {"default": inbox},
        "agent_wrapper": {"default": Mock(spec=BaseAgentWrapper)},
        "agent_runtime": {"default": runtime},
    }
    job = BaseJob(app_context=context, steps=[{"backend": "append_event_step"}])
    append = AsyncMock(side_effect=job.__call__)
    context.jobs["append_event"] = append
    async with inbox, runtime, job:
        runtime.watcher.cancel()
        await asyncio.gather(runtime.watcher, return_exceptions=True)
        runtime.watcher = None
        yield context, inbox, append


def news_row(identifier, when, content="<b>AI news</b><script>untrusted()</script> full text"):
    return {"id": identifier, "ctime": int(when.timestamp()), "title": "AI 新闻", "content": content}


def news_response(rows):
    return httpx.Response(200, json={"errno": 0, "data": {"roll_data": rows}})


def news_batch(clock):
    now = clock.now.return_value
    return [news_row(2, now - timedelta(minutes=1)), news_row(1, now - timedelta(hours=25))]


def tencent_response(request, quoted="2026/11/16 09:45:00", price="90.55", previous="93.8"):
    rows = []
    for key in request.url.params["q"].split(","):
        fields = [""] * 33
        fields[1], fields[2] = "股票名称", key[2:]
        fields[3], fields[4], fields[30], fields[32] = price, previous, quoted, "-3.46"
        rows.append(f'v_{key}="{"~".join(fields)}";')
    return httpx.Response(200, content="\n".join(rows).encode("gb18030"))


def btc_statistics(clock):
    end = clock.now.return_value - timedelta(seconds=1)
    return {
        "symbol": "BTCUSDT",
        "lastPrice": "66000.00",
        "openPrice": "60000.00",
        "openTime": int((end - timedelta(hours=24)).timestamp() * 1000),
        "closeTime": int(end.timestamp() * 1000),
    }


@pytest.mark.asyncio
async def test_news_uses_existing_job_and_source_files_deduplicate_after_restart(tmp_path, clock, mock_http):
    requests = mock_http(lambda _: news_response(news_batch(clock)))
    async with collector_runtime(tmp_path) as (context, inbox, append):
        response = await FinanceNewsStep(app_context=context, request_interval=0)()
        assert response.success
        event = (await inbox.claim("finance"))["items"][0]
        assert event["source"] == "finance_news"
        assert append.await_args.kwargs["agent_id"] == "finance"
        assert append.await_args.kwargs["events"][0]["id"] == event["id"]
        manifest, payload = event["paths"]
        item = FinanceNewsItem.model_validate_json((tmp_path / payload).read_text())
        assert item.news_id == "2" and item.content == "AI news full text"
        assert item.url == "https://www.cls.cn/detail/2"
        assert (tmp_path / manifest).parent.joinpath("delivery-finance.json").is_file()
    async with collector_runtime(tmp_path) as (context, inbox, append):
        response = await FinanceNewsStep(app_context=context, request_interval=0)()
        assert response.answer == {"skipped": "no_new_data"}
        assert (await inbox.snapshot("finance"))["pending_count"] == 1
        append.assert_not_awaited()
    assert len(requests) == 2
    assert len(list(tmp_path.glob("resource/finance_news/*/*/news.jsonl"))) == 1


@pytest.mark.asyncio
async def test_news_paginates_until_window_boundary_and_deduplicates_overlapping_pages(tmp_path, clock, mock_http):
    now = clock.now.return_value
    pages = [
        [news_row(3, now - timedelta(minutes=1)), news_row(2, now - timedelta(hours=2))],
        [news_row(2, now - timedelta(hours=2)), news_row(1, now - timedelta(hours=25))],
    ]
    requests = mock_http(lambda _: news_response(pages.pop(0)))
    async with collector_runtime(tmp_path) as (context, inbox, _):
        await FinanceNewsStep(app_context=context, request_interval=0)()
        event = (await inbox.claim("finance"))["items"][0]
        items = [
            FinanceNewsItem.model_validate_json(line)
            for line in (tmp_path / event["paths"][1]).read_text().splitlines()
        ]
        assert [item.news_id for item in items] == ["2", "3"]
    assert int(requests[1].url.params["last_time"]) == int((now - timedelta(hours=2)).timestamp())
    assert requests[0].headers["referer"] == "https://www.cls.cn/telegraph"
    assert "sign" in requests[0].url.params


@pytest.mark.asyncio
async def test_no_new_news_creates_no_package_or_event(tmp_path, clock, mock_http):
    mock_http(lambda _: news_response(news_batch(clock)[1:]))
    async with collector_runtime(tmp_path) as (context, inbox, append):
        assert (await FinanceNewsStep(app_context=context)()).answer == {"skipped": "no_new_data"}
        append.assert_not_awaited()
        assert (await inbox.snapshot("finance"))["pending_count"] == 0
    assert not (tmp_path / "resource").exists()
    assert not list(tmp_path.glob(".reme-resource-*"))


@pytest.mark.asyncio
async def test_failed_delivery_retries_original_package_before_collecting_again(tmp_path, clock, mock_http):
    mock_http(lambda _: news_response(news_batch(clock)))
    async with collector_runtime(tmp_path) as (context, inbox, append):
        append.return_value = Response(success=False, answer="queue full")
        append.side_effect = lambda **_: append.return_value
        with pytest.raises(RuntimeError, match="queue full"):
            await FinanceNewsStep(app_context=context)()
        original = append.await_args.kwargs["events"][0]
        assert (await inbox.snapshot("finance"))["pending_count"] == 0
        assert not list(tmp_path.glob("resource/**/delivery-finance.json"))
    async with collector_runtime(tmp_path) as (context, inbox, append):
        await FinanceNewsStep(app_context=context)()
        assert append.await_args.kwargs["events"] == [original]
        assert (await inbox.snapshot("finance"))["pending_count"] == 1
    assert len(list(tmp_path.glob("resource/**/metadata.json"))) == 1


@pytest.mark.asyncio
async def test_successful_append_without_delivery_receipt_is_idempotent_after_restart(
    tmp_path,
    clock,
    mock_http,
    monkeypatch,
):
    mock_http(lambda _: news_response(news_batch(clock)))
    write = AgentFiles.write_json

    def fail_receipt(files, path, value):
        if path.name == "delivery-finance.json":
            raise OSError("receipt unavailable")
        return write(files, path, value)

    async with collector_runtime(tmp_path) as (context, inbox, _):
        with monkeypatch.context() as patch:
            patch.setattr(AgentFiles, "write_json", fail_receipt)
            with pytest.raises(OSError, match="receipt unavailable"):
                await FinanceNewsStep(app_context=context)()
        assert (await inbox.snapshot("finance"))["pending_count"] == 1
    async with collector_runtime(tmp_path) as (context, inbox, append):
        await FinanceNewsStep(app_context=context)()
        append.assert_awaited_once()
        assert (await inbox.snapshot("finance"))["pending_count"] == 1
    assert len(list(tmp_path.glob("resource/**/metadata.json"))) == 1


@pytest.mark.asyncio
async def test_concurrent_collectors_do_not_duplicate_news(tmp_path, clock, mock_http):
    mock_http(lambda _: news_response(news_batch(clock)))
    async with collector_runtime(tmp_path) as (context, inbox, append):
        await asyncio.gather(*(FinanceNewsStep(app_context=context)() for _ in range(2)))
        append.assert_awaited_once()
        assert (await inbox.snapshot("finance"))["pending_count"] == 1


@pytest.mark.parametrize("kind", ["api_error", "missing_rows", "invalid_record", "stalled", "oversized"])
@pytest.mark.asyncio
async def test_bad_news_response_never_publishes_partial_data(tmp_path, clock, mock_http, kind):
    data = {"errno": 1} if kind == "api_error" else {"errno": 0, "data": {}}
    if kind == "invalid_record":
        data["data"]["roll_data"] = [news_batch(clock)[0], {"ctime": "invalid"}]
    if kind == "stalled":
        data["data"]["roll_data"] = [news_row(2, clock.now.return_value)]
    mock_http(lambda _: httpx.Response(200, json=data))
    async with collector_runtime(tmp_path) as (context, inbox, append):
        options = {"max_file_bytes": 1} if kind == "oversized" else {}
        with pytest.raises((ValueError, KeyError)):
            await FinanceNewsStep(app_context=context, **options)()
        append.assert_not_awaited()
        assert (await inbox.snapshot("finance"))["pending_count"] == 0
    assert not list(tmp_path.glob("resource/**/metadata.json"))


@pytest.mark.asyncio
async def test_http_errors_retry_before_publishing(tmp_path, clock, mock_http, monkeypatch):
    responses = [httpx.Response(503), news_response(news_batch(clock))]
    requests = mock_http(lambda _: responses.pop(0))
    monkeypatch.setattr(FinanceNewsStep, "wait", lambda *_: None)
    async with collector_runtime(tmp_path) as (context, inbox, _):
        await FinanceNewsStep(app_context=context)()
        assert (await inbox.snapshot("finance"))["pending_count"] == 1
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_scope_denial_precedes_history_reads_and_network(tmp_path, clock, mock_http):
    requests = mock_http(lambda _: news_response(news_batch(clock)))
    async with collector_runtime(tmp_path) as (context, _, append):
        with pytest.raises(PermissionError):
            await FinanceNewsStep(app_context=context)(RuntimeContext(_allowed_paths=["resource"]))
        append.assert_not_awaited()
    assert not requests and not (tmp_path / "resource").exists()


@pytest.mark.asyncio
async def test_cancellation_interrupts_pagination_without_publishing(tmp_path, clock, mock_http):
    requested = ThreadEvent()

    def page(_):
        requested.set()
        return news_response(news_batch(clock)[:1])

    mock_http(page)
    async with collector_runtime(tmp_path) as (context, inbox, append):
        task = asyncio.create_task(FinanceNewsStep(app_context=context, request_interval=30)())
        assert await asyncio.to_thread(requested.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        append.assert_not_awaited()
        assert (await inbox.snapshot("finance"))["pending_count"] == 0
    assert not list(tmp_path.glob("resource/**/metadata.json"))
    assert not list(tmp_path.glob(".reme-resource-*"))


@pytest.mark.parametrize(
    ("market", "time", "expected"),
    [
        ("HK", "2026-11-16T10:00", "2026-11-16"),
        ("HK", "2026-11-16T12:30", None),
        ("HK", "2026-11-15T10:00", None),
        ("HK", "2026-10-01T10:00", None),
        ("HK", "2026-12-24T13:00", None),
        ("HK", "2026-11-16T16:20", "2026-11-16"),
        ("HK", "2026-11-16T16:31", None),
        ("US", "2026-11-14T02:00", "2026-11-13"),
        ("US", "2026-11-15T02:00", None),
        ("US", "2026-11-26T23:00", None),
        ("US", "2026-11-28T02:31", None),
        ("US", "2026-07-10T21:30", "2026-07-10"),
        ("US", "2026-11-13T21:30", None),
    ],
)
def test_market_calendars_cover_holidays_breaks_dst_and_closing_catchup(market, time, expected):
    pytest.importorskip("exchange_calendars")
    session = _active_session(market, datetime.fromisoformat(time).replace(tzinfo=_SHANGHAI))
    assert (session.isoformat() if session else None) == expected


@pytest.mark.asyncio
async def test_quote_snapshots_batch_active_market_and_deduplicate_the_same_minute(tmp_path, clock, mock_http):
    pytest.importorskip("exchange_calendars")
    requests = mock_http(tencent_response)
    async with collector_runtime(tmp_path) as (context, inbox, append):
        await StockQuotesStep(app_context=context)()
        batch = await inbox.claim("finance")
        event = batch["items"][0]
        quotes = [
            StockQuote.model_validate_json(line) for line in (tmp_path / event["paths"][1]).read_text().splitlines()
        ]
        assert {quote.symbol for quote in quotes} == {"01810.HK", "09988.HK", "00700.HK"}
        assert all(quote.quote_age_seconds == 900 for quote in quotes)
        assert all(quote.change_percent == pytest.approx(-3.4648) for quote in quotes)
        assert (await StockQuotesStep(app_context=context)()).answer == {"skipped": "no_new_data"}
        clock.now.return_value += timedelta(minutes=1)
        await StockQuotesStep(app_context=context)()
        assert (await inbox.snapshot("finance"))["pending_count"] == 2
        assert append.await_count == 2
    assert len(requests) == 2
    assert requests[0].url.params["q"] == "hk01810,hk09988,hk00700"


@pytest.mark.asyncio
async def test_quote_freshness_uses_receipt_time_when_prices_update_during_request(tmp_path, clock, mock_http):
    pytest.importorskip("exchange_calendars")
    received_at = clock.now.return_value + timedelta(seconds=2)

    def respond(request):
        clock.now.return_value = received_at
        return tencent_response(request, quoted="2026/11/16 10:00:01")

    mock_http(respond)
    async with collector_runtime(tmp_path) as (context, inbox, _):
        await StockQuotesStep(app_context=context)()
        event = (await inbox.claim("finance"))["items"][0]
        quotes = [
            StockQuote.model_validate_json(line) for line in (tmp_path / event["paths"][1]).read_text().splitlines()
        ]
        assert len(quotes) == 3
        assert all(quote.collected_at == received_at and quote.quote_age_seconds == 1 for quote in quotes)


@pytest.mark.asyncio
async def test_us_quotes_use_new_york_date_even_on_beijing_saturday(tmp_path, clock, mock_http):
    pytest.importorskip("exchange_calendars")
    clock.now.return_value = datetime(2026, 11, 14, 2, 0, tzinfo=_SHANGHAI)
    requests = mock_http(lambda request: tencent_response(request, quoted="2026-11-13 12:45:00"))
    async with collector_runtime(tmp_path) as (context, inbox, _):
        await StockQuotesStep(app_context=context)()
        event = (await inbox.claim("finance"))["items"][0]
        quotes = [
            StockQuote.model_validate_json(line) for line in (tmp_path / event["paths"][1]).read_text().splitlines()
        ]
        assert {quote.symbol for quote in quotes} == {"AAPL", "AMZN", "META"}
        assert {quote.trading_date.isoformat() for quote in quotes} == {"2026-11-13"}
    assert requests[0].url.params["q"] == "usAAPL,usAMZN,usMETA"


@pytest.mark.asyncio
async def test_closed_markets_do_not_request_quotes(tmp_path, clock, mock_http):
    pytest.importorskip("exchange_calendars")
    clock.now.return_value = datetime(2026, 11, 15, 10, 0, tzinfo=_SHANGHAI)
    requests = mock_http(tencent_response)
    async with collector_runtime(tmp_path) as (context, _, append):
        assert (await StockQuotesStep(app_context=context)()).answer == {"skipped": "no_new_data"}
        append.assert_not_awaited()
    assert not requests and not (tmp_path / "resource").exists()


@pytest.mark.parametrize("kind", ["missing", "wrong_symbol", "zero_close", "nan_price", "future"])
@pytest.mark.asyncio
@pytest.mark.usefixtures("clock")
async def test_bad_quotes_fail_without_publishing_a_partial_snapshot(tmp_path, mock_http, kind):
    pytest.importorskip("exchange_calendars")

    def respond(request):
        if kind == "missing":
            return httpx.Response(200, content=b'v_hk01810="";')
        if kind == "future":
            return tencent_response(request, quoted="2026/11/16 11:00:00")
        response = tencent_response(
            request,
            price="nan" if kind == "nan_price" else "90.55",
            previous="0" if kind == "zero_close" else "93.8",
        )
        if kind == "wrong_symbol":
            return httpx.Response(200, content=response.content.replace(b"~01810~", b"~00700~"))
        return response

    mock_http(respond)
    async with collector_runtime(tmp_path) as (context, inbox, append):
        with pytest.raises(ValueError):
            await StockQuotesStep(app_context=context)()
        append.assert_not_awaited()
        assert (await inbox.snapshot("finance"))["pending_count"] == 0
    assert not list(tmp_path.glob("resource/**/metadata.json"))


@pytest.mark.asyncio
@pytest.mark.usefixtures("clock")
async def test_previous_trading_day_quotes_are_not_published_as_today(tmp_path, mock_http):
    pytest.importorskip("exchange_calendars")
    mock_http(lambda request: tencent_response(request, quoted="2026/11/13 16:08:00"))
    async with collector_runtime(tmp_path) as (context, _, append):
        assert (await StockQuotesStep(app_context=context)()).answer == {"skipped": "no_new_data"}
        append.assert_not_awaited()
    assert not list(tmp_path.glob("resource/**/metadata.json"))


@pytest.mark.asyncio
async def test_btc_collects_on_weekends_and_deduplicates_the_observed_minute(tmp_path, clock, mock_http):
    clock.now.return_value = datetime(2026, 11, 15, 10, 0, tzinfo=_SHANGHAI)
    requests = mock_http(lambda _: httpx.Response(200, json=btc_statistics(clock)))
    async with collector_runtime(tmp_path) as (context, inbox, append):
        await BtcQuotesStep(app_context=context)()
        event = (await inbox.claim("finance"))["items"][0]
        assert event["source"] == "btc_quotes"
        quote = CryptoQuote.model_validate_json((tmp_path / event["paths"][1]).read_text())
        assert quote.symbol == "BTCUSDT" and quote.currency == "USDT"
        assert quote.price == 66000 and quote.change_24h_percent == 10
        assert quote.window_end - quote.window_start == timedelta(hours=24)
        assert quote.collected_at == clock.now.return_value
        assert "previous_close" not in quote.model_dump()
        assert (await BtcQuotesStep(app_context=context)()).answer == {"skipped": "no_new_data"}
        clock.now.return_value += timedelta(minutes=1)
        await BtcQuotesStep(app_context=context)()
        assert (await inbox.snapshot("finance"))["pending_count"] == 2
        assert append.await_count == 2
    assert len(requests) == 2
    assert requests[0].url.host == "data-api.binance.vision"
    assert dict(requests[0].url.params) == {"symbol": "BTCUSDT"}
    assert "x-mbx-apikey" not in requests[0].headers


@pytest.mark.asyncio
async def test_btc_same_minute_deduplication_survives_restart(tmp_path, clock, mock_http):
    requests = mock_http(lambda _: httpx.Response(200, json=btc_statistics(clock)))
    async with collector_runtime(tmp_path) as (context, _, _):
        await BtcQuotesStep(app_context=context)()
    async with collector_runtime(tmp_path) as (context, inbox, append):
        assert (await BtcQuotesStep(app_context=context)()).answer == {"skipped": "no_new_data"}
        append.assert_not_awaited()
        assert (await inbox.snapshot("finance"))["pending_count"] == 1
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_btc_failed_delivery_reuses_the_saved_batch_after_restart(tmp_path, clock, mock_http):
    requests = mock_http(lambda _: httpx.Response(200, json=btc_statistics(clock)))
    async with collector_runtime(tmp_path) as (context, _, append):
        append.side_effect = None
        append.return_value = Response(success=False, answer="queue full")
        with pytest.raises(RuntimeError, match="queue full"):
            await BtcQuotesStep(app_context=context)()
        original = append.await_args.kwargs["events"]
    async with collector_runtime(tmp_path) as (context, inbox, append):
        await BtcQuotesStep(app_context=context)()
        assert append.await_args.kwargs["events"] == original
        assert (await inbox.snapshot("finance"))["pending_count"] == 1
    assert len(requests) == 1
    assert len(list(tmp_path.glob("resource/btc_quotes/*/*/btc.jsonl"))) == 1


@pytest.mark.parametrize("kind", ["wrong_symbol", "zero_open", "nan_price", "reversed_window", "api_error"])
@pytest.mark.asyncio
async def test_invalid_btc_statistics_produce_no_files_or_events(tmp_path, clock, mock_http, kind):
    payload = btc_statistics(clock)
    if kind == "wrong_symbol":
        payload["symbol"] = "ETHUSDT"
    elif kind == "zero_open":
        payload["openPrice"] = "0"
    elif kind == "nan_price":
        payload["lastPrice"] = "nan"
    elif kind == "reversed_window":
        payload["openTime"] = payload["closeTime"] + 1
    else:
        payload = {"code": -1121, "msg": "Invalid symbol"}
    mock_http(lambda _: httpx.Response(200, json=payload))
    async with collector_runtime(tmp_path) as (context, inbox, append):
        with pytest.raises(ValueError):
            await BtcQuotesStep(app_context=context)()
        append.assert_not_awaited()
        assert (await inbox.snapshot("finance"))["pending_count"] == 0
    assert not list(tmp_path.glob("resource/**/metadata.json"))


def test_proactive_configuration_connects_both_crons_to_existing_finance_agent():
    config = resolve_app_config(config="proactive", log_config=False)
    jobs = config["jobs"]
    for name, interval, step in (
        ("finance_news_cron", "*/5 * * * *", FinanceNewsStep),
        ("stock_quotes_cron", "* * * * *", StockQuotesStep),
        ("btc_quotes_cron", "* * * * *", BtcQuotesStep),
    ):
        assert jobs[name]["backend"] == "cron" and jobs[name]["cron"] == interval
        assert jobs[name].get("enabled", True)
        spec = jobs[name]["steps"][0]
        assert spec["agent_id"] == "finance"
        assert R.get(ComponentEnum.STEP, spec["backend"]) is step
    agents = {value["name"]: value for value in config["components"]["agent_runtime"]["default"]["agents"]}
    assert agents["finance"]["trigger_type"] == "event" and "cron" not in agents["finance"]
    assert agents["finance"]["enabled"] and agents["work"]["enabled"]
