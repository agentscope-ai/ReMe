"""Finance schedules collect into isolated files and deliver through the Proactive Manager."""

# pylint: disable=protected-access,missing-function-docstring,redefined-outer-name

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from threading import Event as ThreadEvent
from unittest.mock import AsyncMock, Mock
from zoneinfo import ZoneInfo

import httpx
import pytest

from reme.core.context import ApplicationContext, RuntimeContext
from reme.components import LocalProactiveManager
from reme.utils.files.workspace_files import WorkspaceFiles
from reme.components.agent_wrapper import BaseAgentWrapper
from reme.components.agent_inbox import AgentEventInbox
from reme.schema import CryptoQuote, FinanceNewsItem, MarketQuote
from reme.steps.proactive.data_connectors import BtcQuotesStep, FinanceNewsStep, TencentQuotesStep
from reme.steps.proactive.data_connectors import _finance, base_connector

_SHANGHAI = ZoneInfo("Asia/Shanghai")


@pytest.fixture
def clock(monkeypatch):
    value = Mock(wraps=datetime)
    value.now.return_value = datetime(2026, 11, 16, 10, 0, tzinfo=_SHANGHAI)
    monkeypatch.setattr(base_connector, "datetime", value)
    monkeypatch.setattr(_finance, "datetime", value)
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
    runtime = LocalProactiveManager(
        app_context=context,
        runtimes=[{"agent_name": "finance", "task_description": "关注主要行情和财联社电报"}],
    )
    context.components = {
        "agent_inbox": {"default": inbox},
        "agent_wrapper": {"default": Mock(spec=BaseAgentWrapper)},
        "proactive_manager": {"default": runtime},
    }
    append = AsyncMock(side_effect=runtime.deliver_events)
    runtime.deliver_events = append
    async with inbox, runtime:
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


def gold_response(clock, *, price="2420", previous_close="2200", quoted_at=None):
    local = quoted_at or clock.now.return_value - timedelta(seconds=1)
    fields = [price, "10", price, price, price, "2200", local.strftime("%H:%M:%S"), previous_close]
    fields += ["2270", "0", "0", "0", local.date().isoformat(), "伦敦金（现货黄金）"]
    payload = f'v_hf_XAU="{",".join(fields)}";\n'
    return httpx.Response(200, content=payload.encode("gb18030"))


@pytest.mark.parametrize("instrument,ticker,symbol", [("oil", "hf_CL", "CL"), ("nasdaq", "us.IXIC", "IXIC")])
@pytest.mark.asyncio
async def test_new_instruments_publish_the_right_quote_and_timestamp(
    tmp_path,
    clock,
    mock_http,
    instrument,
    ticker,
    symbol,
):
    quoted_at = clock.now.return_value - timedelta(seconds=30)
    if instrument == "oil":
        fields = ["97", "-3", "97", "97", "100", "96", quoted_at.strftime("%H:%M:%S"), "100"]
        fields += ["100", "0", "0", "0", quoted_at.date().isoformat(), "纽约原油"]
        payload = ",".join(fields)
    else:
        new_york = quoted_at.astimezone(ZoneInfo("America/New_York"))
        fields = ["200", "纳斯达克", ".IXIC", "97", "100"] + ["0"] * 25 + [new_york.strftime("%Y-%m-%d %H:%M:%S")]
        payload = "~".join(fields)
    requests = mock_http(lambda _: httpx.Response(200, content=f'v_{ticker}="{payload}";'.encode("gb18030")))

    async with collector_runtime(tmp_path) as (context, inbox, _):
        await TencentQuotesStep(instrument=instrument, app_context=context)()
        (event,) = (await inbox.claim("finance"))["items"]
        assert event["source"] == f"{instrument}_quotes"
        quote = MarketQuote.model_validate_json((tmp_path / event["paths"][0]).read_text())
        assert quote.symbol == symbol
        assert (quote.price, quote.previous_close, quote.change_percent) == (97, 100, -3)
        assert quote.quoted_at == quoted_at
        assert quote.collected_at == clock.now.return_value

    assert dict(requests[0].url.params) == {"q": ticker}


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
async def test_failed_delivery_retries_original_package_before_collecting_again(tmp_path, clock, mock_http):
    mock_http(lambda _: gold_response(clock))
    async with collector_runtime(tmp_path) as (context, inbox, append):
        append.return_value = {"deliveries": {}, "rejected": {}, "errors": {"finance": "queue full"}}
        append.side_effect = lambda **_: append.return_value
        with pytest.raises(RuntimeError, match="queue full"):
            await TencentQuotesStep(instrument="gold", app_context=context)()
        original = append.await_args.kwargs["events"][0]
        assert (await inbox.snapshot("finance"))["pending_count"] == 0
        (receipt,) = tmp_path.glob("resource/**/.delivery.json")
        assert WorkspaceFiles(tmp_path).read_json(receipt, {})["pending"] == ["finance"]
    async with collector_runtime(tmp_path) as (context, inbox, append):
        await TencentQuotesStep(instrument="gold", app_context=context, task_names=["another_task"])()
        assert append.await_args.kwargs["events"] == [original]
        assert append.await_args.kwargs["task_names"] == ["finance"]
        assert (await inbox.snapshot("finance"))["pending_count"] == 1
    assert len(list(tmp_path.glob("resource/**/metadata.json"))) == 1


@pytest.mark.asyncio
async def test_rejected_finance_target_gets_a_terminal_receipt(tmp_path, clock, mock_http):
    mock_http(lambda _: gold_response(clock))
    async with collector_runtime(tmp_path) as (context, inbox, append):
        step = TencentQuotesStep(instrument="gold", app_context=context, task_names=["finance", "missing"])
        assert (await step()).success
        (receipt,) = tmp_path.glob("resource/**/.delivery.json")
        assert (
            WorkspaceFiles(tmp_path).read_json(receipt, {})["rejected"]["missing"]
            == "proactive agent not found: missing"
        )
        assert (await inbox.snapshot("finance"))["pending_count"] == 1
        assert (await step()).success
        assert append.await_count == 1


@pytest.mark.asyncio
async def test_successful_append_without_delivery_receipt_is_idempotent_after_restart(
    tmp_path,
    clock,
    mock_http,
    monkeypatch,
):
    mock_http(lambda _: gold_response(clock))
    write = WorkspaceFiles.write_json

    def fail_receipt(files, path, value):
        if path.name == ".delivery.json" and not value["pending"]:
            raise OSError("receipt unavailable")
        return write(files, path, value)

    async with collector_runtime(tmp_path) as (context, inbox, _):
        with monkeypatch.context() as patch:
            patch.setattr(WorkspaceFiles, "write_json", fail_receipt)
            with pytest.raises(OSError, match="receipt unavailable"):
                await TencentQuotesStep(instrument="gold", app_context=context)()
        assert (await inbox.snapshot("finance"))["pending_count"] == 1
    async with collector_runtime(tmp_path) as (context, inbox, append):
        await TencentQuotesStep(instrument="gold", app_context=context)()
        append.assert_awaited_once()
        assert (await inbox.snapshot("finance"))["pending_count"] == 1
    assert len(list(tmp_path.glob("resource/**/metadata.json"))) == 1


@pytest.mark.asyncio
async def test_concurrent_collectors_do_not_duplicate_quotes(tmp_path, clock, mock_http):
    mock_http(lambda _: gold_response(clock))
    async with collector_runtime(tmp_path) as (context, inbox, append):
        await asyncio.gather(*(TencentQuotesStep(instrument="gold", app_context=context)() for _ in range(2)))
        append.assert_awaited_once()
        assert (await inbox.snapshot("finance"))["pending_count"] == 1


@pytest.mark.asyncio
async def test_http_errors_retry_before_publishing(tmp_path, clock, mock_http, monkeypatch):
    responses = [httpx.Response(503), gold_response(clock)]
    requests = mock_http(lambda _: responses.pop(0))
    monkeypatch.setattr(TencentQuotesStep, "wait", lambda *_: None)
    async with collector_runtime(tmp_path) as (context, inbox, _):
        await TencentQuotesStep(instrument="gold", app_context=context)()
        assert (await inbox.snapshot("finance"))["pending_count"] == 1
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_scope_denial_precedes_history_reads_and_network(tmp_path, clock, mock_http):
    requests = mock_http(lambda _: gold_response(clock))
    async with collector_runtime(tmp_path) as (context, _, append):
        with pytest.raises(PermissionError):
            await TencentQuotesStep(instrument="gold", app_context=context)(RuntimeContext(_allowed_paths=["resource"]))
        append.assert_not_awaited()
    assert not requests and not (tmp_path / "resource").exists()


@pytest.mark.asyncio
async def test_gold_saves_only_changed_xau_quotes(tmp_path, clock, mock_http):
    original_time = clock.now.return_value - timedelta(seconds=1)
    price, quoted_at = "2420", original_time
    requests = mock_http(lambda _: gold_response(clock, price=price, quoted_at=quoted_at))
    async with collector_runtime(tmp_path) as (context, inbox, append):
        assert (await TencentQuotesStep(instrument="gold", app_context=context)()).success
        event = (await inbox.claim("finance"))["items"][0]
        assert event["source"] == "gold_quotes"
        quote = MarketQuote.model_validate_json((tmp_path / event["paths"][0]).read_text())
        assert (quote.symbol, quote.currency, quote.source) == ("XAUUSD", "USD", "tencent")
        assert (quote.price, quote.previous_close, quote.change_percent) == (2420, 2200, 10)
        assert quote.quoted_at == original_time
        assert quote.collected_at == clock.now.return_value

        clock.now.return_value += timedelta(minutes=5)
        quoted_at += timedelta(minutes=5)
        assert (await TencentQuotesStep(instrument="gold", app_context=context)()).answer == {"skipped": "no_new_data"}

        clock.now.return_value += timedelta(minutes=5)
        price, quoted_at = "2430", original_time - timedelta(seconds=1)
        assert (await TencentQuotesStep(instrument="gold", app_context=context)()).answer == {"skipped": "no_new_data"}
        assert append.await_count == 1

    async with collector_runtime(tmp_path) as (context, inbox, _):
        quoted_at = clock.now.return_value - timedelta(seconds=1)
        assert (await TencentQuotesStep(instrument="gold", app_context=context)()).success
        assert (await inbox.snapshot("finance"))["pending_count"] == 2

    assert len(requests) == 4
    assert all(
        request.url.host == "qt.gtimg.cn" and dict(request.url.params) == {"q": "hf_XAU"} for request in requests
    )


@pytest.mark.parametrize("kind", ["missing", "zero_close", "nan_price", "future_timestamp"])
@pytest.mark.asyncio
async def test_invalid_gold_quotes_publish_nothing(tmp_path, clock, mock_http, kind):
    def respond(_):
        if kind == "missing":
            return httpx.Response(200, text='v_hf_XAU="";')
        if kind == "zero_close":
            return gold_response(clock, previous_close="0")
        if kind == "nan_price":
            return gold_response(clock, price="NaN")
        return gold_response(clock, quoted_at=clock.now.return_value + timedelta(minutes=5))

    mock_http(respond)
    async with collector_runtime(tmp_path) as (context, inbox, append):
        with pytest.raises(ValueError):
            await TencentQuotesStep(instrument="gold", app_context=context)()
        append.assert_not_awaited()
        assert (await inbox.snapshot("finance"))["pending_count"] == 0
    assert not list(tmp_path.glob("resource/**/metadata.json"))


@pytest.mark.asyncio
async def test_btc_collects_on_weekends_and_deduplicates_the_observed_minute(tmp_path, clock, mock_http):
    clock.now.return_value = datetime(2026, 11, 15, 10, 0, tzinfo=_SHANGHAI)
    requests = mock_http(lambda _: httpx.Response(200, json=btc_statistics(clock)))
    async with collector_runtime(tmp_path) as (context, inbox, append):
        await BtcQuotesStep(app_context=context)()
        event = (await inbox.claim("finance"))["items"][0]
        assert event["source"] == "btc_quotes"
        quote = CryptoQuote.model_validate_json((tmp_path / event["paths"][0]).read_text())
        assert quote.symbol == "BTCUSDT" and quote.currency == "USDT"
        assert quote.price == 66000 and quote.change_24h_percent == 10
        assert quote.window_end - quote.window_start == timedelta(hours=24)
        assert quote.collected_at == clock.now.return_value
        assert "previous_close" not in quote.model_dump()
        assert (await BtcQuotesStep(app_context=context)()).answer == {"skipped": "no_new_data"}
        clock.now.return_value += timedelta(minutes=1)
        await BtcQuotesStep(app_context=context)()
        assert (await inbox.snapshot("finance"))["pending_count"] == 1
        assert append.await_count == 1
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
        append.return_value = {"deliveries": {}, "rejected": {}, "errors": {"finance": "queue full"}}
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


@pytest.mark.asyncio
async def test_btc_business_changes_survive_restart_and_ignore_window_only_updates(tmp_path, clock, mock_http):
    price, opening = "66000", "60000"

    def respond(_):
        return httpx.Response(200, json={**btc_statistics(clock), "lastPrice": price, "openPrice": opening})

    mock_http(respond)
    for price, opening, count in (
        ("66000", "60000", 1),
        ("66000", "60000", 1),
        ("67000", "60000", 2),
        ("66000", "60000", 3),
        ("66000", "61000", 4),
    ):
        async with collector_runtime(tmp_path) as (context, inbox, _):
            await BtcQuotesStep(app_context=context)()
            assert (await inbox.snapshot("finance"))["pending_count"] == count
        clock.now.return_value += timedelta(minutes=5)
    assert len(list(tmp_path.glob("resource/btc_quotes/*/*/btc.jsonl"))) == 4


@pytest.mark.asyncio
async def test_btc_older_statistics_do_not_replace_the_latest_observation(tmp_path, clock, mock_http):
    payload = btc_statistics(clock)
    mock_http(lambda _: httpx.Response(200, json=payload))
    async with collector_runtime(tmp_path) as (context, inbox, append):
        await BtcQuotesStep(app_context=context)()
        clock.now.return_value += timedelta(minutes=5)
        payload["lastPrice"] = "67000"
        payload["closeTime"] -= 1000
        assert (await BtcQuotesStep(app_context=context)()).answer == {"skipped": "no_new_data"}
        assert append.await_count == 1 and (await inbox.snapshot("finance"))["pending_count"] == 1


@pytest.mark.asyncio
async def test_news_uses_existing_job_and_source_files_deduplicate_after_restart(tmp_path, clock, mock_http):
    requests = mock_http(lambda _: news_response(news_batch(clock)))
    async with collector_runtime(tmp_path) as (context, inbox, append):
        response = await FinanceNewsStep(app_context=context, request_interval=0)()
        assert response.success
        event = (await inbox.claim("finance"))["items"][0]
        assert event["source"] == "finance_news"
        assert append.await_args.kwargs["task_names"] == ["finance"]
        assert append.await_args.kwargs["events"][0].id == event["id"]
        (markdown_path,) = event["paths"]
        markdown = tmp_path / markdown_path
        assert markdown.name == "news.md"
        assert "[AI 新闻](https://www.cls.cn/detail/2)" in markdown.read_text()
        assert "AI news full text" in markdown.read_text()
        item = FinanceNewsItem.model_validate_json(markdown.with_name("news.jsonl").read_text())
        assert item.news_id == "2" and item.content == "AI news full text"
        assert item.url == "https://www.cls.cn/detail/2"
        assert WorkspaceFiles(tmp_path).read_json(markdown.with_name(".delivery.json"), {})["pending"] == []
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
            for line in (tmp_path / event["paths"][0]).with_name("news.jsonl").read_text().splitlines()
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
    assert not list(tmp_path.glob("resource/*/*/*"))
    assert not list(tmp_path.glob(".reme-resource-*"))


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


@pytest.mark.asyncio
async def test_news_advances_progress_on_empty_polls_and_keeps_an_overlap_after_restart(tmp_path, clock, mock_http):
    start = clock.now.return_value
    observed = []

    def respond(request):
        cursor = int(request.url.params["last_time"])
        observed.append(cursor)
        return news_response(
            [news_row(2, start - timedelta(minutes=1)), news_row(1, start - timedelta(hours=25))],
        )

    mock_http(respond)
    async with collector_runtime(tmp_path) as (context, _, _):
        await FinanceNewsStep(app_context=context)()
    clock.now.return_value += timedelta(minutes=5)
    async with collector_runtime(tmp_path) as (context, _, append):
        assert (await FinanceNewsStep(app_context=context)()).answer == {"skipped": "no_new_data"}
        append.assert_not_awaited()
    saved = WorkspaceFiles(tmp_path).read_json(tmp_path / "resource/finance_news/progress.json", None)
    assert saved == {"checked_until": clock.now.return_value.isoformat()}
    assert len(observed) == 2
    clock.now.return_value += timedelta(minutes=5)
    with pytest.MonkeyPatch.context() as patch:
        fetch = Mock(return_value=[])
        patch.setattr(FinanceNewsStep, "_fetch", fetch)
        async with collector_runtime(tmp_path) as (context, _, _):
            await FinanceNewsStep(app_context=context)()
        assert fetch.call_args.args[0] == start - timedelta(minutes=5)


@pytest.mark.asyncio
async def test_news_progress_is_not_advanced_when_manifest_publication_fails(tmp_path, clock, mock_http, monkeypatch):
    mock_http(lambda _: news_response(news_batch(clock)))
    write = WorkspaceFiles.write_json

    def fail_manifest(files, path, value):
        if path.name == "metadata.json":
            raise OSError("manifest unavailable")
        return write(files, path, value)

    async with collector_runtime(tmp_path) as (context, inbox, _):
        with monkeypatch.context() as patch:
            patch.setattr(WorkspaceFiles, "write_json", fail_manifest)
            with pytest.raises(OSError, match="manifest unavailable"):
                await FinanceNewsStep(app_context=context)()
        assert not (tmp_path / "resource/finance_news/progress.json").exists()
        await FinanceNewsStep(app_context=context)()
        assert (await inbox.snapshot("finance"))["pending_count"] == 1


@pytest.mark.asyncio
async def test_news_overlap_keeps_late_records_without_repeating_known_ids(tmp_path, clock, mock_http):
    start = clock.now.return_value
    rows = news_batch(clock)
    mock_http(lambda _: news_response(rows))
    async with collector_runtime(tmp_path) as (context, inbox, append):
        await FinanceNewsStep(app_context=context)()
        clock.now.return_value += timedelta(minutes=5)
        identical = {**news_row(3, start + timedelta(minutes=4), "新消息"), "title": "新消息"}
        rows = [identical, news_row(4, start - timedelta(minutes=2)), *rows]
        result = await FinanceNewsStep(app_context=context)()
        package = (tmp_path / result.answer["path"]).parent
        lines = package.joinpath("news.jsonl").read_text().splitlines()
        assert {FinanceNewsItem.model_validate_json(line).news_id for line in lines} == {"3", "4"}
        markdown = package.joinpath("news.md").read_text()
        assert markdown.count("新消息") == 1
        assert "/detail/3)" in markdown and "/detail/4)" in markdown and "/detail/2)" not in markdown
        assert append.await_count == 2 and (await inbox.snapshot("finance"))["pending_count"] == 2
