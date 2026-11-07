"""CLS collection, incremental files, and failures without external requests."""

# pylint: disable=protected-access

import asyncio
from datetime import datetime, timedelta
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx
import pytest

from reme.components.application_context import ApplicationContext
from reme.components.job import BaseJob
from reme.components.runtime_context import RuntimeContext
from reme.config.config_parser import resolve_app_config
from reme.schema import FinanceNewsItem
from reme.steps.proactive.data_connectors import FinanceNewsStep

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _row(news_id, when, **kwargs):
    return {"id": news_id, "ctime": int(when.timestamp()), "title": "新闻标题", "content": "新闻正文", **kwargs}


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(name="clock")
def _clock(monkeypatch):
    state = SimpleNamespace(now=datetime(2026, 11, 7, 15, 30, 0, 123456, tzinfo=SHANGHAI))

    class Clock(datetime):
        """Fix collection timestamps without adding public replay parameters."""

        @classmethod
        def now(cls, tz=None):
            return state.now.astimezone(tz)

    monkeypatch.setattr("reme.steps.proactive.data_connectors.base_connector.datetime", Clock)
    return state


@pytest.fixture(name="cls")
def _cls(monkeypatch, clock):
    state = SimpleNamespace(requests=[], threads=[], clients=[], waits=[], outcomes=[])
    state.rows = [_row(1, clock.now - timedelta(hours=1)), _row(0, clock.now - timedelta(hours=25))]
    original_client = httpx.Client
    state.native_wait = FinanceNewsStep._wait

    def handle(request):
        state.requests.append(request)
        state.threads.append(threading.get_ident())
        if state.outcomes:
            outcome = state.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            if isinstance(outcome, httpx.Response):
                return outcome
            return httpx.Response(200, json={"errno": 0, "data": {"roll_data": outcome}})
        return httpx.Response(200, json={"errno": 0, "data": {"roll_data": state.rows}})

    def client(**kwargs):
        instance = original_client(transport=httpx.MockTransport(handle), **kwargs)
        state.clients.append(instance)
        return instance

    def wait(step, seconds):
        step._check_stopped(step._cancelled)
        state.waits.append(seconds)

    monkeypatch.setattr("reme.steps.proactive.data_connectors.finance_news.httpx.Client", client)
    monkeypatch.setattr(FinanceNewsStep, "_wait", wait)
    return state


def _step(tmp_path, **kwargs):
    app = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai")
    return FinanceNewsStep(app_context=app, request_interval=0, **kwargs)


@pytest.mark.asyncio
async def test_finance_news_covers_full_window_and_saves_only_reading_instructions(tmp_path, clock, cls):
    """Window edges, future and broken rows, duplicate pages, HTML, and original text survive normalization."""
    end = clock.now.replace(microsecond=0)
    cls.outcomes = [
        [
            _row(1, end, title="<b>标题甲</b>", content="<p>甲&amp;乙</p><style>隐藏</style><p>丙</p>"),
            _row(2, end - timedelta(hours=1)),
            _row(9, end + timedelta(seconds=1)),
            {"id": "broken", "ctime": int(end.timestamp())},
            "broken row",
        ],
        [
            _row(2, end - timedelta(hours=1)),
            _row(3, end - timedelta(hours=24)),
            _row(4, end - timedelta(hours=24, seconds=1)),
            _row(5, end - timedelta(hours=2), title="", content=""),
        ],
    ]
    app = ApplicationContext(workspace_dir=str(tmp_path), resource_dir="assets", timezone="Pacific/Honolulu")
    result = (await FinanceNewsStep(app_context=app, request_interval=0)()).answer
    assert set(result) == {"context", "paths"}
    assert result["paths"] == ["assets/finance_news/2026-11-06_21-30-00-123456.jsonl"]
    assert "3 new CLS" in result["context"]
    assert "Read the JSONL file" in result["context"]
    assert "甲&乙" not in result["context"]
    assert "2026-11-06T15:30:00+08:00 to 2026-11-07T15:30:00+08:00" in result["context"]
    rows = _read(tmp_path / result["paths"][0])
    assert [row["news_id"] for row in rows] == ["3", "2", "1"]
    assert rows[-1] == {
        "news_id": "1",
        "event_time": end.isoformat(),
        "source": "财联社",
        "title": "标题甲",
        "content": "甲&乙 丙",
    }
    assert all(FinanceNewsItem.model_validate(row) for row in rows)
    assert int(cls.requests[1].url.params["last_time"]) == int((end - timedelta(hours=1)).timestamp())
    assert all(request.url.params["rn"] == "50" and request.url.params["sign"] for request in cls.requests)
    assert all(client.is_closed for client in cls.clients)
    assert all(thread != threading.get_ident() for thread in cls.threads)
    assert not list(tmp_path.rglob("metadata.json"))
    assert not list(tmp_path.glob(".reme-resource-*"))


@pytest.mark.asyncio
async def test_retained_files_deduplicate_after_restart_and_only_new_ids_get_a_file(tmp_path, clock, cls):
    """Each call constructs a fresh application, retaining no in-memory ID cache."""
    first = (await _step(tmp_path)()).answer
    old_path = tmp_path / first["paths"][0]
    old_bytes = old_path.read_bytes()
    clock.now += timedelta(seconds=1)
    cls.rows[0]["content"] = "同 ID 的正文修订"
    repeated = (await _step(tmp_path)()).answer
    assert repeated["paths"] == []
    assert "No new CLS" in repeated["context"]
    cls.rows.insert(0, _row(2, clock.now - timedelta(minutes=1)))
    clock.now += timedelta(seconds=1)
    added = (await _step(tmp_path)()).answer
    assert len(added["paths"]) == 1
    assert [row["news_id"] for row in _read(tmp_path / added["paths"][0])] == ["2"]
    assert old_path.read_bytes() == old_bytes
    assert len(list((tmp_path / "resource/finance_news").glob("*.jsonl"))) == 2


@pytest.mark.asyncio
async def test_concurrent_calls_publish_each_id_once(tmp_path, cls):
    """Independent callers share the source-file deduplication boundary."""
    results = await asyncio.gather(*(_step(tmp_path)() for _ in range(3)))
    paths = [path for response in results for path in response.answer["paths"]]
    assert len(paths) == 1
    assert [row["news_id"] for row in _read(tmp_path / paths[0])] == ["1"]
    assert len(cls.requests) == 3


@pytest.mark.asyncio
async def test_successful_empty_window_creates_no_file(tmp_path, clock, cls):
    """A fully covered window can legitimately contain no valid current items."""
    cls.rows = [_row(1, clock.now - timedelta(hours=25))]
    result = (await _step(tmp_path)()).answer
    assert result["paths"] == []
    assert not list(tmp_path.rglob("*.jsonl"))


@pytest.mark.asyncio
async def test_oversized_collection_is_not_published_as_unreadable_history(tmp_path, cls):
    """Every published batch must fit the next call's complete-history read limit."""
    with pytest.raises(ValueError, match="collection exceeds max_file_bytes"):
        await _step(tmp_path, max_file_bytes=10)()
    assert len(cls.requests) == 1
    assert not list(tmp_path.rglob("*.jsonl"))
    assert not list(tmp_path.glob(".reme-resource-*"))


@pytest.mark.asyncio
async def test_transient_page_failure_retries_and_preserves_rate_limit(tmp_path, cls):
    """Every request attempt observes the configured delay, including retries."""
    cls.outcomes = [httpx.ReadTimeout("temporary failure"), cls.rows]
    step = FinanceNewsStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    result = (await step()).answer
    assert len(result["paths"]) == 1
    assert len(cls.requests) == 2
    assert cls.waits == [1, 10, 10]
    assert all(client.is_closed for client in cls.clients)


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [[], {"errno": 9}, {"errno": 0}, {"errno": 0, "data": {"roll_data": {}}}])
async def test_invalid_api_responses_fail_without_publication(tmp_path, cls, payload):
    """Unexpected API envelopes cannot masquerade as a successful empty window."""
    cls.outcomes = [httpx.Response(200, json=payload) for _ in range(2)]
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        await _step(tmp_path, max_retries=2)()
    assert len(cls.requests) == 2
    assert not list(tmp_path.rglob("*.jsonl"))
    assert not list(tmp_path.glob(".reme-resource-*"))


@pytest.mark.asyncio
async def test_exhausted_http_errors_preserve_history(tmp_path, cls):
    """Failure on a later call never rewrites previously published news."""
    first = (await _step(tmp_path)()).answer
    path = tmp_path / first["paths"][0]
    before = path.read_bytes()
    cls.outcomes = [httpx.Response(503) for _ in range(3)]
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        await _step(tmp_path)()
    assert path.read_bytes() == before
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["empty", "timestamps", "cursor", "partial"])
async def test_incomplete_or_stalled_pagination_never_publishes_partial_news(tmp_path, clock, cls, failure):
    """Returning a partial rolling window would silently lose observations."""
    if failure == "empty":
        cls.rows = []
    elif failure == "timestamps":
        cls.rows = [{"id": 1, "ctime": "bad"}]
    elif failure == "cursor":
        cls.rows = [_row(1, clock.now)]
    else:
        cls.outcomes = [[_row(1, clock.now - timedelta(hours=1))], []]
    with pytest.raises(RuntimeError):
        await _step(tmp_path)()
    assert not list(tmp_path.rglob("*.jsonl"))


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["json", "schema", "symlink", "directory", "oversized"])
async def test_invalid_history_fails_closed_before_requesting_news(tmp_path, cls, kind):
    """Unreadable deduplication evidence must not trigger duplicate publication."""
    target = tmp_path / "resource/finance_news"
    target.mkdir(parents=True)
    path = target / "history.jsonl"
    outside = tmp_path / "preserve.txt"
    outside.write_text("preserve", encoding="utf-8")
    if kind == "symlink":
        path.symlink_to(outside)
    elif kind == "directory":
        path.mkdir()
    elif kind == "schema":
        path.write_text('{"news_id":"1"}\n', encoding="utf-8")
    else:
        path.write_text("broken JSON\n", encoding="utf-8")
    with pytest.raises(ValueError):
        await _step(tmp_path, max_file_bytes=3 if kind == "oversized" else 1024)()
    assert cls.requests == []
    assert path.exists()
    assert outside.read_text(encoding="utf-8") == "preserve"


@pytest.mark.asyncio
async def test_publication_failure_does_not_mark_ids_as_seen(tmp_path, clock, cls, monkeypatch):
    """Only a successfully published source file advances deduplication."""
    first = (await _step(tmp_path)()).answer
    old = tmp_path / first["paths"][0]
    before = old.read_bytes()
    cls.rows.insert(0, _row(2, clock.now - timedelta(minutes=1)))
    clock.now += timedelta(seconds=1)
    original = Path.hardlink_to

    def fail(path, source):
        raise OSError("publication failed")

    monkeypatch.setattr(Path, "hardlink_to", fail)
    with pytest.raises(OSError, match="publication failed"):
        await _step(tmp_path)()
    assert list(old.parent.iterdir()) == [old]
    assert old.read_bytes() == before
    monkeypatch.setattr(Path, "hardlink_to", original)
    result = (await _step(tmp_path)()).answer
    assert [row["news_id"] for row in _read(tmp_path / result["paths"][0])] == ["2"]
    assert not list(tmp_path.glob(".reme-resource-*"))


@pytest.mark.asyncio
@pytest.mark.parametrize("signal", ["cancel", "stop"])
async def test_rate_limit_wait_is_interruptible_without_publication(tmp_path, cls, monkeypatch, signal):
    """Neither cancellation nor a stop waits out the full rate-limit delay."""
    started = asyncio.Event()
    loop = asyncio.get_running_loop()

    def wait(step, seconds):
        loop.call_soon_threadsafe(started.set)
        cls.native_wait(step, seconds)

    monkeypatch.setattr(FinanceNewsStep, "_wait", wait)
    context = RuntimeContext(stop_event=asyncio.Event(), request_interval=60)
    task = asyncio.create_task(_step(tmp_path)(context))
    await asyncio.wait_for(started.wait(), timeout=3)
    if signal == "cancel":
        task.cancel()
    else:
        context.stop_event.set()
    with pytest.raises(asyncio.CancelledError if signal == "cancel" else RuntimeError):
        await asyncio.wait_for(task, timeout=2)
    assert all(client.is_closed for client in cls.clients)
    assert not list(tmp_path.rglob("*.jsonl"))
    assert not list(tmp_path.glob(".reme-resource-*"))


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [[], ["daily"], ["../outside"]])
async def test_scope_denial_prevents_history_reads_and_network(tmp_path, cls, scope):
    """An unrelated or invalid scope rejects the complete collection operation."""
    with pytest.raises(PermissionError):
        await _step(tmp_path)(_allowed_paths=scope)
    assert cls.requests == []
    assert not list(tmp_path.glob(".reme-resource-*"))


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [{"request_interval": -1}, {"request_interval": "nan"}, {"max_retries": 0}])
async def test_invalid_request_settings_fail_before_network(tmp_path, cls, kwargs):
    """Invalid configuration must not be silently clamped or send requests."""
    with pytest.raises(ValueError):
        await _step(tmp_path)(**kwargs)
    assert cls.requests == []


@pytest.mark.asyncio
async def test_configured_job_uses_registered_connector_without_tool_arguments(tmp_path, cls):
    """The public no-argument tool binds to the built-in connector and defaults."""
    app = ApplicationContext(**resolve_app_config(config="proactive", log_config=False, workspace_dir=str(tmp_path)))
    job = app.app_config.jobs["finance_news"]
    assert job.parameters["properties"] == {}
    async with BaseJob(name="finance_news", app_context=app, **job.model_dump()) as configured:
        response = await configured()
    assert response.success
    result = response.answer
    assert len(result["paths"]) == 1
    assert _read(tmp_path / result["paths"][0])[0]["source"] == "财联社"
    assert len(cls.requests) == 1
