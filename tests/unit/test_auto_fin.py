"""Focused tests for the four-step Auto Fin workflow."""

# pylint: disable=missing-function-docstring,protected-access

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from reme.components import ApplicationContext
from reme.components.agent_wrapper.base_agent_wrapper import BaseAgentWrapper
from reme.components.runtime_context import RuntimeContext
from reme.config.config_parser import _load_config
from reme.schema import (
    AutoFinEtfHistoricalEvents,
    AutoFinEtfHistoricalResearch,
    AutoFinEtfSelection,
    AutoFinEtfsOutput,
    AutoFinHistoricalEvent,
    AutoFinHistoricalEventReference,
    AutoFinMarketSelection,
    AutoFinMarketSample,
    AutoFinReportOutput,
)
from reme.steps.cookbook.auto_fin._base import _write
from reme.steps.cookbook.auto_fin.data import AutoFinDataStep
from reme.steps.cookbook.auto_fin.history import AutoFinHistoryStep
from reme.steps.cookbook.auto_fin.history_search import AutoFinHistorySearchStep
from reme.steps.cookbook.auto_fin.merge import AutoFinMergeStep
from reme.steps.cookbook.auto_fin.market import AutoFinMarketStep
from reme.steps.cookbook.auto_fin.topic import AutoFinTopicStep, _plain_text


def test_atomic_write_preserves_existing_file_and_cleans_temporary_file_on_failure(
    tmp_path: Path,
    monkeypatch,
):
    path = tmp_path / "result.json"
    path.write_text("existing", encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("replace failed")

    monkeypatch.setattr("reme.steps.cookbook.auto_fin._base.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        _write(path, "replacement")

    assert path.read_text(encoding="utf-8") == "existing"
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.asyncio
async def test_read_jsonl_preserves_unicode_line_separator(tmp_path: Path):
    path = tmp_path / "news.jsonl"
    rows = [{"title": "包含\u2028行分隔符"}, {"title": "下一条"}]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    assert await AutoFinDataStep._read_jsonl(path) == rows


def test_plain_news_text_removes_markup_images_and_hidden_content():
    content = '<p>甲&amp;乙</p><img src="https://example.com/large.png"><style>隐藏样式</style><p>丙</p>'

    assert _plain_text(content) == "甲&乙 丙"


def test_topic_repairs_unknown_news_id_by_unique_content_hash():
    output = AutoFinEtfsOutput.model_validate(
        {
            "etfs": [
                {
                    "etf_code": "159819.SZ",
                    "etf_name": "人工智能ETF",
                    "events": [{"reason": "穆迪警告AI投资风险", "news_id": "20260725061826_1c76"}],
                },
            ],
        },
    )
    news = [
        {"news_id": "20260725050638_1c76"},
        {"news_id": "20260725061826_765a"},
    ]

    repaired, repairs = AutoFinTopicStep._repair_news_ids(output, news)

    assert repaired.etfs[0].events[0].news_id == "20260725050638_1c76"
    assert repairs == {"20260725061826_1c76": "20260725050638_1c76"}


def test_topic_does_not_repair_ambiguous_content_hash():
    output = AutoFinEtfsOutput.model_validate(
        {
            "etfs": [
                {
                    "etf_code": "159819.SZ",
                    "etf_name": "人工智能ETF",
                    "events": [{"reason": "相关事件", "news_id": "20260725061826_abcd"}],
                },
            ],
        },
    )
    news = [{"news_id": "20260725050638_abcd"}, {"news_id": "20260725070000_abcd"}]

    repaired, repairs = AutoFinTopicStep._repair_news_ids(output, news)

    assert repaired.etfs[0].events[0].news_id == "20260725061826_abcd"
    assert not repairs


def test_published_time_is_normalized_once_to_shanghai_local_time():
    parsed = AutoFinDataStep._published_at({"published_at": "2026-07-24T01:00:00+00:00"})

    assert parsed == datetime(2026, 7, 24, 9)
    assert parsed.tzinfo is None


@pytest.mark.asyncio
async def test_current_news_keeps_all_items_with_per_item_and_total_content_limits(tmp_path: Path):
    day_dir = tmp_path / "daily" / "2026-07-24"
    day_dir.mkdir(parents=True)
    rows = [
        {
            "title": f"新闻标题{index}",
            "pub_time": f"2026-07-24 09:0{index}:00",
            "src": "财联社",
            "content": f"<p>正文内容{index}ABCDEFGHIJ</p>",
        }
        for index in range(3)
    ]
    (day_dir / "auto_fin_news_data.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    step = AutoFinTopicStep(app_context=ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai"))
    step.context = RuntimeContext(
        auto_fin_news_start="2026-07-24",
        auto_fin_date="2026-07-24",
        news_title_max_chars=5,
        news_content_max_chars=10,
        news_total_content_max_chars=12,
    )

    news = await step._current_news(
        datetime.fromisoformat("2026-07-24T08:59:00"),
        datetime.fromisoformat("2026-07-24T10:00:00"),
    )

    assert len(news) == 3
    assert all(len(row["title"]) <= 5 for row in news)
    assert all(len(row["content"]) <= 4 for row in news)
    assert sum(len(row["content"]) for row in news) <= 12
    assert all("<" not in row["content"] for row in news)
    assert {row["news_id"] for row in news} == {
        f"20260724090{index}00_" f"{hashlib.sha256(f'财联社<p>正文内容{index}ABCDEFGHIJ</p>'.encode()).hexdigest()[:4]}"
        for index in range(3)
    }


@pytest.mark.asyncio
async def test_data_fills_missing_news_refreshes_today_and_force_refreshes_history(
    tmp_path: Path,
):
    calls = []

    def provider(endpoint: str, **kwargs):
        calls.append((endpoint, kwargs))
        if endpoint == "major_news":
            day = kwargs["start_date"][:10]
            return [
                {
                    "title": day,
                    "pub_time": f"{day} 08:00:00",
                    "src": "财联社",
                    "content": day,
                },
            ]
        raise AssertionError(endpoint)

    app_context = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai")
    context = RuntimeContext(
        date="2026-07-24",
        now="2026-07-24T09:30:00+08:00",
        lookback_days=3,
        progress_interval=30,
        trade_dates=["2026-07-23"],
        tushare_provider=provider,
    )

    await AutoFinDataStep(app_context=app_context)(context)
    assert [endpoint for endpoint, _ in calls] == ["major_news"] * 3
    for day in ("2026-07-22", "2026-07-23", "2026-07-24"):
        [row] = AutoFinDataStep._read_jsonl_sync(
            tmp_path / "daily" / day / "auto_fin_news_data.jsonl",
        )
        news_hash = hashlib.sha256(f"财联社{day}".encode()).hexdigest()[:4]
        assert row["news_id"] == f"{day.replace('-', '')}080000_{news_hash}"

    calls.clear()
    await AutoFinDataStep(app_context=app_context)(context)
    assert [endpoint for endpoint, _ in calls] == ["major_news"]
    assert calls[0][1]["start_date"] == "2026-07-24 00:00:00"
    assert calls[0][1]["end_date"] == "2026-07-24 09:30:00"

    calls.clear()
    context["force"] = True
    await AutoFinDataStep(app_context=app_context)(context)
    assert [endpoint for endpoint, _ in calls] == ["major_news"] * 3
    assert context["auto_fin_previous_trade_date"] == "2026-07-23"


@pytest.mark.asyncio
async def test_data_deduplicates_by_publish_time_and_short_hash(tmp_path: Path):
    def provider(endpoint: str, **_kwargs):
        assert endpoint == "major_news"
        return [
            {
                "title": "重复新闻的较晚记录",
                "pub_time": "2026-07-24 09:00:00",
                "src": "财联社",
                "content": "相同正文",
            },
            {
                "title": "另一条新闻",
                "pub_time": "2026-07-24 08:00:00",
                "src": "财联社",
                "content": "另一篇正文",
            },
            {
                "title": "重复新闻的最早记录",
                "pub_time": "2026-07-24 07:00:00",
                "src": "财联社",
                "content": "相同正文",
            },
            {
                "title": "同一时间和正文的重复记录",
                "pub_time": "2026-07-24 07:00:00",
                "src": "财联社",
                "content": "相同正文",
            },
        ]

    context = RuntimeContext(
        date="2026-07-24",
        now="2026-07-24T09:30:00+08:00",
        lookback_days=1,
        progress_interval=30,
        trade_dates=["2026-07-23"],
        tushare_provider=provider,
    )
    await AutoFinDataStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai"),
    )(context)

    rows = AutoFinDataStep._read_jsonl_sync(
        tmp_path / "daily" / "2026-07-24" / "auto_fin_news_data.jsonl",
    )
    assert [row["title"] for row in rows] == ["重复新闻的最早记录", "另一条新闻", "重复新闻的较晚记录"]
    assert len({row["news_id"] for row in rows}) == len(rows)


@pytest.mark.asyncio
async def test_topic_etfs_join_rank_deduplicate_name_and_index(tmp_path: Path):
    def provider(endpoint: str, **kwargs):
        if endpoint == "etf_basic":
            assert kwargs["list_status"] == "L"
            return [
                {"ts_code": "510001.SH", "csname": "同名 ETF", "index_code": "I1", "index_name": "指数一"},
                {"ts_code": "510002.SH", "csname": "同名ETF", "index_code": "I2", "index_name": "指数二"},
                {"ts_code": "510003.SH", "csname": "另一名称", "index_code": "I1", "index_name": "指数一"},
                {"ts_code": "510004.SH", "csname": "独立ETF", "index_code": "I4", "index_name": "指数四"},
            ]
        if endpoint == "fund_daily":
            assert kwargs["trade_date"] == "20260723"
            return [
                {"ts_code": "510002.SH", "amount": 90},
                {"ts_code": "510004.SH", "amount": 70},
                {"ts_code": "510001.SH", "amount": 100},
                {"ts_code": "510003.SH", "amount": 80},
                {"ts_code": "510001.SH", "amount": 95},
            ]
        raise AssertionError(endpoint)

    step = AutoFinTopicStep(app_context=ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai"))
    step.context = RuntimeContext(tushare_provider=provider, etf_candidate_limit=150)

    assert await step._filtered_etfs(date(2026, 7, 23)) == [
        {"code": "510001.SH", "name": "同名 ETF"},
        {"code": "510004.SH", "name": "独立ETF"},
    ]


class _Agent(BaseAgentWrapper):
    """Return deterministic structured replies for every analysis stage."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    async def reply(self, inputs, **kwargs):
        schema = kwargs["output_schema"]
        prompt = str(inputs)
        self.calls.append((schema, prompt, kwargs))
        assert "resume" not in kwargs
        assert "session_id" not in kwargs
        task = prompt
        assert "schema" not in task.casefold()
        assert "```json" in task
        if schema is AutoFinEtfsOutput:
            assert "filtered_news.jsonl" in task
            assert "filtered_etf.jsonl" in task
            assert "Top 150" in task
            assert "最多返回 20" in task
            value = {
                "etfs": [
                    {
                        "etf_code": "159018.SZ",
                        "etf_name": "油气ETF",
                        "events": [
                            {
                                "reason": "供应中断直接影响油气产业链盈利预期",
                                "news_id": (
                                    f"20260723160000_"
                                    f"{hashlib.sha256('财联社主要产油区供应中断'.encode()).hexdigest()[:4]}"
                                ),
                            },
                            {
                                "reason": "供应恢复时间影响油气价格预期",
                                "news_id": (
                                    f"20260724090000_"
                                    f"{hashlib.sha256('财联社供应恢复时间仍不确定'.encode()).hexdigest()[:4]}"
                                ),
                            },
                        ],
                    },
                ],
            }
        elif schema is AutoFinEtfHistoricalEvents:
            assert "memory_search" in task
            assert "不查询行情" in task
            assert "159018.SZ（油气ETF）" in task
            current_news_id = (
                f"20260724090000_" f"{hashlib.sha256('财联社供应恢复时间仍不确定'.encode()).hexdigest()[:4]}"
            )
            assert current_news_id not in task
            assert "当前事件时间线（仅作为检索线索，不含 news_id）" in task
            assert "时间、标题、正文和行情都由程序在你返回后补充" in task
            assert "每一项只包含 reason、news_id 和 source_path" in task
            tool_context_id = kwargs.get("tool_context_id", "")
            assert tool_context_id.startswith("auto_fin_history_01_159018.SZ_")
            assert tool_context_id not in task
            assert "系统会自动过滤本次研究中已经" in task
            self.app_context.metadata.setdefault("tool_contexts", {})[tool_context_id] = {
                "search_seen_chunk_ids": {},
            }
            value = {
                "etf_code": "159018.SZ",
                "etf_name": "油气ETF",
                "historical_events": [
                    {
                        "reason": "供应中断的事件类型和传导机制相同",
                        "news_id": (
                            f"20260601100000_" f"{hashlib.sha256('财联社历史供应中断'.encode()).hexdigest()[:4]}"
                        ),
                        "source_path": "daily/2026-06-01/auto_fin_news_data.jsonl",
                    },
                ],
            }
        elif schema is AutoFinMarketSelection:
            assert "ETF：159018.SZ（油气ETF）" in task
            assert "[2026-07-23T16:00:00] 原油供应中断" in task
            assert "判断影响方向相同还是相反" in task
            assert "不要依据" in task
            assert "程序会校验" in task
            assert "每项只包含 reason、news_id" in task
            assert "$tushare-data" not in task
            history_path = Path(
                next(line.strip() for line in task.splitlines() if line.strip().endswith("_output.json")),
            )
            history = json.loads(history_path.read_text(encoding="utf-8"))
            assert "historical_samples" not in history
            assert len(history["historical_events"]) == 1
            assert history["historical_events"][0]["event_title"] == "历史供应中断"
            assert len(history["historical_events"][0]["future_returns"]) == 10
            value = {
                "same_direction_events": [
                    {
                        "reason": "供应中断的事件类型和传导机制相同",
                        "news_id": history["historical_events"][0]["news_id"],
                    },
                ],
                "opposite_direction_events": [],
            }
        elif schema is AutoFinReportOutput:
            assert "不重新搜索新闻" in task
            assert "auto_fin_history_output.jsonl" in task
            assert "不生成 YAML frontmatter" in task
            assert '"etf_code": "159018.SZ"' in task
            assert '"suggested_holding_days": 10' in task
            assert '"horizon": 1' in task
            assert '"horizon": 10' in task
            assert "自行判断事件对 ETF 的影响方向" in task
            assert "不得使用程序计算结果反推事件方向" in task
            assert "以推荐 ETF 为主要内容" in task
            assert "用一句话合并简述" in task
            value = {
                "title": "Auto Fin ETF 结论",
                "body": "## 结论\n\n推荐 159018.SZ（油气ETF），参考持有 10 个交易日，"
                "当前加权预估收益 +10%；核心风险：供应恢复。",
            }
        else:  # pragma: no cover
            raise AssertionError(schema)
        return {"structured_output": schema.model_validate(value)}


@pytest.mark.asyncio
async def test_four_step_pipeline_writes_plain_markdown_and_cleans_temporary_data(
    tmp_path: Path,
):
    def provider(endpoint: str, **_kwargs):
        if endpoint == "major_news":
            return [
                {
                    "title": "原油供应中断",
                    "pub_time": "2026-07-23 16:00:00",
                    "src": "财联社",
                    "content": "主要产油区供应中断",
                },
                {
                    "title": "供应恢复时间不确定",
                    "pub_time": "2026-07-24 09:00:00",
                    "src": "财联社",
                    "content": "供应恢复时间仍不确定",
                },
            ]
        if endpoint == "etf_basic":
            return [
                {
                    "ts_code": "159018.SZ",
                    "csname": "油气ETF",
                    "index_code": "930987.CSI",
                    "index_name": "中证油气产业指数",
                    "list_status": "L",
                },
            ]
        if endpoint == "fund_daily":
            if "trade_date" in _kwargs:
                return [{"ts_code": "159018.SZ", "trade_date": "20260723", "amount": 1000}]
            trade_dates = [
                "20260601",
                "20260602",
                "20260603",
                "20260604",
                "20260605",
                "20260608",
                "20260609",
                "20260610",
                "20260611",
                "20260612",
                "20260615",
            ]
            return [
                {
                    "ts_code": "159018.SZ",
                    "trade_date": trade_date,
                    "open": 0.995 + index / 100,
                    "close": 1.0 + index / 100,
                }
                for index, trade_date in enumerate(trade_dates)
            ]
        if endpoint == "fund_adj":
            return [
                {"ts_code": "159018.SZ", "trade_date": trade_date, "adj_factor": 1.0}
                for trade_date in (
                    "20260601",
                    "20260602",
                    "20260603",
                    "20260604",
                    "20260605",
                    "20260608",
                    "20260609",
                    "20260610",
                    "20260611",
                    "20260612",
                    "20260615",
                )
            ]
        raise AssertionError(endpoint)

    app_context = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai")
    agent = _Agent(app_context=app_context)
    context = RuntimeContext(
        date="2026-07-24",
        now="2026-07-24T09:30:00+08:00",
        lookback_days=2,
        progress_interval=30,
        trade_dates=["2026-07-23"],
        tushare_provider=provider,
    )

    await AutoFinDataStep(app_context=app_context)(context)
    historical_path = tmp_path / "daily" / "2026-06-01" / "auto_fin_news_data.jsonl"
    historical_path.parent.mkdir(parents=True)
    historical_content = "历史供应中断"
    historical_news_id = f"20260601100000_" f"{hashlib.sha256(f'财联社{historical_content}'.encode()).hexdigest()[:4]}"
    historical_path.write_text(
        json.dumps(
            {
                "title": "历史供应中断",
                "pub_time": "2026-06-01 10:00:00",
                "src": "财联社",
                "content": historical_content,
                "news_id": historical_news_id,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    logs = []
    topic_step = AutoFinTopicStep(app_context=app_context, agent_wrapper=agent)
    history_step = AutoFinHistoryStep(app_context=app_context, agent_wrapper=agent)
    merge_step = AutoFinMergeStep(app_context=app_context, agent_wrapper=agent)
    for step in (topic_step, history_step, merge_step):
        step.logger = SimpleNamespace(info=logs.append, debug=lambda _message: None)
    await topic_step(context)
    await history_step(context)
    response = await merge_step(context)

    assert [schema for schema, _, _ in agent.calls] == [
        AutoFinEtfsOutput,
        AutoFinEtfHistoricalEvents,
        AutoFinMarketSelection,
        AutoFinReportOutput,
    ]
    assert "tool_contexts" not in app_context.metadata
    report = (tmp_path / "daily" / "2026-07-24" / "auto_fin.md").read_text(encoding="utf-8")
    assert report.startswith("# Auto Fin ETF 结论\n\n")
    assert not report.startswith("---")
    detail = context["auto_fin_history_details"][0]
    assert detail["etf"]["etf_code"] == "159018.SZ"
    first_event = detail["etf"]["events"][0]
    assert first_event["reason"] == "供应中断直接影响油气产业链盈利预期"
    assert first_event["news_id"].startswith("20260723160000_")
    analysis = detail["market_analysis"]
    assert analysis["matched_historical_events"][0]["weight"] == 1.0
    assert analysis["matched_historical_events"][0]["news_id"] == historical_news_id
    assert analysis["matched_historical_events"][0]["direction"] == "same"
    assert analysis["forecast"]["suggested_holding_days"] == 10
    assert analysis["forecast"]["returns"][-1]["expected_return"] == pytest.approx(0.1)
    assert "calculation_code" not in analysis
    assert detail["historical_research"]["historical_events"][0]["event_content"] == "历史供应中断"
    assert detail["historical_research"]["historical_events"][0]["reason"] == "供应中断的事件类型和传导机制相同"
    assert "当前加权预估收益 +10%" in report
    assert "历史事件" not in report
    assert not (tmp_path / "daily" / "2026-07-24" / "auto_fin_brief.md").exists()
    assert response.answer.startswith("## 结论")
    assert response.metadata["etf_count"] == 1
    assert context["markdown_path"] == "daily/2026-07-24/auto_fin.md"
    assert context["auto_fin_digest_path"] == "daily/2026-07-24/auto_fin.md"
    assert response.metadata["digest_path"] == "daily/2026-07-24/auto_fin.md"
    daily_index = (tmp_path / "daily" / "2026-07-24.md").read_text(encoding="utf-8")
    assert "[[daily/2026-07-24/auto_fin.md]]" in daily_index
    assert sum("agent input prompt=" in line for line in logs) == 2
    assert sum("agent output prompt=" in line for line in logs) == 2
    assert all("agent start prompt=" not in line and "agent done prompt=" not in line for line in logs)
    assert any('query="你只负责从候选文件中筛选' in line for line in logs)
    assert any('output={"etfs":[{"etf_code":"159018.SZ"' in line for line in logs)
    resource_dir = tmp_path / "resource" / "2026-07-24"
    filtered_news = AutoFinDataStep._read_jsonl_sync(resource_dir / "filtered_news.jsonl")
    filtered_etfs = AutoFinDataStep._read_jsonl_sync(resource_dir / "filtered_etf.jsonl")
    topic_etfs = AutoFinDataStep._read_jsonl_sync(resource_dir / "auto_fin_topic_output.jsonl")
    history_output = json.loads(
        (resource_dir / "auto_fin_history_01_159018.SZ_output.json").read_text(encoding="utf-8"),
    )
    history_details = AutoFinDataStep._read_jsonl_sync(resource_dir / "auto_fin_history_output.jsonl")
    assert not list(resource_dir.glob("*_input.md"))
    assert [row["news_id"] for row in filtered_news] == [event["news_id"] for event in detail["etf"]["events"]]
    assert filtered_etfs == [{"code": "159018.SZ", "name": "油气ETF"}]
    assert topic_etfs == [detail["etf"]]
    historical_event = history_output["historical_events"][0]
    assert historical_event["news_id"] == historical_news_id
    assert historical_event["event_time"] == "2026-06-01T10:00:00"
    assert historical_event["event_title"] == "历史供应中断"
    assert historical_event["market_entry"]["price_type"] == "close"
    assert [point["horizon"] for point in historical_event["future_returns"]] == list(
        range(1, 11),
    )
    assert historical_event["future_returns"][-1]["cumulative_return"] == pytest.approx(0.1)
    assert '\n  "etf_code"' in (resource_dir / "auto_fin_history_01_159018.SZ_output.json").read_text(encoding="utf-8")
    assert history_details == context["auto_fin_history_details"]


def test_historical_market_sample_rejects_look_ahead_and_incorrect_adjusted_returns():
    sample = {
        "event_time": "2026-06-01T10:00:00",
        "entry": {
            "entry_time": "2026-06-01T15:00:00",
            "trade_date": "2026-06-01",
            "price_type": "close",
            "raw_price": 1.0,
            "adj_factor": 1.2,
        },
        "future_returns": [
            {
                "horizon": 1,
                "trade_date": "2026-06-02",
                "raw_close": 1.1,
                "adj_factor": 1.2,
                "cumulative_return": 0.1,
            },
        ],
        "reaction_summary": "第一个有效收盘点上涨。",
    }

    assert AutoFinMarketSample.model_validate(sample).future_returns[0].cumulative_return == pytest.approx(0.1)

    sample["event_time"] = "2026-06-01T16:00:00"
    with pytest.raises(ValueError, match="entry must be strictly after the event"):
        AutoFinMarketSample.model_validate(sample)

    sample["event_time"] = "2026-06-01T10:00:00"
    sample["future_returns"][0]["cumulative_return"] = 0.2
    with pytest.raises(ValueError, match="incorrect adjusted return"):
        AutoFinMarketSample.model_validate(sample)


def test_market_calculation_equal_weights_and_reverses_opposite_direction_event():
    item = AutoFinEtfSelection.model_validate(
        {
            "etf_code": "518880.SH",
            "etf_name": "黄金ETF",
            "events": [{"reason": "黄金涨价", "news_id": "20260724090000_abcd"}],
        },
    )
    history = AutoFinEtfHistoricalResearch.model_validate(
        {
            "etf_code": "518880.SH",
            "etf_name": "黄金ETF",
            "historical_events": [
                {
                    "reason": "黄金价格方向相反",
                    "news_id": "20260601100000_abcd",
                    "source_path": "daily/2026-06-01/auto_fin_news_data.jsonl",
                    "event_time": "2026-06-01T10:00:00",
                    "event_title": "黄金价格下跌",
                    "event_content": "黄金价格出现明显下跌。",
                    "market_entry": {
                        "entry_time": "2026-06-01T15:00:00",
                        "trade_date": "2026-06-01",
                        "price_type": "close",
                        "raw_price": 1.0,
                        "adj_factor": 1.0,
                    },
                    "future_returns": [
                        {
                            "horizon": 1,
                            "trade_date": "2026-06-02",
                            "raw_close": 1.1,
                            "adj_factor": 1.0,
                            "cumulative_return": 0.1,
                        },
                    ],
                },
                {
                    "reason": "黄金价格方向相同",
                    "news_id": "20260602100000_efgh",
                    "source_path": "daily/2026-06-02/auto_fin_news_data.jsonl",
                    "event_time": "2026-06-02T10:00:00",
                    "event_title": "黄金价格上涨",
                    "event_content": "黄金价格出现明显上涨。",
                    "market_entry": {
                        "entry_time": "2026-06-02T15:00:00",
                        "trade_date": "2026-06-02",
                        "price_type": "close",
                        "raw_price": 1.0,
                        "adj_factor": 1.0,
                    },
                    "future_returns": [
                        {
                            "horizon": 1,
                            "trade_date": "2026-06-03",
                            "raw_close": 1.3,
                            "adj_factor": 1.0,
                            "cumulative_return": 0.3,
                        },
                    ],
                },
            ],
        },
    )
    selection = AutoFinMarketSelection.model_validate(
        {
            "same_direction_events": [
                {
                    "reason": "机制和价格方向均相同",
                    "news_id": "20260602100000_efgh",
                },
            ],
            "opposite_direction_events": [
                {
                    "reason": "机制可比但价格方向相反",
                    "news_id": "20260601100000_abcd",
                },
            ],
        },
    )

    analysis = AutoFinMarketStep._calculate_analysis(item, history, selection)

    assert [match.direction for match in analysis.matched_historical_events] == ["same", "opposite"]
    assert [match.weight for match in analysis.matched_historical_events] == [0.5, 0.5]
    assert analysis.forecast.returns[0].expected_return == pytest.approx(0.1)
    assert analysis.forecast.suggested_holding_days == 1
    assert "相似历史样本的收益方向存在分歧" in analysis.limitations


def test_market_selection_rejects_news_repeated_across_direction_groups():
    duplicate = {"reason": "方向判断", "news_id": "20260601100000_abcd"}

    with pytest.raises(ValueError, match="news IDs must be unique"):
        AutoFinMarketSelection.model_validate(
            {
                "same_direction_events": [duplicate],
                "opposite_direction_events": [duplicate],
            },
        )


@pytest.mark.asyncio
async def test_market_skips_agent_when_all_historical_references_were_invalid(tmp_path: Path):
    class _UnexpectedAgent(BaseAgentWrapper):
        async def reply(self, inputs, **kwargs):
            raise AssertionError((inputs, kwargs))

    app_context = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai")
    step = AutoFinMarketStep(app_context=app_context, agent_wrapper=_UnexpectedAgent(app_context=app_context))
    step.logger = SimpleNamespace(warning=lambda _message: None)
    step.context = RuntimeContext(
        auto_fin_current_etf={
            "etf_code": "159819.SZ",
            "etf_name": "人工智能ETF",
            "events": [{"reason": "AI事件", "news_id": "20260725050638_1c76"}],
        },
        auto_fin_current_events=[
            {
                "reason": "AI事件",
                "news_id": "20260725050638_1c76",
                "event_time": "2026-07-25T05:06:38",
                "event_title": "当前新闻",
                "event_content": "当前内容",
            },
        ],
        auto_fin_current_history={
            "etf_code": "159819.SZ",
            "etf_name": "人工智能ETF",
            "historical_events": [],
            "limitations": ["跳过无法解析的历史新闻 20260427100000_dead"],
        },
        auto_fin_current_history_resource=str(tmp_path / "history.json"),
        auto_fin_current_index=1,
        auto_fin_date="2026-07-26",
        auto_fin_decision_at="2026-07-26T18:00:00",
    )

    await step.execute()

    analysis = step.context["auto_fin_current_analysis"]
    assert not analysis["matched_historical_events"]
    assert "没有匹配的历史事件" in analysis["limitations"]
    assert (tmp_path / "resource" / "2026-07-26" / "auto_fin_market_01_159819.SZ_output.json").is_file()


@pytest.mark.asyncio
async def test_history_search_calculates_adjusted_returns_for_event_time_boundaries(tmp_path: Path):
    calls = []

    def provider(endpoint: str, **kwargs):
        calls.append((endpoint, kwargs))
        if endpoint == "fund_daily":
            return [
                {"trade_date": "20260609", "open": 7.5, "close": 8.0},
                {"trade_date": "20260608", "open": 6.0, "close": 7.0},
                {"trade_date": "20260605", "open": 10.0, "close": 11.0},
            ]
        if endpoint == "fund_adj":
            return [
                {"trade_date": "20260609", "adj_factor": 2.0},
                {"trade_date": "20260608", "adj_factor": 2.0},
                {"trade_date": "20260605", "adj_factor": 1.0},
            ]
        raise AssertionError(endpoint)

    step = AutoFinHistorySearchStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai"),
    )
    step.context = RuntimeContext(tushare_provider=provider)
    events = [
        AutoFinHistoricalEvent(
            reason="历史行情边界测试",
            news_id=f"{event_time.replace('-', '').replace(':', '').replace('T', '')}_abcd",
            source_path=f"daily/{event_time[:10]}/auto_fin_news_data.jsonl",
            event_time=event_time,
            event_title=label,
            event_content=label,
        )
        for event_time, label in (
            ("2026-06-05T08:00:00", "盘前事件"),
            ("2026-06-05T10:00:00", "盘中事件"),
            ("2026-06-05T16:00:00", "盘后事件"),
            ("2026-06-06T10:00:00", "休市日事件"),
        )
    ]

    samples, limitations = await step._calculate_samples(
        "159018.SZ",
        events,
        datetime.fromisoformat("2026-06-10T09:00:00"),
    )

    assert [sample.entry.price_type for sample in samples if sample.entry] == ["open", "close", "open", "open"]
    assert [sample.entry.trade_date.isoformat() for sample in samples if sample.entry] == [
        "2026-06-05",
        "2026-06-05",
        "2026-06-08",
        "2026-06-08",
    ]
    assert samples[0].future_returns[0].cumulative_return == pytest.approx(0.1)
    assert samples[1].future_returns[0].cumulative_return == pytest.approx(14 / 11 - 1)
    assert samples[2].future_returns[0].cumulative_return == pytest.approx(14 / 12 - 1)
    assert samples[3].future_returns[0].cumulative_return == pytest.approx(14 / 12 - 1)
    assert limitations
    assert [endpoint for endpoint, _ in calls] == ["fund_daily", "fund_adj"]


@pytest.mark.asyncio
async def test_history_search_recovers_source_path_from_news_id_date(tmp_path: Path):
    actual_path = tmp_path / "daily" / "2026-04-26" / "auto_fin_news_data.jsonl"
    actual_path.parent.mkdir(parents=True)
    actual_path.write_text(
        json.dumps(
            {
                "news_id": "20260426221449_de86",
                "pub_time": "2026-04-26 22:14:49",
                "title": "PCB厂商一季度业绩增长",
                "content": "AI硬件需求带动PCB订单增长。",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    step = AutoFinHistorySearchStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai"),
    )
    references = [
        AutoFinHistoricalEventReference(
            reason="PCB订单增长的传导机制相同",
            news_id="20260426221449_de86",
            source_path="daily/2026-04-27/auto_fin_news_data.jsonl",
        ),
    ]

    events, limitations = await step._resolve_historical_events(
        references,
        set(),
        datetime.fromisoformat("2026-07-24T15:00:00"),
    )

    assert len(events) == 1
    assert events[0].source_path == "daily/2026-04-26/auto_fin_news_data.jsonl"
    assert not limitations


@pytest.mark.asyncio
async def test_history_search_skips_one_invalid_reference_and_continues(tmp_path: Path):
    actual_path = tmp_path / "daily" / "2026-04-26" / "auto_fin_news_data.jsonl"
    actual_path.parent.mkdir(parents=True)
    actual_path.write_text(
        json.dumps(
            {
                "news_id": "20260426221449_de86",
                "pub_time": "2026-04-26 22:14:49",
                "title": "有效历史新闻",
                "content": "有效内容",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    step = AutoFinHistorySearchStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai"),
    )
    step.logger = SimpleNamespace(warning=lambda _message: None)
    references = [
        AutoFinHistoricalEventReference(
            reason="有效引用",
            news_id="20260426221449_de86",
            source_path="daily/2026-04-26/auto_fin_news_data.jsonl",
        ),
        AutoFinHistoricalEventReference(
            reason="无效引用",
            news_id="20260427100000_dead",
            source_path="daily/2026-04-27/auto_fin_news_data.jsonl",
        ),
    ]

    events, limitations = await step._resolve_historical_events(
        references,
        set(),
        datetime.fromisoformat("2026-07-24T15:00:00"),
    )

    assert [event.news_id for event in events] == ["20260426221449_de86"]
    assert len(limitations) == 1
    assert "20260427100000_dead" in limitations[0]


def test_daily_cookbook_wires_enabled_auto_fin_steps_and_tushare_skill():
    config = _load_config("daily_cookbook")
    steps = config["jobs"]["auto_fin"]["steps"]

    assert [step["backend"] for step in steps] == [
        "auto_fin_data_step",
        "auto_fin_topic_step",
        "auto_fin_history_step",
        "auto_fin_merge_step",
        "dingtalk_markdown_send_step",
    ]
    expected_cron_jobs = {
        "auto_fin_0930_cron": "30 9 * * *",
        "auto_fin_1145_cron": "45 11 * * *",
        "auto_fin_1800_cron": "0 18 * * *",
    }
    for job_name, cron in expected_cron_jobs.items():
        assert config["jobs"][job_name]["cron"] == cron
        assert config["jobs"][job_name]["steps"] == steps
    assert steps[0]["lookback_days"] == 360
    assert steps[0]["progress_interval"] == 30
    assert steps[-1]["input_mapping"] == {"auto_fin_digest_path": "markdown_path"}
    assert config["components"]["agent_wrapper"]["auto_fin"]["skills"] == ["tushare-data"]
