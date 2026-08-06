"""Focused tests for the tool-free Auto Fin workflow."""

# pylint: disable=missing-function-docstring,protected-access

import hashlib
from datetime import date, datetime
from pathlib import Path

import pytest

from reme.components import ApplicationContext
from reme.components.agent_wrapper.base_agent_wrapper import BaseAgentWrapper
from reme.components.runtime_context import RuntimeContext
from reme.schema import (
    AutoFinEtfsOutput,
    AutoFinHistoricalOutput,
    AutoFinReportOutput,
    Response,
)
from reme.steps.cookbook.auto_fin._base import _plain_text, _write
from reme.steps.cookbook.auto_fin.data import AutoFinDataStep
from reme.steps.cookbook.auto_fin.history import AutoFinHistoryStep
from reme.steps.cookbook.auto_fin.merge import AutoFinMergeStep
from reme.steps.cookbook.auto_fin.topic import AutoFinTopicStep


def test_atomic_write_preserves_existing_file_on_failure(tmp_path: Path, monkeypatch):
    path = tmp_path / "result.json"
    path.write_text("existing", encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("replace failed")

    monkeypatch.setattr("reme.steps.cookbook.auto_fin._base.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        _write(path, "replacement")
    assert path.read_text(encoding="utf-8") == "existing"
    assert not list(tmp_path.glob(".*.tmp"))


def test_news_markdown_round_trip_and_plain_text(tmp_path: Path):
    rows = [
        {
            "news_id": "20260724070000_abcd",
            "event_time": "2026-07-24T07:00:00",
            "title": "黄金上涨",
            "content": "避险需求增强",
        },
    ]
    path = tmp_path / "news.md"
    path.write_text(AutoFinDataStep._render_news(date(2026, 7, 24), rows), encoding="utf-8")

    assert AutoFinDataStep.read_news(path) == rows
    assert _plain_text("<p>甲&amp;乙</p><style>隐藏</style><p>丙</p>") == "甲&乙 丙"


@pytest.mark.asyncio
async def test_non_trade_day_stops_after_trade_calendar(tmp_path: Path):
    calls = []

    def provider(endpoint, **kwargs):
        calls.append((endpoint, kwargs))
        return [{"cal_date": "20260725", "is_open": 0}]

    context = RuntimeContext(date="2026-07-25", now="2026-07-25T09:00:00+08:00", tushare_provider=provider)
    response = await AutoFinDataStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai"),
    )(context)

    assert [call[0] for call in calls] == ["trade_cal"]
    assert context["auto_fin_skipped"] is True
    assert response.metadata["skipped"] is True


def test_adjusted_returns_use_close_before_1500_and_next_open_after_1500():
    rows = [
        {"trade_date": "20260601", "open": 9, "close": 10, "adj_factor": 1},
        {"trade_date": "20260602", "open": 10, "close": 11, "adj_factor": 1},
        {"trade_date": "20260603", "open": 11, "close": 12, "adj_factor": 1},
        {"trade_date": "20260604", "open": 12, "close": 13, "adj_factor": 1},
        {"trade_date": "20260605", "open": 13, "close": 14, "adj_factor": 1},
        {"trade_date": "20260608", "open": 14, "close": 15, "adj_factor": 1},
    ]

    before = AutoFinHistoryStep._returns(datetime(2026, 6, 1, 14), rows)
    after = AutoFinHistoryStep._returns(datetime(2026, 6, 1, 16), rows)

    assert before == pytest.approx({"d1": 0.1, "d2": 0.2, "d3": 0.3, "d5": 0.5})
    assert after == pytest.approx({"d1": 0.1, "d2": 0.2, "d3": 0.3, "d5": 0.5})


class _SearchJob:
    def __init__(self, path: str, news_id: str):
        self.path = path
        self.news_id = news_id
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return Response(
            metadata={
                "results": [
                    {
                        "path": self.path,
                        "text": f"- news_id: `{self.news_id}`\n历史降息新闻",
                    },
                ],
            },
        )


class _Agent(BaseAgentWrapper):
    def __init__(self, current_id: str, historical_id: str, **kwargs):
        super().__init__(**kwargs)
        self.current_id = current_id
        self.historical_id = historical_id
        self.schemas = []

    async def reply(self, inputs, **kwargs):
        assert kwargs.keys() == {"output_schema"}
        schema = kwargs["output_schema"]
        self.schemas.append(schema)
        prompt = str(inputs)
        assert "不得搜索、调用工具" in prompt
        assert "```json" not in prompt
        if schema is AutoFinEtfsOutput:
            value = {
                "etfs": [
                    {
                        "etf_code": "518880.SH",
                        "etf_name": "错误名称",
                        "events": [{"news_id": self.current_id, "reason": "降息预期利好黄金"}],
                    },
                ],
            }
        elif schema is AutoFinHistoricalOutput:
            value = {
                "historical_events": [
                    {
                        "news_id": self.historical_id,
                        "reason": "同属利率政策变化且对黄金影响相同",
                        "direction": "same",
                    },
                    {
                        "news_id": "20260101000000_ffff",
                        "reason": "模型虚构",
                        "direction": "same",
                    },
                ],
            }
        elif schema is AutoFinReportOutput:
            assert '"d5": 0.5' in prompt
            assert "上一份建议" in prompt
            value = {
                "title": "# 黄金观察",
                "description": "降息事件偏利好黄金。",
                "body": "## 建议\n\n关注黄金ETF。",
            }
        else:  # pragma: no cover
            raise AssertionError(schema)
        return {"structured_output": schema.model_validate(value)}


@pytest.mark.asyncio
async def test_new_pipeline_prepares_context_and_uses_three_tool_free_agents(
    tmp_path: Path,
):
    current_content = "美联储释放降息信号"
    historical_content = "美联储宣布降息"
    current_id = f"20260724090000_{hashlib.sha256(f'财联社{current_content}'.encode()).hexdigest()[:4]}"
    historical_id = f"20260601100000_{hashlib.sha256(f'财联社{historical_content}'.encode()).hexdigest()[:4]}"
    market_dates = [
        "20260601",
        "20260602",
        "20260603",
        "20260604",
        "20260605",
        "20260608",
    ]

    def provider(endpoint, **kwargs):
        if endpoint == "trade_cal":
            return [{"cal_date": "20260724", "is_open": 1}]
        if endpoint == "major_news":
            day = kwargs["start_date"][:10]
            if day == "2026-07-24":
                return [
                    {
                        "title": "降息信号",
                        "pub_time": "2026-07-24 09:00:00",
                        "src": "财联社",
                        "content": current_content,
                    },
                ]
            return []
        if endpoint == "etf_basic":
            return [{"ts_code": "518880.SH", "csname": "黄金ETF", "list_status": "L"}]
        if endpoint == "fund_daily":
            return [
                {
                    "ts_code": "518880.SH",
                    "trade_date": trade_date,
                    "open": 9 + index,
                    "close": 10 + index,
                    "pct_chg": 1,
                }
                for index, trade_date in enumerate(market_dates)
            ]
        if endpoint == "fund_adj":
            return [{"trade_date": trade_date, "adj_factor": 1} for trade_date in market_dates]
        raise AssertionError(endpoint)

    historical_path = tmp_path / "daily" / "2026-06-01" / "auto_fin_news.md"
    historical_path.parent.mkdir(parents=True)
    historical_path.write_text(
        AutoFinDataStep._render_news(
            date(2026, 6, 1),
            [
                {
                    "news_id": historical_id,
                    "event_time": "2026-06-01T10:00:00",
                    "title": "历史降息",
                    "content": historical_content,
                },
            ],
        ),
        encoding="utf-8",
    )
    previous = tmp_path / "daily" / "2026-07-23" / "auto_fin.md"
    previous.parent.mkdir(parents=True)
    previous.write_text("# 上一份建议\n", encoding="utf-8")

    app_context = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai")
    search = _SearchJob("daily/2026-06-01/auto_fin_news.md", historical_id)
    app_context.jobs["memory_search"] = search
    agent = _Agent(current_id, historical_id, app_context=app_context)
    context = RuntimeContext(
        date="2026-07-24",
        now="2026-07-24T09:30:00+08:00",
        tushare_provider=provider,
        etf_codes=["518880.SH"],
        news_lookback_days=1,
        historical_search_limit=7,
    )

    await AutoFinDataStep(app_context=app_context)(context)
    await AutoFinTopicStep(app_context=app_context, agent_wrapper=agent)(context)
    await AutoFinHistoryStep(app_context=app_context, agent_wrapper=agent)(context)
    response = await AutoFinMergeStep(app_context=app_context, agent_wrapper=agent)(context)

    assert agent.schemas == [
        AutoFinEtfsOutput,
        AutoFinHistoricalOutput,
        AutoFinReportOutput,
    ]
    assert search.calls[0]["end_date"] == "2026-07-23"
    assert search.calls[0]["limit"] == 7
    assert context["auto_fin_etfs"][0]["etf_name"] == "黄金ETF"
    historical = context["auto_fin_analyses"][0]["events"][0]["historical_events"]
    assert [event["news_id"] for event in historical] == [historical_id]
    assert historical[0]["returns"]["d5"] == pytest.approx(0.5)
    assert response.metadata["digest_path"] == "daily/2026-07-24/auto_fin.md"
    report = (tmp_path / "daily" / "2026-07-24" / "auto_fin.md").read_text(encoding="utf-8")
    assert report.startswith("# 黄金观察\n\n> 降息事件偏利好黄金。")


def test_config_has_fixed_codes_and_no_agent_tools():
    from reme.config.config_parser import _load_config

    config = _load_config("daily_cookbook")
    job = config["jobs"]["auto_fin"]
    assert job["etf_codes"] == [
        "518880.SH",
        "159530.SZ",
        "512760.SH",
    ]
    search_limit = job["parameters"]["properties"]["historical_search_limit"]
    assert job["parameters"]["properties"]["now"]["default"] == ""
    assert search_limit["default"] == 10
    assert search_limit["minimum"] == 1
    assert job["historical_search_limit"] == 10
    assert job["steps"][3]["historical_search_limit"] == 10
    assert config["jobs"]["auto_fin_1200_cron"]["cron"] == "0 12 * * *"
    assert config["jobs"]["auto_fin_1200_cron"]["steps"] == job["steps"]
    assert not {"auto_fin_0930_cron", "auto_fin_1145_cron", "auto_fin_1800_cron"} & config["jobs"].keys()
    wrapper = config["components"]["agent_wrapper"]["default"]
    assert wrapper["backend"] == "agentscope"
    assert wrapper["builtin_tools"] is False
    assert "skills" not in wrapper and "job_tools" not in wrapper
    assert "agent_wrapper" not in job["steps"][2]
