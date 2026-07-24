"""Focused tests for the simple Auto Fin news-case workflow."""

# pylint: disable=missing-function-docstring,protected-access

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from reme.components import ApplicationContext
from reme.components.agent_wrapper.base_agent_wrapper import BaseAgentWrapper
from reme.components.file_chunker import JsonlFileChunker
from reme.components.runtime_context import RuntimeContext
from reme.config.config_parser import _load_config
from reme.schema import AutoFinDecisionOutput, AutoFinResearchPlan
from reme.steps.cookbook.auto_fin.analysis import AutoFinAnalysisStep
from reme.steps.cookbook.auto_fin.data import AutoFinDataStep

TZ = ZoneInfo("Asia/Shanghai")


class _CaseAgent(BaseAgentWrapper):
    """Return one historical case and one reference decision."""

    async def reply(self, inputs, **kwargs):
        schema = kwargs["output_schema"]
        prompt = str(inputs)
        if schema is AutoFinResearchPlan:
            assert "memory_search" in prompt
            assert "daily/2026-07-23/auto_fin_fund_daily.csv" in prompt
            assert "scan_csv" in prompt
            value = {
                "themes": [
                    {
                        "theme": "油气开采",
                        "direction": "POSITIVE",
                        "news_ids": [],
                        "etf_code": "159018.SZ",
                        "etf_name": "油气ETF",
                        "memory_query": "油价 地缘冲突 油气开采",
                        "historical_cases": [
                            {
                                "trade_date": "2026-07-22",
                                "source_path": "daily/2026-07-22/auto_fin_cases.jsonl",
                                "summary": "油价上涨后油气ETF走强",
                            },
                        ],
                    },
                ],
                "limitations": [],
            }
        elif schema is AutoFinDecisionOutput:
            assert "close_return_d0" in prompt
            assert "daily/2026-07-23/auto_fin_fund_adj.csv" in prompt
            assert "Polars" in prompt
            value = {
                "description": "油价新闻尚未完全反映。",
                "body": "历史案例偏正向，当前涨幅低于历史事件当日表现。",
                "recommendations": [
                    {
                        "theme": "油气开采",
                        "etf_code": "159018.SZ",
                        "etf_name": "油气ETF",
                        "action": "BUY",
                        "price_in": "NO",
                        "confidence": 0.7,
                        "reason": "当前涨幅较小",
                        "historical_evidence": "历史相似案例当日上涨",
                        "invalidation_condition": "油价回落",
                    },
                ],
                "limitations": ["历史案例较少"],
            }
        else:  # pragma: no cover
            raise AssertionError(schema)
        return {"structured_output": schema.model_validate(value)}


@pytest.mark.asyncio
async def test_news_case_pipeline_obeys_cutoff_and_writes_indexable_jsonl(
    tmp_path: Path,
):
    calls: list[tuple[str, dict]] = []

    def provider(endpoint: str, **kwargs):
        calls.append((endpoint, kwargs))
        if endpoint == "major_news":
            return [
                {
                    "title": "昨日收盘前旧消息",
                    "pub_time": "2026-07-23 15:00:00",
                    "src": "财联社",
                    "content": "必须排除",
                },
                {
                    "title": "油价因供应扰动上涨",
                    "pub_time": "2026-07-24 09:30:00",
                    "src": "财联社",
                    "content": "当前窗口新闻",
                },
                {
                    "title": "其他来源消息",
                    "pub_time": "2026-07-24 08:30:00",
                    "src": "新浪财经",
                    "content": "必须按来源排除",
                },
                {
                    "title": "决策后消息",
                    "pub_time": "2026-07-24 09:31:00",
                    "src": "财联社",
                    "content": "必须排除",
                },
            ]
        if endpoint == "fund_daily":
            prices = {"20260721": 1.04, "20260722": 1.06, "20260723": 1.05}
            trade_date = kwargs["trade_date"]
            return [
                {
                    "ts_code": "159018.SZ",
                    "trade_date": trade_date,
                    "close": prices[trade_date],
                    "amount": 130,
                },
            ]
        if endpoint == "fund_adj":
            return [
                {
                    "ts_code": "159018.SZ",
                    "trade_date": kwargs["trade_date"],
                    "adj_factor": 1.0,
                },
            ]
        raise AssertionError(endpoint)

    app_context = ApplicationContext(
        workspace_dir=str(tmp_path),
        timezone="Asia/Shanghai",
    )
    wrapper = _CaseAgent(app_context=app_context)
    context = RuntimeContext(
        date="2026-07-24",
        now="2026-07-24T09:31:00+08:00",
        lookback_days=4,
        trade_dates=["2026-07-21", "2026-07-22", "2026-07-23", "2026-07-24"],
        tushare_provider=provider,
        force=True,
    )
    await AutoFinDataStep(app_context=app_context)(context)
    first_calls = list(calls)

    calls.clear()
    await AutoFinDataStep(app_context=app_context)(context)
    assert not calls

    (tmp_path / "daily" / "2026-07-22" / "auto_fin_fund_adj.csv").unlink()
    await AutoFinDataStep(app_context=app_context)(context)
    assert [(endpoint, kwargs["trade_date"]) for endpoint, kwargs in calls] == [
        ("fund_adj", "20260722"),
    ]

    response = await AutoFinAnalysisStep(
        app_context=app_context,
        agent_wrapper=wrapper,
    )(context)

    day_dir = tmp_path / "daily" / "2026-07-24"
    cached_news = json.loads((day_dir / "auto_fin_news_data.jsonl").read_text().strip())
    news = [json.loads(line) for line in (day_dir / "auto_fin_news.jsonl").read_text().splitlines()]
    cases = [json.loads(line) for line in (day_dir / "auto_fin_cases.jsonl").read_text().splitlines()]
    etf = [json.loads(line) for line in (day_dir / "auto_fin_etf.jsonl").read_text().splitlines()]

    assert [item["title"] for item in news] == ["油价因供应扰动上涨"]
    assert cached_news["pub_time"] == "2026-07-24 09:30:00"
    assert cases[0]["action"] == "BUY"
    assert cases[0]["price_in"] == "NO"
    assert cases[0]["latest"]["pct_change"] == pytest.approx(1.05 / 1.06 - 1.0)
    assert any(item["record_type"] == "auto_fin_latest_etf" for item in etf)
    assert any(item["record_type"] == "auto_fin_etf_daily" for item in etf)
    assert "最近收盘涨跌" in (day_dir / "auto_fin.md").read_text(encoding="utf-8")
    assert response.metadata["news_count"] == 1
    assert response.metadata["case_count"] == 1
    assert context["markdown_path"] == "daily/2026-07-24/auto_fin.md"

    _, chunks = await JsonlFileChunker(app_context=app_context).chunk(
        day_dir / "auto_fin_cases.jsonl",
    )
    assert chunks
    assert json.loads(chunks[0].text)["record_type"] == "auto_fin_case"

    news_call = next(
        kwargs
        for endpoint, kwargs in first_calls
        if endpoint == "major_news" and kwargs["start_date"].startswith("2026-07-24")
    )
    assert news_call["start_date"] == "2026-07-24 00:00:00"
    assert news_call["end_date"] == "2026-07-24 09:30:00"
    assert news_call["src"] == "财联社"
    daily_calls = [kwargs for endpoint, kwargs in first_calls if endpoint == "fund_daily"]
    assert [call["trade_date"] for call in daily_calls] == [
        "20260721",
        "20260722",
        "20260723",
    ]


@pytest.mark.asyncio
async def test_pipeline_rejects_before_0930(tmp_path: Path):
    app_context = ApplicationContext(
        workspace_dir=str(tmp_path),
        timezone="Asia/Shanghai",
    )
    step = AutoFinDataStep(app_context=app_context)

    with pytest.raises(ValueError, match="09:30 decision has not been reached"):
        await step(
            date="2026-07-24",
            now=datetime(2026, 7, 24, 9, 29, tzinfo=TZ),
        )


@pytest.mark.asyncio
async def test_pipeline_rejects_historical_rerun_to_avoid_realtime_leakage(
    tmp_path: Path,
):
    app_context = ApplicationContext(
        workspace_dir=str(tmp_path),
        timezone="Asia/Shanghai",
    )
    step = AutoFinDataStep(app_context=app_context)

    with pytest.raises(ValueError, match="only supports the current trade date"):
        await step(
            date="2026-07-23",
            now="2026-07-24T09:31:00+08:00",
        )


def test_case_statistics_use_only_available_adjusted_closes():
    history = AutoFinAnalysisStep._adjusted_history(
        [
            {"trade_date": "20260718", "close": 1.0},
            {"trade_date": "20260721", "close": 1.1},
            {"trade_date": "20260722", "close": 1.2},
        ],
        [
            {"trade_date": "20260718", "adj_factor": 1.0},
            {"trade_date": "20260721", "adj_factor": 1.0},
            {"trade_date": "20260722", "adj_factor": 1.0},
        ],
    )

    stat = AutoFinAnalysisStep._case_stat(history, datetime(2026, 7, 21).date())

    assert stat["close_return_d0"] == pytest.approx(0.1)
    assert stat["close_return_d1"] == pytest.approx(0.2)
    assert "close_return_d3" not in stat


def test_daily_cookbook_wires_one_0930_case_job_and_jsonl_indexing():
    config = _load_config("daily_cookbook")

    assert "auto_fin_0930_cron" in config["jobs"]
    assert "auto_fin_0900_cron" not in config["jobs"]
    assert "auto_fin_1145_cron" not in config["jobs"]
    assert "auto_fin_1445_cron" not in config["jobs"]
    assert config["jobs"]["auto_fin_0930_cron"]["cron"] == "30 9 * * 1-5"
    assert config["jobs"]["auto_fin"]["lookback_days"] == 60
    assert config["jobs"]["auto_fin"]["parameters"]["properties"]["lookback_days"]["default"] == 60
    steps = config["jobs"]["auto_fin"]["steps"]
    assert len(steps) == 3
    assert steps[0] == {"backend": "auto_fin_data_step", "outbound_proxy": "default"}
    assert steps[1]["backend"] == "auto_fin_analysis_step"
    assert steps[1]["agent_wrapper"] == "auto_fin"
    assert steps[2]["backend"] == "dingtalk_markdown_send_step"
    assert steps[2]["title"] == "ReMe Auto Fin"
    assert config["jobs"]["auto_fin_0930_cron"]["lookback_days"] == 60
    assert config["jobs"]["auto_fin_0930_cron"]["steps"] == steps
    assert config["jobs"]["index_update_loop"]["watch_suffixes"] == ["md", "jsonl"]
    assert config["jobs"]["reindex"]["watch_suffixes"] == ["md", "jsonl"]
    assert config["components"]["file_chunker"]["jsonl"]["backend"] == "jsonl"
