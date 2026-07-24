"""Focused tests for the four-step Auto Fin workflow."""

# pylint: disable=missing-function-docstring,protected-access

import json
from pathlib import Path

import pytest
import yaml

from reme.components import ApplicationContext
from reme.components.agent_wrapper.base_agent_wrapper import BaseAgentWrapper
from reme.components.runtime_context import RuntimeContext
from reme.config.config_parser import _load_config
from reme.schema import AutoFinHistoricalResearch, AutoFinReportOutput, AutoFinTopicAnalysis, AutoFinTopicsOutput
from reme.steps.cookbook.auto_fin.data import AutoFinDataStep
from reme.steps.cookbook.auto_fin.history import AutoFinHistoryStep
from reme.steps.cookbook.auto_fin.merge import AutoFinMergeStep
from reme.steps.cookbook.auto_fin.topic import AutoFinTopicStep


@pytest.mark.asyncio
async def test_read_jsonl_preserves_unicode_line_separator(tmp_path: Path):
    path = tmp_path / "news.jsonl"
    rows = [{"title": "包含\u2028行分隔符"}, {"title": "下一条"}]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    assert await AutoFinDataStep._read_jsonl(path) == rows


@pytest.mark.asyncio
async def test_data_fills_missing_news_refreshes_today_and_force_refreshes_history(tmp_path: Path):
    calls = []

    def provider(endpoint: str, **kwargs):
        calls.append((endpoint, kwargs))
        if endpoint == "major_news":
            day = kwargs["start_date"][:10]
            return [{"title": day, "pub_time": f"{day} 08:00:00", "src": "财联社", "content": day}]
        raise AssertionError(endpoint)

    app_context = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai")
    context = RuntimeContext(
        date="2026-07-24",
        now="2026-07-24T09:30:00+08:00",
        lookback_days=3,
        trade_dates=["2026-07-23"],
        tushare_provider=provider,
    )

    await AutoFinDataStep(app_context=app_context)(context)
    assert [endpoint for endpoint, _ in calls] == ["major_news"] * 3

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


class _Agent(BaseAgentWrapper):
    """Return deterministic structured replies for every analysis stage."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []
        self.temporary_dirs = []

    async def reply(self, inputs, **kwargs):
        schema = kwargs["output_schema"]
        prompt = str(inputs)
        self.calls.append((schema, prompt, kwargs))
        assert "resume" not in kwargs
        assert "session_id" not in kwargs
        if schema is AutoFinTopicsOutput:
            assert "不搜索历史" in prompt
            value = {
                "topics": {
                    "原油供应扰动": [
                        {"event_time": "2026-07-23T16:00:00+08:00", "event_content": "主要产油区供应中断"},
                        {"event_time": "2026-07-24T09:00:00+08:00", "event_content": "供应恢复时间仍不确定"},
                    ],
                },
            }
        elif schema is AutoFinHistoricalResearch:
            assert "memory_search" in prompt
            assert "不查询行情" in prompt
            value = {
                "topic": "原油供应扰动",
                "historical_events": [
                    {
                        "event_time": "2026-06-01T10:00:00+08:00",
                        "event_content": "历史供应中断",
                        "source_path": "daily/2026-06-01/auto_fin_news_data.jsonl",
                    },
                ],
                "limitations": [],
            }
        elif schema is AutoFinTopicAnalysis:
            assert "不再搜索" in prompt
            assert "$tushare-data" in prompt
            assert "D1-D5" in prompt
            temporary_dir = next(
                line.split("：", 1)[1] for line in prompt.splitlines() if line.startswith("临时目录：")
            )
            assert Path(temporary_dir).is_dir()
            self.temporary_dirs.append(temporary_dir)
            value = {
                "topic": "原油供应扰动",
                "historical_events": [
                    {
                        "event_time": "2026-06-01T10:00:00+08:00",
                        "event_content": "历史供应中断",
                        "source_path": "daily/2026-06-01/auto_fin_news_data.jsonl",
                    },
                ],
                "etfs": [
                    {
                        "etf_code": "159018.SZ",
                        "etf_name": "油气ETF",
                        "asset_type": "境内商品产业链股票 ETF",
                        "market": "SZSE",
                        "relationship": "上游油气企业盈利受油价影响",
                        "current_intraday_returns": [],
                        "historical_samples": [
                            {
                                "event_time": "2026-06-01T10:00:00+08:00",
                                "baseline_time": "2026-06-01T09:59:00+08:00",
                                "baseline_price": 1.0,
                                "intraday_returns": [
                                    {"bar_time": "2026-06-01T10:15:00+08:00", "return_from_baseline": 0.006},
                                ],
                                "reaction_summary": "事件后走高，尾盘略有回落",
                                "returns": {
                                    "d1_return": 0.01,
                                    "d1_d2_return": 0.02,
                                    "d1_d3_return": 0.03,
                                    "d1_d4_return": 0.025,
                                    "d1_d5_return": 0.015,
                                },
                            },
                        ],
                        "forecast": {
                            "anchor_event_time": "2026-07-24T09:00:00+08:00",
                            "baseline_time": "2026-07-23T15:00:00+08:00",
                            "baseline_price": 1.1,
                            "returns": {
                                "d1_return": 0.008,
                                "d1_d2_return": 0.016,
                                "d1_d3_return": 0.02,
                                "d1_d4_return": 0.015,
                                "d1_d5_return": 0.01,
                            },
                            "suggested_holding_period": "D1-D3",
                            "confidence": 0.65,
                            "reason": "历史反应通常在第三个收盘附近达到高点",
                            "exit_condition": "累计收益达到历史样本上沿",
                            "invalidation_condition": "供应快速恢复",
                        },
                    },
                ],
                "summary": "供应扰动短期利好油气产业链。",
                "limitations": ["历史样本较少"],
            }
        elif schema is AutoFinReportOutput:
            assert "不重新搜索新闻" in prompt
            assert "D1-D3" in prompt
            value = {
                "title": "Auto Fin ETF 事件分析",
                "description": "原油供应扰动的历史行情与当前预估。",
                "body": "## 原油供应扰动\n\n油气ETF参考持有时间为 D1-D3。",
                "limitations": ["历史样本较少"],
            }
        else:  # pragma: no cover
            raise AssertionError(schema)
        return {"structured_output": schema.model_validate(value)}


@pytest.mark.asyncio
async def test_four_step_pipeline_writes_structured_frontmatter_and_cleans_temporary_data(tmp_path: Path):
    def provider(endpoint: str, **_kwargs):
        if endpoint != "major_news":
            raise AssertionError(endpoint)
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

    app_context = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai")
    agent = _Agent(app_context=app_context)
    context = RuntimeContext(
        date="2026-07-24",
        now="2026-07-24T09:30:00+08:00",
        lookback_days=2,
        trade_dates=["2026-07-23"],
        tushare_provider=provider,
    )

    await AutoFinDataStep(app_context=app_context)(context)
    await AutoFinTopicStep(app_context=app_context, agent_wrapper=agent)(context)
    await AutoFinHistoryStep(app_context=app_context, agent_wrapper=agent)(context)
    response = await AutoFinMergeStep(app_context=app_context, agent_wrapper=agent)(context)

    assert [schema for schema, _, _ in agent.calls] == [
        AutoFinTopicsOutput,
        AutoFinHistoricalResearch,
        AutoFinTopicAnalysis,
        AutoFinReportOutput,
    ]
    assert all(not Path(path).exists() for path in agent.temporary_dirs)
    report = (tmp_path / "daily" / "2026-07-24" / "auto_fin.md").read_text(encoding="utf-8")
    metadata = yaml.safe_load(report.split("---", 2)[1])
    assert metadata["schema_version"] == "auto-fin/v2"
    assert metadata["topics"]["原油供应扰动"][0]["event_content"] == "主要产油区供应中断"
    assert metadata["analyses"][0]["etfs"][0]["forecast"]["suggested_holding_period"] == "D1-D3"
    assert response.metadata["topic_count"] == 1
    assert response.metadata["etf_count"] == 1
    assert context["markdown_path"] == "daily/2026-07-24/auto_fin.md"


def test_daily_cookbook_wires_four_auto_fin_steps_and_tushare_skill():
    config = _load_config("daily_cookbook")
    steps = config["jobs"]["auto_fin"]["steps"]

    assert [step["backend"] for step in steps] == [
        "auto_fin_data_step",
        "auto_fin_topic_step",
        "auto_fin_history_step",
        "auto_fin_merge_step",
        "dingtalk_markdown_send_step",
    ]
    assert config["jobs"]["auto_fin_0930_cron"]["steps"] == steps
    assert config["components"]["agent_wrapper"]["auto_fin"]["skills"] == ["tushare-data"]
