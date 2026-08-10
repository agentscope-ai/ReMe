"""Focused tests for the topic-news Auto Fin workflow."""

# pylint: disable=missing-function-docstring,protected-access

import json
from datetime import date, datetime
from pathlib import Path

import pytest

from reme.components import ApplicationContext
from reme.components.agent_wrapper.base_agent_wrapper import BaseAgentWrapper
from reme.components.runtime_context import RuntimeContext
from reme.schema import AutoFinReportOutput
from reme.steps.cookbook.auto_fin._base import _plain_text, _write
from reme.steps.cookbook.auto_fin.data import AutoFinDataStep
from reme.steps.cookbook.auto_fin.merge import AutoFinMergeStep


def _cls_row(news_id: int, timestamp: int, title: str, content: str) -> dict:
    return {"id": news_id, "ctime": timestamp, "title": title, "content": content}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_atomic_write_preserves_existing_file_on_failure(tmp_path: Path, monkeypatch):
    path = tmp_path / "result.json"
    path.write_text("existing", encoding="utf-8")

    monkeypatch.setattr(
        "reme.steps.cookbook.auto_fin._base.os.replace",
        lambda *_args: (_ for _ in ()).throw(OSError()),
    )
    with pytest.raises(OSError):
        _write(path, "replacement")
    assert path.read_text(encoding="utf-8") == "existing"
    assert not list(tmp_path.glob(".*.tmp"))


def test_news_markdown_round_trip_and_topics(tmp_path: Path):
    rows = [
        {
            "news_id": "2448247",
            "event_time": "2026-08-10T07:00:00",
            "title": "黄金上涨",
            "content": "避险需求增强",
        },
    ]
    path = tmp_path / "news.md"
    path.write_text(AutoFinDataStep._render_news(date(2026, 8, 10), rows), encoding="utf-8")

    assert AutoFinDataStep.read_news(path) == rows
    assert AutoFinDataStep._topics("黄金，机器人, 黄金") == ["黄金", "机器人"]
    assert _plain_text("<p>甲&amp;乙</p><style>隐藏</style><p>丙</p>") == "甲&乙 丙"


@pytest.mark.asyncio
async def test_data_step_reads_free_local_cls_jsonl(tmp_path: Path):
    source = tmp_path / "cls.jsonl"
    timestamp = int(datetime(2026, 8, 10, 9).timestamp())
    _write_jsonl(source, [_cls_row(2448247, timestamp, "黄金上涨", "避险需求增强")])
    context = RuntimeContext(
        date="2026-08-10",
        now="2026-08-10T09:30:00+08:00",
        news_file=str(source),
        news_lookback_days=1,
        topics="黄金,机器人",
    )

    response = await AutoFinDataStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai"),
    )(context)

    news_path = tmp_path / "daily" / "2026-08-10" / "auto_fin_news.md"
    assert "黄金上涨" in news_path.read_text(encoding="utf-8")
    assert context["auto_fin_topics"] == ["黄金", "机器人"]
    assert context["auto_fin_news_path"] == "daily/2026-08-10/auto_fin_news.md"
    assert response.metadata["topics"] == ["黄金", "机器人"]


@pytest.mark.asyncio
async def test_data_step_explains_how_to_create_missing_source(tmp_path: Path):
    context = RuntimeContext(
        date="2026-08-10",
        now="2026-08-10T09:30:00+08:00",
        news_file=str(tmp_path / "missing.jsonl"),
    )
    step = AutoFinDataStep(app_context=ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai"))

    with pytest.raises(FileNotFoundError, match="Provide a local CLS JSONL file"):
        await step(context)


class _ResearchAgent(BaseAgentWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    async def reply(self, inputs, **kwargs):
        self.calls.append((str(inputs), kwargs))
        schema = kwargs["output_schema"]
        return {
            "structured_output": schema.model_validate(
                {
                    "title": "# 主题新闻观察",
                    "description": "关注黄金政策变化。",
                    "body": (
                        "## 今日判断\n\n"
                        "本次事件与 [[daily/2026-08-01/auto_fin.md|历史黄金观察]] 背景相似。\n\n"
                        "无效引用 [[daily/missing.md|缺失文章]] 和 [[../../outside.md|越界文章]] 应降级。"
                    ),
                },
            ),
        }


@pytest.mark.asyncio
async def test_research_agent_gets_search_tools_and_code_builds_valid_wikilinks(tmp_path: Path):
    current = tmp_path / "daily" / "2026-08-10" / "auto_fin_news.md"
    current.parent.mkdir(parents=True)
    current.write_text(
        AutoFinDataStep._render_news(
            date(2026, 8, 10),
            [{"news_id": "1", "event_time": "2026-08-10T09:00:00", "title": "黄金", "content": "上涨"}],
        ),
        encoding="utf-8",
    )
    historical = tmp_path / "daily" / "2026-08-01" / "auto_fin.md"
    historical.parent.mkdir(parents=True)
    historical.write_text("# 历史黄金观察\n", encoding="utf-8")
    agent = _ResearchAgent(app_context=ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai"))
    context = RuntimeContext(
        auto_fin_date="2026-08-10",
        auto_fin_decision_at="2026-08-10T09:30:00",
        auto_fin_topics=["黄金"],
        auto_fin_news_path="daily/2026-08-10/auto_fin_news.md",
    )

    response = await AutoFinMergeStep(app_context=agent.app_context, agent_wrapper=agent)(context)

    prompt, kwargs = agent.calls[0]
    assert "黄金" in prompt
    assert kwargs == {"output_schema": AutoFinReportOutput, "job_tools": ["memory_search", "read"]}
    report = (tmp_path / "daily" / "2026-08-10" / "auto_fin.md").read_text(encoding="utf-8")
    assert "[[daily/2026-08-10/auto_fin_news.md|auto fin news]]" in report
    assert "[[daily/2026-08-01/auto_fin.md|历史黄金观察]]" in report
    assert "缺失文章" in report and "越界文章" in report
    assert "missing.md" not in report and "outside.md" not in report
    assert response.metadata["source_paths"] == [
        "daily/2026-08-10/auto_fin_news.md",
        "daily/2026-08-01/auto_fin.md",
    ]


def test_config_uses_local_news_topics_and_two_read_only_agent_tools():
    from reme.config.config_parser import _load_config

    config = _load_config("daily_cookbook")
    job = config["jobs"]["auto_fin"]
    assert "etf_codes" not in job
    assert job["news_lookback_days"] == 7
    assert job["parameters"]["properties"]["topics"]["default"] == ""
    assert [step["backend"] for step in job["steps"]] == [
        "auto_fin_data_step",
        "update_index_step",
        "auto_fin_merge_step",
        "dingtalk_markdown_send_step",
    ]
    for name, schedule in {
        "auto_fin_0930_cron": "30 9 * * *",
        "auto_fin_1130_cron": "30 11 * * *",
        "auto_fin_1800_cron": "0 18 * * *",
    }.items():
        assert config["jobs"][name]["cron"] == schedule
        assert config["jobs"][name]["steps"] == job["steps"]


def test_report_schema_has_only_required_markdown_fields():
    schema = AutoFinReportOutput.model_json_schema()

    assert schema["required"] == ["title", "description", "body"]
    assert set(schema["properties"]) == {"title", "description", "body"}
