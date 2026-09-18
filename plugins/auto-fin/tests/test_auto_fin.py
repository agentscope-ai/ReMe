"""Focused tests for the rolling CLS Auto Fin workflow."""

# pylint: disable=missing-function-docstring,protected-access

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

from reme_auto_fin.base import _plain_text, _write
from reme_auto_fin.data import AutoFinDataStep
from reme_auto_fin.merge import AutoFinMergeStep
from reme_auto_fin.schema import AutoFinReportOutput, AutoFinTopicOutput
from reme_auto_fin.topic import AutoFinTopicStep
from reme.components import ApplicationContext
from reme.components.agent_wrapper.base_agent_wrapper import BaseAgentWrapper
from reme.components.runtime_context import RuntimeContext

SHANGHAI = ZoneInfo("Asia/Shanghai")
PLUGIN_MANIFEST = yaml.safe_load(
    (Path(__file__).parents[1] / "src" / "reme_auto_fin" / "plugin.yaml").read_text(encoding="utf-8"),
)


def _row(news_id: int, value: datetime, title: str = "新闻", content: str = "正文") -> dict:
    return {
        "id": news_id,
        "ctime": int(value.timestamp()),
        "title": title,
        "content": content,
    }


def test_atomic_write_preserves_existing_file_on_failure(tmp_path: Path, monkeypatch):
    path = tmp_path / "result.md"
    path.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(
        "reme_auto_fin.base.os.replace",
        lambda *_args: (_ for _ in ()).throw(OSError()),
    )

    with pytest.raises(OSError):
        _write(path, "replacement")

    assert path.read_text(encoding="utf-8") == "existing"
    assert not list(tmp_path.glob(".*.tmp"))
    assert _plain_text("<p>甲&amp;乙</p><style>隐藏</style><p>丙</p>") == "甲&乙 丙"


@pytest.mark.asyncio
async def test_data_step_fetches_exact_24_hours_with_default_topics(tmp_path: Path, monkeypatch):
    end = datetime(2026, 8, 10, 9, 30, tzinfo=SHANGHAI)

    async def page(_self, _client, _last_time):
        return [
            _row(1, end, "黄金上涨"),
            _row(2, end.replace(day=9), "窗口边界"),
            _row(3, end.replace(day=9, minute=29), "窗口之外"),
            _row(1, end, "重复"),
        ]

    monkeypatch.setattr(AutoFinDataStep, "_request_page", page)
    context = RuntimeContext(date="2026-08-10", now=end.isoformat(), topics="")
    response = await AutoFinDataStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai"),
        request_interval=0,
    )(context)

    assert [row["news_id"] for row in context["auto_fin_news"]] == ["2", "1"]
    assert context["auto_fin_topics"] == ["黄金", "机器人", "半导体"]
    assert context["auto_fin_window_start"] == "2026-08-09T09:30:00+08:00"
    assert response.metadata["fetched_news_count"] == 2
    assert not list(tmp_path.rglob("*.md"))


@pytest.mark.asyncio
async def test_data_step_uses_configurable_window_hours(tmp_path: Path, monkeypatch):
    end = datetime(2026, 8, 10, 9, 30, tzinfo=SHANGHAI)

    async def page(_self, _client, _last_time):
        return [
            _row(1, end, "窗口内"),
            _row(2, end.replace(day=9, hour=21, minute=30), "窗口边界"),
            _row(3, end.replace(day=9, hour=21, minute=29), "窗口之外"),
        ]

    monkeypatch.setattr(AutoFinDataStep, "_request_page", page)
    context = RuntimeContext(date="2026-08-10", now=end.isoformat(), topics="黄金", window_hours=12)
    await AutoFinDataStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai"),
        request_interval=0,
    )(context)

    assert [row["news_id"] for row in context["auto_fin_news"]] == ["2", "1"]
    assert context["auto_fin_window_start"] == "2026-08-09T21:30:00+08:00"
    assert context["auto_fin_window_hours"] == 12


class _TopicAgent(BaseAgentWrapper):
    def __init__(self, selected: dict[str, list[str]], **kwargs):
        super().__init__(**kwargs)
        self.selected = selected
        self.calls = []

    async def reply(self, inputs, **kwargs):
        self.calls.append((str(inputs), kwargs))
        return {"result": f"筛选结果：\n```json\n{json.dumps(self.selected)}\n```\n以上是相关 ID。"}


@pytest.mark.asyncio
async def test_topic_step_keeps_real_ids_in_memory_only(tmp_path: Path):
    app_context = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai")
    agent = _TopicAgent({"黄金": ["2", "missing", "2"]}, app_context=app_context)
    context = RuntimeContext(
        auto_fin_news=[
            {
                "news_id": "1",
                "event_time": "2026-08-10T08:00:00+08:00",
                "title": "甲",
                "content": "甲",
            },
            {
                "news_id": "2",
                "event_time": "2026-08-10T09:00:00+08:00",
                "title": "乙",
                "content": "乙",
            },
        ],
        auto_fin_topics=["黄金"],
    )

    response = await AutoFinTopicStep(app_context=app_context, agent_wrapper=agent)(context)

    assert [row["news_id"] for row in context["auto_fin_selected_news"]] == ["2"]
    assert [row["news_id"] for row in context["auto_fin_news_by_topic"]["黄金"]] == ["2"]
    assert agent.calls[0][1] == {}
    assert '```json\n{"黄金": []}' in agent.calls[0][0]
    assert agent.calls[0][0].index("## 输入材料") < agent.calls[0][0].index("## 任务指令")
    assert response.metadata["relevant_news_count"] == 1
    assert not list(tmp_path.rglob("*.*"))


@pytest.mark.asyncio
async def test_topic_step_marks_empty_selection_as_successful_skip(tmp_path: Path):
    app_context = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai")
    agent = _TopicAgent({"黄金": []}, app_context=app_context)
    context = RuntimeContext(
        auto_fin_news=[
            {
                "news_id": "1",
                "event_time": "2026-08-10T08:00:00+08:00",
                "title": "甲",
                "content": "甲",
            },
        ],
        auto_fin_topics=["黄金"],
        auto_fin_window_hours=12,
    )

    response = await AutoFinTopicStep(
        app_context=app_context,
        agent_wrapper=agent,
    )(context)

    assert context["auto_fin_skipped"] is True
    assert response.metadata["skipped"] is True
    assert response.answer == "最近12小时没有与 黄金 相关的财联社新闻。"
    assert "最近 12 小时" in agent.calls[0][0]
    assert not list(tmp_path.rglob("*.md"))


@pytest.mark.asyncio
async def test_topic_step_retries_invalid_json_once(tmp_path: Path):
    class RetryAgent(_TopicAgent):
        """Return one malformed response before a fenced topic mapping."""

        async def reply(self, inputs, **kwargs):
            self.calls.append((str(inputs), kwargs))
            return {"result": ('{"new_ids": "[\\"1\\"]"}' if len(self.calls) == 1 else '```json\n{"黄金": ["1"]}\n```')}

    app_context = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai")
    agent = RetryAgent({"黄金": []}, app_context=app_context)
    context = RuntimeContext(
        auto_fin_news=[
            {
                "news_id": "1",
                "event_time": "2026-08-10T08:00:00+08:00",
                "title": "甲",
                "content": "甲",
            },
        ],
        auto_fin_topics=["黄金"],
    )

    await AutoFinTopicStep(app_context=app_context, agent_wrapper=agent)(context)

    assert len(agent.calls) == 2
    assert [row["news_id"] for row in context["auto_fin_selected_news"]] == ["1"]


@pytest.mark.parametrize(
    "value",
    ['{"new_ids": ["1"]}', "```json\n[1]\n```", '```json\n{"黄金": [1]}\n```', "not json", ""],
)
def test_topic_step_rejects_non_array_or_non_string_ids(value: str):
    with pytest.raises(ValueError):
        AutoFinTopicStep._parse_news_ids(value, ["黄金"])


@pytest.mark.asyncio
async def test_topic_step_batches_by_prompt_length_and_merges_topics(tmp_path: Path):
    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    agent = _TopicAgent({"黄金": ["1", "2", "2"], "机器人": ["2"]}, app_context=app_context)
    news = [
        {
            "news_id": str(index),
            "event_time": f"2026-08-10T0{index}:00:00+08:00",
            "title": "新闻",
            "content": "正文" * 80,
        }
        for index in (1, 2, 3)
    ]
    step = AutoFinTopicStep(app_context=app_context, agent_wrapper=agent)
    one_item_length = len(step._prompt([{**news[0], "content": news[0]["content"][:1000]}], ["黄金", "机器人"], "24"))
    step.PROMPT_CHAR_LIMIT = one_item_length + 5
    context = RuntimeContext(auto_fin_news=news, auto_fin_topics=["黄金", "机器人"])

    response = await step(context)

    assert len(agent.calls) == 3
    assert all(len(prompt) <= step.PROMPT_CHAR_LIMIT for prompt, _ in agent.calls)
    assert [row["news_id"] for row in context["auto_fin_news_by_topic"]["黄金"]] == ["1", "2"]
    assert [row["news_id"] for row in context["auto_fin_news_by_topic"]["机器人"]] == ["2"]
    assert [row["news_id"] for row in context["auto_fin_selected_news"]] == ["1", "2"]
    assert response.metadata["topic_batch_count"] == 3


class _ResearchAgent(BaseAgentWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    async def reply(self, inputs, **kwargs):
        self.calls.append((str(inputs), kwargs))
        return {
            "structured_output": AutoFinReportOutput(
                title="# 主题新闻观察",
                description="关注黄金政策变化。",
                body=(
                    "## 今日判断\n\n"
                    "CLS 1（09:00，黄金上涨）与 "
                    "[[daily/2026-08-01/auto_fin.md|历史黄金观察]]"
                    "(daily/2026-08-01/auto_fin.md) 背景相似。\n\n"
                    "无效引用 [[daily/missing.md|缺失文章]] 和 [[../../outside.md|越界文章]] 应降级。"
                ),
            ),
        }


@pytest.mark.asyncio
async def test_merge_writes_only_final_report_and_validates_historical_links(
    tmp_path: Path,
):
    historical = tmp_path / "daily" / "2026-08-01" / "auto_fin.md"
    historical.parent.mkdir(parents=True)
    historical.write_text("# 历史黄金观察\n", encoding="utf-8")
    app_context = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai")
    agent = _ResearchAgent(app_context=app_context)
    context = RuntimeContext(
        auto_fin_date="2026-08-10",
        auto_fin_decision_at="2026-08-10T09:30:00+08:00",
        auto_fin_window_start="2026-08-09T09:30:00+08:00",
        auto_fin_topics=["黄金"],
        auto_fin_news_by_topic={
            "黄金": [
                {
                    "news_id": "1",
                    "event_time": "2026-08-10T09:00:00+08:00",
                    "title": "黄金上涨",
                    "content": "避险需求增强",
                },
            ],
        },
        auto_fin_selected_news=[
            {
                "news_id": "1",
                "event_time": "2026-08-10T09:00:00+08:00",
                "title": "黄金上涨",
                "content": "避险需求增强",
            },
        ],
    )

    response = await AutoFinMergeStep(
        app_context=app_context,
        agent_wrapper=agent,
        job_tools=["search"],
    )(context)

    prompt, kwargs = agent.calls[0]
    assert "end_date" not in prompt
    assert "`search`" in prompt
    assert "`read`" not in prompt
    assert prompt.index("## 输入材料") < prompt.index("## 任务指令")
    assert kwargs["output_schema"] == AutoFinReportOutput
    assert kwargs["job_tools"] == ["search"]
    assert kwargs["tool_context_id"].startswith("auto_fin:")
    assert kwargs["injected_job_kwargs"] == {
        "limit": 5,
        "min_score": 0.0,
        "start_date": None,
        "end_date": "2026-08-09",
        "max_search_calls": 3,
    }
    report = (tmp_path / "daily" / "2026-08-10" / "auto_fin.md").read_text(encoding="utf-8")
    assert "[[daily/2026-08-01/auto_fin.md|历史黄金观察]]" in report
    assert "](daily/2026-08-01/auto_fin.md)" not in report
    assert "缺失文章" in report and "越界文章" in report
    assert "missing.md" not in report and "outside.md" not in report
    assert not (tmp_path / "daily" / "2026-08-10" / "auto_fin_news.md").exists()
    assert not (tmp_path / "resource").exists()
    assert context["changes"] == [{"change": "added", "path": "daily/2026-08-10/auto_fin.md"}]
    assert response.metadata["source_paths"] == ["daily/2026-08-01/auto_fin.md"]


@pytest.mark.asyncio
async def test_merge_researches_each_topic_with_latest_twenty_news(tmp_path: Path):
    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    agent = _ResearchAgent(app_context=app_context)
    gold = [
        {"news_id": str(index), "event_time": f"2026-08-10T09:{index:02}:00+08:00", "title": "黄金", "content": "正文"}
        for index in range(25)
    ]
    robot = [{"news_id": "30", "event_time": "2026-08-10T08:00:00+08:00", "title": "机器人", "content": "正文"}]
    context = RuntimeContext(
        auto_fin_date="2026-08-10",
        auto_fin_decision_at="2026-08-10T09:30:00+08:00",
        auto_fin_window_start="2026-08-09T09:30:00+08:00",
        auto_fin_topics=["黄金", "机器人", "半导体"],
        auto_fin_news_by_topic={"黄金": gold, "机器人": robot, "半导体": []},
        auto_fin_selected_news=[*gold, *robot],
    )

    response = await AutoFinMergeStep(app_context=app_context, agent_wrapper=agent)(context)

    assert len(agent.calls) == 2
    gold_prompt, gold_kwargs = agent.calls[0]
    robot_prompt, robot_kwargs = agent.calls[1]
    assert '"news_id": "24"' in gold_prompt
    assert '"news_id": "4"' not in gold_prompt
    assert "另有 5 篇" in gold_prompt
    assert "当前主题：机器人" in robot_prompt
    assert gold_kwargs["tool_context_id"] != robot_kwargs["tool_context_id"]
    assert response.answer.count("## ") >= 2
    assert response.metadata["selected_news_count"] == 26
    report = (tmp_path / "daily" / "2026-08-10" / "auto_fin.md").read_text(encoding="utf-8")
    assert "## 黄金：" in report and "## 机器人：" in report


def test_hybrid_wikilink_normalization_is_conservative_and_failure_safe(tmp_path: Path, monkeypatch):
    import reme_auto_fin.merge as merge_module

    step = AutoFinMergeStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai"),
    )
    body = (
        "[[digest/wiki/gold.md]](digest/wiki/gold.md) "
        "[[digest/wiki/gold.md|黄金]](<digest/wiki/gold.md>) "
        "[[digest/wiki/gold.md#L2|黄金]](digest/wiki/gold.md) "
        "[[digest/wiki/gold.md]](digest/wiki/other.md)"
    )
    assert step._normalize_hybrid_wikilinks(body) == (
        "[[digest/wiki/gold.md]] "
        "[[digest/wiki/gold.md|黄金]] "
        "[[digest/wiki/gold.md#L2|黄金]] "
        "[[digest/wiki/gold.md]](digest/wiki/other.md)"
    )

    class _BrokenPattern:
        @staticmethod
        def sub(_replace, _body):
            raise RuntimeError("normalization failed")

    monkeypatch.setattr(merge_module, "_HYBRID_WIKILINK_RE", _BrokenPattern())
    assert step._normalize_hybrid_wikilinks(body) == body


def test_plugin_config_has_default_topics_and_no_intermediate_index_step():
    jobs = PLUGIN_MANIFEST["application_defaults"]["jobs"]
    job = jobs["auto_fin"]
    assert job["parameters"]["properties"]["topics"]["default"] == "黄金,机器人,半导体"
    assert job["parameters"]["properties"]["window_hours"]["default"] == 24
    assert job["parameters"]["properties"]["request_interval"]["default"] == 10
    assert job["parameters"]["properties"]["max_retries"]["default"] == 3
    assert "news_file" not in job["parameters"]["properties"]
    assert [step["backend"] for step in job["steps"]] == [
        "auto_fin_data_step",
        "auto_fin_topic_step",
        "auto_fin_merge_step",
        "auto_tag_step",
    ]
    assert job["steps"][2]["job_tools"] == ["search"]
    assert job["steps"][3] == {"backend": "auto_tag_step"}
    assert jobs["auto_fin_cron"]["cron"] == "0 9 * * *"
    assert jobs["auto_fin_cron"]["steps"] == job["steps"]
    assert (
        not {
            "auto_fin_0930_cron",
            "auto_fin_1130_cron",
            "auto_fin_1800_cron",
        }
        & jobs.keys()
    )


def test_agent_schemas_are_small_and_required():
    topic = AutoFinTopicOutput.model_json_schema()
    report = AutoFinReportOutput.model_json_schema()

    assert topic["required"] == ["news_ids"]
    assert set(topic["properties"]) == {"news_ids"}
    assert report["required"] == ["title", "description", "body"]
    assert set(report["properties"]) == {"title", "description", "body"}
