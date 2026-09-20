"""Focused tests for the rolling CLS Auto Fin workflow."""

# pylint: disable=missing-function-docstring,protected-access

import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

from reme_auto_fin.base import (
    FIRST_RUN_NOTICE,
    normalize_hybrid_wikilinks,
    normalize_title,
    plain_text,
    read_note,
    write_markdown,
)
from reme_auto_fin.data import AutoFinDataStep
from reme_auto_fin.digest import AutoFinDigestStep
from reme_auto_fin.research import AutoFinResearchStep
from reme_auto_fin.schema import AutoFinNote, AutoFinReportOutput
from reme_auto_fin.topic import AutoFinTopicStep
from reme.components import ApplicationContext
from reme.components.agent_wrapper.base_agent_wrapper import BaseAgentWrapper
from reme.components.runtime_context import RuntimeContext
from reme.utils.wikilink_handler import WikilinkHandler

SHANGHAI = ZoneInfo("Asia/Shanghai")
PLUGIN_MANIFEST = yaml.safe_load(
    (Path(__file__).parents[1] / "src" / "reme_auto_fin" / "plugin.yaml").read_text(encoding="utf-8"),
)
LINKED_BODY = (
    "## 今日判断\n\n"
    "CLS 1（09:00，黄金上涨）与 "
    "[[daily/2026-08-01/auto_fin.md|历史黄金观察]]"
    "(daily/2026-08-01/auto_fin.md) 背景相似。\n\n"
    "无效引用 [[daily/missing.md|缺失文章]] 和 [[../../outside.md|越界文章]] 应降级。"
)


def _row(news_id: int, value: datetime, title: str = "新闻", content: str = "正文") -> dict:
    return {
        "id": news_id,
        "ctime": int(value.timestamp()),
        "title": title,
        "content": content,
    }


def _news(news_id: str, event_time: str, title: str = "新闻") -> dict:
    return {"news_id": news_id, "event_time": event_time, "title": title, "content": "正文"}


def _context(**kwargs) -> RuntimeContext:
    defaults = {
        "auto_fin_date": "2026-08-10",
        "auto_fin_decision_at": "2026-08-10T09:30:00+08:00",
        "auto_fin_window_start": "2026-08-09T09:30:00+08:00",
        "auto_fin_topics": ["黄金", "机器人", "半导体"],
        "auto_fin_selected_news": [_news("1", "2026-08-10T09:00:00+08:00")],
    }
    return RuntimeContext(**{**defaults, **kwargs})


def _history(tmp_path: Path) -> None:
    historical = tmp_path / "daily" / "2026-08-01" / "auto_fin.md"
    historical.parent.mkdir(parents=True, exist_ok=True)
    historical.write_text("# 历史黄金观察\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_write_markdown_preserves_existing_file_on_failure(tmp_path: Path, monkeypatch):
    path = tmp_path / "result.md"
    path.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(
        "reme_auto_fin.base.os.replace",
        lambda *_args: (_ for _ in ()).throw(OSError()),
    )

    with pytest.raises(OSError):
        await write_markdown(path, "replacement", {"name": "标题"})

    assert path.read_text(encoding="utf-8") == "existing"
    assert not list(tmp_path.glob(".*.tmp"))


def test_plain_text_drops_hidden_and_unescapes_entities():
    assert plain_text("<p>甲&amp;乙</p><style>隐藏</style><p>丙</p>") == "甲&乙 丙"


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


class _ReportAgent(BaseAgentWrapper):
    """Return one titled Markdown report per call, keyed off the research topic."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    async def reply(self, inputs, **kwargs):
        prompt = str(inputs)
        self.calls.append((prompt, kwargs))
        topic = re.search(r"当前主题：(\S+)", prompt)
        title = f"{topic.group(1)}观察" if topic else "主题新闻观察"
        return {
            "structured_output": AutoFinReportOutput(
                title=f"# {title}",
                description="关注政策变化。",
                body=LINKED_BODY,
            ),
        }


@pytest.mark.asyncio
async def test_research_writes_one_note_per_topic_with_latest_twenty_news(tmp_path: Path):
    _history(tmp_path)
    app_context = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai")
    agent = _ReportAgent(app_context=app_context)
    gold = [_news(str(index), f"2026-08-10T09:{index:02}:00+08:00", "黄金") for index in range(25)]
    context = _context(
        auto_fin_news_by_topic={"黄金": gold, "机器人": [_news("30", "2026-08-10T08:00:00+08:00")], "半导体": []},
        auto_fin_selected_news=[*gold, _news("30", "2026-08-10T08:00:00+08:00")],
    )

    response = await AutoFinResearchStep(app_context=app_context, agent_wrapper=agent)(context)

    assert len(agent.calls) == 2
    gold_prompt, gold_kwargs = agent.calls[0]
    assert '"news_id": "24"' in gold_prompt
    assert '"news_id": "4"' not in gold_prompt
    assert "另有 5 篇" in gold_prompt
    assert "当前主题：机器人" in agent.calls[1][0]
    assert gold_kwargs["job_tools"] == ["search"]
    assert gold_kwargs["output_schema"] == AutoFinReportOutput
    assert gold_kwargs["tool_context_id"].startswith("auto_fin:")
    assert gold_kwargs["tool_context_id"] != agent.calls[1][1]["tool_context_id"]
    assert gold_kwargs["injected_job_kwargs"] == {
        "limit": 5,
        "min_score": 0.0,
        "start_date": None,
        "end_date": "2026-08-09",
        "max_search_calls": 3,
    }

    day = tmp_path / "daily" / "2026-08-10"
    assert sorted(path.name for path in day.glob("*.md")) == ["机器人.md", "黄金.md"]
    note = (day / "黄金.md").read_text(encoding="utf-8")
    assert "kind: auto-fin-topic" in note and "topic: 黄金" in note
    assert "title: 黄金观察" in note
    assert "[[daily/2026-08-01/auto_fin.md|历史黄金观察]]" in note
    assert "](daily/2026-08-01/auto_fin.md)" not in note
    assert "缺失文章" in note and "越界文章" in note
    assert "missing.md" not in note and "outside.md" not in note
    assert "不提供收益、目标价或买卖建议" in note

    assert [item.path for item in context["auto_fin_notes"]] == [
        "daily/2026-08-10/黄金.md",
        "daily/2026-08-10/机器人.md",
    ]
    assert [item.title for item in context["auto_fin_notes"]] == ["黄金观察", "机器人观察"]
    assert context["changes"] == [
        {"change": "added", "path": "daily/2026-08-10/黄金.md"},
        {"change": "added", "path": "daily/2026-08-10/机器人.md"},
    ]
    assert response.metadata["selected_news_count"] == 26
    assert response.metadata["note_paths"] == [item.path for item in context["auto_fin_notes"]]


@pytest.mark.asyncio
async def test_research_names_the_note_after_the_topic_not_the_agent_title(tmp_path: Path):
    """A whole-paragraph Agent title used to become a filename and fail with ENAMETOOLONG."""

    class _VerboseAgent(BaseAgentWrapper):
        async def reply(self, _prompt, **_kwargs):
            return {
                "structured_output": AutoFinReportOutput(
                    title="机器人主题 9-18 研究：" + "量产与政策" * 60,
                    description="关注政策变化。",
                    body=LINKED_BODY,
                ),
            }

    app_context = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai")
    context = _context(
        auto_fin_news_by_topic={"黄金": [], "机器人": [_news("1", "2026-08-10T09:00:00+08:00")], "半导体": []},
    )

    await AutoFinResearchStep(app_context=app_context, agent_wrapper=_VerboseAgent(app_context=app_context))(context)

    day = tmp_path / "daily" / "2026-08-10"
    assert [path.name for path in day.glob("*.md")] == ["机器人.md"]
    assert "title: 机器人主题 9-18 研究：量产与政策" in (day / "机器人.md").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_research_rerun_replaces_the_same_note_for_a_topic(tmp_path: Path):
    app_context = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai")
    agent = _ReportAgent(app_context=app_context)
    news = {"黄金": [_news("1", "2026-08-10T09:00:00+08:00")], "机器人": [], "半导体": []}
    step = AutoFinResearchStep(app_context=app_context, agent_wrapper=agent)

    first = _context(auto_fin_news_by_topic=news)
    await step(first)
    second = _context(auto_fin_news_by_topic=news)
    await step(second)

    assert len(agent.calls) == 2
    assert "本次为当日首次生成。" in agent.calls[0][0]
    assert "## 今日判断" in agent.calls[1][0]
    assert second["changes"] == [{"change": "modified", "path": "daily/2026-08-10/黄金.md"}]
    assert [path.name for path in (tmp_path / "daily" / "2026-08-10").glob("*.md")] == ["黄金.md"]


@pytest.mark.asyncio
async def test_research_skips_topics_without_news_and_honours_the_skip_flag(tmp_path: Path):
    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    agent = _ReportAgent(app_context=app_context)
    context = _context(auto_fin_news_by_topic={"黄金": [], "机器人": [], "半导体": []})

    response = await AutoFinResearchStep(app_context=app_context, agent_wrapper=agent)(context)

    assert not agent.calls
    assert context["auto_fin_notes"] == []
    assert response.metadata["note_paths"] == []
    assert not (tmp_path / "daily").exists()

    skipped = _context(auto_fin_skipped=True, auto_fin_news_by_topic={"黄金": []})
    await AutoFinResearchStep(app_context=app_context, agent_wrapper=agent)(skipped)
    assert len(agent.calls) == 0


@pytest.mark.asyncio
async def test_research_keeps_the_other_topics_when_one_fails(tmp_path: Path):
    """One broken topic must not discard the notes the other topics already produced."""

    class _FlakyAgent(_ReportAgent):
        async def reply(self, inputs, **kwargs):
            if "当前主题：机器人" in str(inputs):
                raise RuntimeError("agent exploded")
            return await super().reply(inputs, **kwargs)

    app_context = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai")
    agent = _FlakyAgent(app_context=app_context)
    context = _context(
        auto_fin_news_by_topic={
            "黄金": [_news("1", "2026-08-10T09:00:00+08:00")],
            "机器人": [_news("2", "2026-08-10T09:05:00+08:00")],
            "半导体": [],
        },
    )

    response = await AutoFinResearchStep(app_context=app_context, agent_wrapper=agent)(context)

    assert [item.path for item in context["auto_fin_notes"]] == ["daily/2026-08-10/黄金.md"]
    assert response.success is True
    assert response.metadata["failed_topics"] == [{"topic": "机器人", "error": "agent exploded"}]
    assert [path.name for path in (tmp_path / "daily" / "2026-08-10").glob("*.md")] == ["黄金.md"]


@pytest.mark.asyncio
async def test_research_ignores_notes_with_unparsable_frontmatter(tmp_path: Path):
    """A note the user is editing by hand must not abort every topic in the run."""

    day = tmp_path / "daily" / "2026-08-10"
    day.mkdir(parents=True)
    hand_edited = "---\ntags: [unfinished\n---\n\n正在编辑的笔记。\n"
    (day / "手记.md").write_text(hand_edited, encoding="utf-8")
    app_context = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai")
    context = _context(
        auto_fin_news_by_topic={"黄金": [_news("1", "2026-08-10T09:00:00+08:00")], "机器人": [], "半导体": []},
    )

    response = await AutoFinResearchStep(app_context=app_context, agent_wrapper=_ReportAgent(app_context=app_context))(
        context,
    )

    assert response.success is True
    assert response.metadata["failed_topics"] == []
    assert [item.path for item in context["auto_fin_notes"]] == ["daily/2026-08-10/黄金.md"]
    assert (day / "手记.md").read_text(encoding="utf-8") == hand_edited


def test_read_note_falls_back_when_the_frontmatter_is_unparsable(tmp_path: Path):
    broken = tmp_path / "手记.md"
    broken.write_text("---\ntags: [unfinished\n---\n\n正文\n", encoding="utf-8")

    assert read_note(broken) == FIRST_RUN_NOTICE
    assert read_note(tmp_path / "missing.md") == FIRST_RUN_NOTICE
    assert read_note(None) == FIRST_RUN_NOTICE


@pytest.mark.asyncio
async def test_research_fails_the_run_when_every_topic_fails(tmp_path: Path):
    """A run that produced no note at all must fail instead of sending an empty brief."""

    class _DeadAgent(BaseAgentWrapper):
        async def reply(self, *_args, **_kwargs):
            raise RuntimeError("agent exploded")

    app_context = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai")
    context = _context(
        auto_fin_news_by_topic={"黄金": [_news("1", "2026-08-10T09:00:00+08:00")], "机器人": [], "半导体": []},
    )

    with pytest.raises(RuntimeError, match="failed for every topic"):
        await AutoFinResearchStep(app_context=app_context, agent_wrapper=_DeadAgent(app_context=app_context))(context)


@pytest.mark.asyncio
async def test_digest_merges_notes_and_links_back_to_each_of_them(tmp_path: Path):
    _history(tmp_path)
    app_context = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai")
    agent = _ReportAgent(app_context=app_context)
    context = _context(
        auto_fin_news_by_topic={"黄金": [_news("1", "2026-08-10T09:00:00+08:00")], "机器人": [], "半导体": []},
    )
    await AutoFinResearchStep(app_context=app_context, agent_wrapper=agent)(context)
    note_path = context["auto_fin_notes"][0].path

    response = await AutoFinDigestStep(app_context=app_context, agent_wrapper=agent)(context)

    prompt, kwargs = agent.calls[-1]
    assert kwargs["job_tools"] == []
    assert '"topic": "黄金"' in prompt
    assert "各主题笔记" in prompt
    assert "当前主题：" not in prompt
    assert "本次为当日首次生成。" in prompt

    digest_path = "daily/2026-08-10/主题新闻观察（2026-08-10）.md"
    digest = (tmp_path / digest_path).read_text(encoding="utf-8")
    assert "kind: auto-fin-digest" in digest
    assert "title: 主题新闻观察" in digest
    assert "## 主题详解" in digest
    assert f"- [[{note_path}]]" in digest
    assert "[[daily/2026-08-01/auto_fin.md|历史黄金观察]]" in digest
    assert "缺失文章" in digest and "missing.md" not in digest
    assert digest.rstrip().endswith("不提供收益、目标价或买卖建议。")

    assert context["markdown_path"] == digest_path
    assert context["changes"][-1] == {"change": "added", "path": digest_path}
    assert response.metadata["digest_path"] == context["markdown_path"]
    assert response.metadata["source_paths"] == ["daily/2026-08-01/auto_fin.md"]
    assert response.metadata["note_paths"] == [note_path]


@pytest.mark.parametrize("topic", ["黄金", "AI[算力]", "C#", "新能源/储能", "#热点", "a|b"])
@pytest.mark.asyncio
async def test_digest_trailer_resolves_to_each_topic_note(tmp_path: Path, topic: str):
    """A topic name must not smuggle a wikilink delimiter into the trailer it lands in."""

    app_context = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai")
    agent = _ReportAgent(app_context=app_context)
    context = _context(
        auto_fin_topics=[topic],
        auto_fin_news_by_topic={topic: [_news("1", "2026-08-10T09:00:00+08:00")]},
    )
    await AutoFinResearchStep(app_context=app_context, agent_wrapper=agent)(context)
    note_path = context["auto_fin_notes"][0].path

    response = await AutoFinDigestStep(app_context=app_context, agent_wrapper=agent)(context)

    digest = (tmp_path / response.metadata["digest_path"]).read_text(encoding="utf-8")
    targets = [match.target for match in WikilinkHandler.iter_matches(digest)]
    assert (tmp_path / note_path).is_file()
    assert note_path in targets
    # Every ``[[`` the digest emits opens a link the workspace parser can read.
    assert digest.count("[[") == len(targets)


@pytest.mark.asyncio
async def test_digest_returns_the_validated_body_to_the_caller(tmp_path: Path):
    """The API answer must not carry a link that the note itself downgraded to plain text."""

    app_context = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai")
    agent = _ReportAgent(app_context=app_context)
    context = _context(
        auto_fin_news_by_topic={"黄金": [_news("1", "2026-08-10T09:00:00+08:00")], "机器人": [], "半导体": []},
    )
    await AutoFinResearchStep(app_context=app_context, agent_wrapper=agent)(context)

    response = await AutoFinDigestStep(app_context=app_context, agent_wrapper=agent)(context)

    digest = (tmp_path / response.metadata["digest_path"]).read_text(encoding="utf-8")
    assert "缺失文章" in response.answer
    assert "missing.md" not in response.answer
    assert response.answer in digest


@pytest.mark.asyncio
async def test_digest_skips_when_the_run_was_already_skipped(tmp_path: Path):
    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    agent = _ReportAgent(app_context=app_context)
    context = _context(auto_fin_skipped=True)

    await AutoFinDigestStep(app_context=app_context, agent_wrapper=agent)(context)

    assert not agent.calls
    assert not (tmp_path / "daily").exists()


def test_normalize_hybrid_wikilinks_is_conservative():
    body = (
        "[[digest/wiki/gold.md]](digest/wiki/gold.md) "
        "[[digest/wiki/gold.md|黄金]](<digest/wiki/gold.md>) "
        "[[digest/wiki/gold.md#L2|黄金]](digest/wiki/gold.md) "
        "[[digest/wiki/gold.md]](digest/wiki/other.md)"
    )

    assert normalize_hybrid_wikilinks(body) == (
        "[[digest/wiki/gold.md]] "
        "[[digest/wiki/gold.md|黄金]] "
        "[[digest/wiki/gold.md#L2|黄金]] "
        "[[digest/wiki/gold.md]](digest/wiki/other.md)"
    )


def test_normalize_title_sanitizes_agent_titles_and_keeps_notes_importable():
    assert normalize_title("# 黄金/政策：观察", "黄金观察") == "黄金-政策：观察"
    assert normalize_title("  ", "黄金观察") == "黄金观察"
    assert normalize_title("解读.md", "黄金观察") == "解读"
    assert AutoFinNote(topic="黄金", title="标题", description="说明", body="正文", path="a.md").topic == "黄金"


def test_normalize_title_keeps_wikilink_delimiters_out_of_the_stem():
    """`AI[算力]` used to produce a stem the wikilink parser could not read at all."""

    assert normalize_title("AI[算力]", "主题观察") == "AI-算力"
    assert normalize_title("C#", "主题观察") == "C"
    assert normalize_title("# 热点", "主题观察") == "热点"
    for raw in ("AI[算力]", "C#", "# 热点", "a|b"):
        assert not set("[]#|") & set(normalize_title(raw, "主题观察"))


def test_normalize_title_fits_the_filename_component_byte_budget():
    """A whole-paragraph title used to reach os.stat and fail with ENAMETOOLONG."""
    assert normalize_title("长" * 200, "黄金观察") == "长" * 60
    assert len(normalize_title("long" * 100, "黄金观察").encode()) <= 180
    assert normalize_title("  /  ", "黄金观察") == "黄金观察"


def test_plugin_config_has_default_topics_and_two_report_steps():
    jobs = PLUGIN_MANIFEST["application_defaults"]["jobs"]
    job = jobs["auto_fin"]
    assert job["parameters"]["properties"]["topics"]["default"] == "黄金,机器人,半导体"
    assert job["parameters"]["properties"]["window_hours"]["default"] == 24
    assert job["parameters"]["properties"]["request_interval"]["default"] == 10
    assert job["parameters"]["properties"]["max_retries"]["default"] == 3
    assert "news_file" not in job["parameters"]["properties"]
    assert job["steps"] == [
        {"backend": "auto_fin_data_step"},
        {"backend": "auto_fin_topic_step"},
        {"backend": "auto_fin_research_step", "job_tools": ["search"]},
        {"backend": "auto_fin_digest_step"},
        {"backend": "auto_tag_step"},
    ]
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


def test_report_schema_is_small_and_required():
    report = AutoFinReportOutput.model_json_schema()

    assert report["required"] == ["title", "description", "body"]
    assert set(report["properties"]) == {"title", "description", "body"}
