"""Focused tests for the daily-paper cookbook workflow."""

import datetime as dt
import json
from pathlib import Path
from unittest.mock import MagicMock

import frontmatter
import httpx
import pytest

from reme.components import ApplicationContext
from reme.components.agent_wrapper.base_agent_wrapper import BaseAgentWrapper
from reme.components.runtime_context import RuntimeContext
from reme.config.config_parser import _load_config
from reme.schema import DailyBriefOutput, PaperInfo, PaperNoteOutput, PaperSelection
from reme.steps.cookbook.daily_paper import (
    DailyPaperAnalyzeStep,
    DailyPaperCollectStep,
    DailyPaperDigestStep,
    DailyPaperRankStep,
    DailyPaperSelectStep,
)
from reme.steps.cookbook.daily_paper import analyze, collect
from reme.steps.cookbook.daily_paper.rank import build_candidate_pool, rrf_score
from reme.steps.cookbook.dingtalk import DingTalkMarkdownSendStep
from reme.steps.cookbook.dingtalk import send as dingtalk_send
from reme.utils.huggingface_papers import paper_ids_from_html, paper_info_from_payload


class _QueuedAgentWrapper(BaseAgentWrapper):
    """Return queued structured responses without contacting an LLM."""

    def __init__(self, outputs: list[dict], **kwargs):
        super().__init__(**kwargs)
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    async def reply(self, inputs, **kwargs) -> dict:
        """Record the request and pop the next structured fixture."""
        self.calls.append({"inputs": inputs, "kwargs": kwargs})
        return {"structured_output": self.outputs.pop(0), "result": "ok"}


def _paper(arxiv_id: str, *, title: str = "Paper", upvotes: int = 10) -> PaperInfo:
    return PaperInfo(
        arxiv_id=arxiv_id,
        title=title,
        summary=f"Summary for {title}",
        authors=["A. Author"],
        upvotes=upvotes,
    )


def test_hf_payload_and_html_normalization():
    """HF list/detail shapes normalize and HTML rank order de-duplicates."""
    payload = {
        "paper": {
            "id": "2607.16051",
            "title": "Loop the Loopies!",
            "summary": "Abstract",
            "authors": [{"name": "Zitian Gao"}],
            "upvotes": 53,
            "githubRepo": "https://github.com/example/repo",
        },
        "organization": {"fullname": "IQuest"},
    }

    paper = paper_info_from_payload(payload)

    assert paper.arxiv_id == "2607.16051"
    assert paper.authors == ["Zitian Gao"]
    assert paper.organization == "IQuest"
    assert paper.github_repo == "https://github.com/example/repo"
    assert paper_ids_from_html(
        '<a href="/papers/2607.16051">one</a><a href="/papers/2607.16051">dup</a>'
        '<a href="/papers/2607.10001">two</a>',
    ) == ["2607.16051", "2607.10001"]


def test_rrf_and_memory_candidate_reserve():
    """RRF is exact and the candidate pool preserves a memory-related slot."""
    general = _paper("2607.10001", title="General model", upvotes=100)
    memory = _paper("2607.10002", title="Long-term memory for agents", upvotes=1)
    general.fused_score = rrf_score(1, None, rrf_k=60, weekly_weight=0.7)
    memory.fused_score = rrf_score(100, None, rrf_k=60, weekly_weight=0.7)

    candidates = build_candidate_pool([general, memory], limit=2, memory_reserve=1)

    assert candidates == [general, memory]
    assert general.fused_score == pytest.approx(1 / 61)


def test_history_exclusion_reads_prior_frontmatter_only(tmp_path: Path):
    """Only prior dated paper notes contribute historical exclusions."""
    prior = tmp_path / "daily" / "2026-07-20"
    current = tmp_path / "daily" / "2026-07-21"
    prior.mkdir(parents=True)
    current.mkdir(parents=True)
    (prior / "paper-2607.10001.md").write_text(
        frontmatter.dumps(frontmatter.Post("body", arxiv_id="2607.10001")),
        encoding="utf-8",
    )
    (current / "paper-2607.10002.md").write_text(
        frontmatter.dumps(frontmatter.Post("body", arxiv_id="2607.10002")),
        encoding="utf-8",
    )

    found = DailyPaperCollectStep.load_historical_arxiv_ids(
        tmp_path,
        dt.date(2026, 7, 21),
        30,
        "daily",
    )

    assert found == {"2607.10001"}


def test_standalone_config_uses_only_claude_code_and_eight_am_cron(monkeypatch):
    """The standalone config schedules 08:00 and routes all agent work to CC."""
    for name in (
        "DINGTALK_APP_KEY",
        "DINGTALK_APP_SECRET",
        "DINGTALK_ROBOT_CODE",
        "DINGTALK_CONVERSATION_IDS",
    ):
        monkeypatch.delenv(name, raising=False)
    config = _load_config("daily_cookbook")

    assert config.get("extends") is None
    assert config["jobs"]["daily_paper_cron"]["cron"] == "0 8 * * *"
    steps = config["jobs"]["daily_paper"]["steps"]
    assert config["jobs"]["daily_paper_cron"]["steps"] == steps
    agent_steps = [
        step
        for step in steps
        if step["backend"]
        in {
            "daily_paper_select_step",
            "daily_paper_analyze_step",
            "daily_paper_digest_step",
        }
    ]
    assert {step.get("agent_wrapper") for step in agent_steps} == {"claude_code"}
    assert steps[-1] == {
        "backend": "dingtalk_markdown_send_step",
        "input_mapping": {"daily_paper_digest_path": "markdown_path"},
        "app_key": "",
        "app_secret": "",
        "robot_code": "",
        "conversation_ids": "",
        "title": "ReMe Daily Paper",
        "timeout": 15,
    }
    assert set(config["components"]["agent_wrapper"]) == {"claude_code"}
    assert "as_llm" not in config["components"]
    assert config["components"]["agent_wrapper"]["claude_code"]["project_path"] == ".."
    assert "skills" not in config["components"]["agent_wrapper"]["claude_code"]


def test_daily_paper_config_passes_dingtalk_environment(monkeypatch):
    """The notifier receives all proactive-message settings from the environment."""
    values = {
        "DINGTALK_APP_KEY": "app-key",
        "DINGTALK_APP_SECRET": "app-secret",
        "DINGTALK_ROBOT_CODE": "robot-code",
        "DINGTALK_CONVERSATION_IDS": "group-one,group-two",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    step = _load_config("daily_cookbook")["jobs"]["daily_paper"]["steps"][-1]

    assert {key: step[key] for key in ("app_key", "app_secret", "robot_code", "conversation_ids")} == {
        "app_key": "app-key",
        "app_secret": "app-secret",
        "robot_code": "robot-code",
        "conversation_ids": "group-one,group-two",
    }


@pytest.mark.asyncio
async def test_pipeline_filters_strict_yesterday_and_writes_outputs(
    tmp_path: Path,
    monkeypatch,
):
    """The complete mocked pipeline filters yesterday/history and writes linked notes."""
    papers = {
        "2607.10001": _paper("2607.10001", title="Best monthly paper", upvotes=100),
        "2607.10002": _paper("2607.10002", title="Yesterday paper", upvotes=90),
        "2607.10003": _paper("2607.10003", title="Previously recommended", upvotes=80),
    }

    prior_dir = tmp_path / "daily" / "2026-07-19"
    prior_dir.mkdir(parents=True)
    (prior_dir / "paper-2607.10003.md").write_text(
        frontmatter.dumps(frontmatter.Post("old", arxiv_id="2607.10003")),
        encoding="utf-8",
    )

    class _FakeHfClient:
        requested_daily: list[str] = []

        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def fetch_scope(self, scope: str, value: str):
            """Return deterministic weekly/monthly fixtures."""
            if scope == "month":
                assert value == "2026-07"
                return list(papers.values())
            assert value == "2026-W30"
            return [papers["2607.10001"], papers["2607.10002"]]

        async def fetch_daily_ids(self, day: str):
            """Record and return the exact requested day."""
            self.requested_daily.append(day)
            return {"2607.10002"}

    async def fake_download(_self, _arxiv_id: str, target: Path):
        """Create a minimal cached-PDF fixture."""
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"%PDF-fake")
        return target

    def fake_extract(_self, _path: Path, _max_pages: int, _max_chars: int):
        """Return deterministic extracted text."""
        return "--- PAGE 1 ---\nPaper content", 1, False

    monkeypatch.setattr(collect, "HuggingFacePapersClient", _FakeHfClient)
    monkeypatch.setattr(analyze.ArxivPdfClient, "download", fake_download)
    monkeypatch.setattr(analyze.DailyPaperAnalyzeStep, "_extract_pdf_text_sync", fake_extract)

    cc_wrapper = _QueuedAgentWrapper(
        [
            {
                "selection_reasoning": "Best remaining ranked paper.",
                "selected": [
                    {
                        "arxiv_id": "2607.10001",
                        "rank": 1,
                        "reason": "Strong result",
                        "memory_relevance": "low",
                    },
                ],
                "alternates": [],
            },
            {
                "description": "Detailed note",
                "body": "# Detailed reading\n\nEvidence [p. 1].",
            },
            {
                "description": "Five-minute brief",
                "body": "# 今日论文速读\n\n[[daily/2026-07-21/paper-2607.10001.md]]",
            },
        ],
    )
    app_context = ApplicationContext(
        workspace_dir=str(tmp_path),
        timezone="Asia/Shanghai",
        language="zh",
    )
    context = RuntimeContext(
        date="2026-07-21",
        top_k=1,
        candidate_limit=2,
        memory_reserve=0,
    )

    await DailyPaperCollectStep(app_context=app_context)(context)
    await DailyPaperRankStep(app_context=app_context)(context)
    await DailyPaperSelectStep(app_context=app_context, agent_wrapper=cc_wrapper)(context)
    await DailyPaperAnalyzeStep(app_context=app_context, agent_wrapper=cc_wrapper)(context)
    await DailyPaperDigestStep(app_context=app_context, agent_wrapper=cc_wrapper)(context)

    assert _FakeHfClient.requested_daily == ["2026-07-20"]
    assert context.response.metadata["selected_arxiv_ids"] == ["2607.10001"]
    assert context.response.metadata["excluded_yesterday_count"] == 1
    assert context.response.metadata["excluded_history_count"] == 1
    note_path = tmp_path / "daily" / "2026-07-21" / "paper-2607.10001.md"
    digest_path = tmp_path / "daily" / "2026-07-21" / "daily-paper-brief.md"
    assert frontmatter.load(note_path).metadata["arxiv_id"] == "2607.10001"
    assert "[[daily/2026-07-21/paper-2607.10001.md]]" in digest_path.read_text(
        encoding="utf-8",
    )
    manifest = json.loads(
        (tmp_path / "metadata" / "daily_paper" / "2026-07-21.json").read_text(),
    )
    assert manifest["status"] == "complete"
    assert manifest["thinking"] == "Best remaining ranked paper."
    assert manifest["top_arxiv_ids"] == ["2607.10001"]
    assert all(set(call["kwargs"]) == {"output_schema"} for call in cc_wrapper.calls)
    assert [call["kwargs"]["output_schema"] for call in cc_wrapper.calls] == [
        PaperSelection,
        PaperNoteOutput,
        DailyBriefOutput,
    ]
    analysis_prompt = cc_wrapper.calls[1]["inputs"]
    assert "长期记忆相关性初筛：low" in analysis_prompt
    assert "必须先使用代码读取和搜索工具查看当前 ReMe 代码仓库" in analysis_prompt
    assert "这应当是少数例外：一般情况下不要给建议" in analysis_prompt

    rerun = RuntimeContext(date="2026-07-21")
    await DailyPaperCollectStep(app_context=app_context)(rerun)
    assert rerun.response.metadata["skipped"] is True
    assert rerun.get("daily_paper_digest_path") == "daily/2026-07-21/daily-paper-brief.md"
    assert _FakeHfClient.requested_daily == ["2026-07-20"]


@pytest.mark.asyncio
async def test_dingtalk_markdown_sends_groups_serially_in_configured_order(tmp_path: Path, monkeypatch):
    """The notifier gets one app token and posts once per group in list order."""
    digest_path = tmp_path / "daily" / "2026-07-21" / "daily-paper-brief.md"
    digest_path.parent.mkdir(parents=True)
    digest_path.write_text(
        frontmatter.dumps(frontmatter.Post("# 今日论文\n\n测试内容", name="daily-paper-brief")),
        encoding="utf-8",
    )
    token_calls = 0
    seen_payloads: list[dict] = []

    def get_access_token(client):
        nonlocal token_calls
        token_calls += 1
        assert client.credential.client_id == "app-key"
        assert client.credential.client_secret == "app-secret"
        return "app-access-token"

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1.0/robot/groupMessages/send"
        assert request.headers["x-acs-dingtalk-access-token"] == "app-access-token"
        seen_payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"processQueryKey": f"query-{len(seen_payloads)}"})

    transport = httpx.MockTransport(handler)
    transport_kwargs: dict = {}

    def ipv4_transport(**kwargs):
        transport_kwargs.update(kwargs)
        return transport

    monkeypatch.setattr(dingtalk_send.dingtalk_stream.DingTalkStreamClient, "get_access_token", get_access_token)
    monkeypatch.setattr(dingtalk_send.httpx, "AsyncHTTPTransport", ipv4_transport)
    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    context = RuntimeContext(markdown_path="daily/2026-07-21/daily-paper-brief.md")

    step = DingTalkMarkdownSendStep(
        app_context=app_context,
        app_key="app-key",
        app_secret="app-secret",
        robot_code="robot-code",
        conversation_ids=" group-one,group-two ",
        title="ReMe Daily Paper",
    )
    step.logger = MagicMock()
    response = await step(context)

    assert token_calls == 1
    assert transport_kwargs == {"local_address": "0.0.0.0"}
    assert [payload["openConversationId"] for payload in seen_payloads] == ["group-one", "group-two"]
    assert all(payload["robotCode"] == "robot-code" for payload in seen_payloads)
    assert all(payload["msgKey"] == "sampleMarkdown" for payload in seen_payloads)
    assert [json.loads(payload["msgParam"]) for payload in seen_payloads] == [
        {"title": "ReMe Daily Paper", "text": "# 今日论文\n\n测试内容"},
    ] * 2
    assert response.metadata["dingtalk_configured_count"] == 2
    assert response.metadata["dingtalk_sent_count"] == 2
    logs = "\n".join(call.args[0] for call in step.logger.info.call_args_list)
    assert "sending DingTalk Markdown" in logs
    assert "delivery complete sent=2 total=2" in logs
    assert all(value not in logs for value in ("app-key", "app-secret", "robot-code", "group-one", "group-two"))


@pytest.mark.asyncio
async def test_dingtalk_markdown_without_conversations_is_a_noop(tmp_path: Path):
    """An empty conversation list keeps daily-paper generation usable without DingTalk."""
    context = RuntimeContext(markdown_path="missing.md")

    response = await DingTalkMarkdownSendStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))(context)

    assert response.success is True
    assert response.metadata["dingtalk_configured_count"] == 0
    assert response.metadata["dingtalk_sent_count"] == 0


@pytest.mark.asyncio
async def test_existing_daily_paper_is_reused_and_sent_to_dingtalk(tmp_path: Path, monkeypatch):
    """An idempotent daily-paper run skips generation but still notifies DingTalk."""
    digest_path = tmp_path / "daily" / "2026-07-22" / "daily-paper-brief.md"
    digest_path.parent.mkdir(parents=True)
    digest_path.write_text(
        frontmatter.dumps(frontmatter.Post("# 已有日报\n\n复用正文", name="daily-paper-brief")),
        encoding="utf-8",
    )
    seen_payloads: list[dict] = []

    monkeypatch.setattr(
        dingtalk_send.dingtalk_stream.DingTalkStreamClient,
        "get_access_token",
        lambda _client: "app-access-token",
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"processQueryKey": "query-1"})

    transport = httpx.MockTransport(handler)

    monkeypatch.setattr(dingtalk_send.httpx, "AsyncHTTPTransport", lambda **_kwargs: transport)
    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    context = RuntimeContext(date="2026-07-22")

    await DailyPaperCollectStep(app_context=app_context)(context)
    response = await DingTalkMarkdownSendStep(
        app_context=app_context,
        input_mapping={"daily_paper_digest_path": "markdown_path"},
        app_key="app-key",
        app_secret="app-secret",
        robot_code="robot-code",
        conversation_ids="existing-group",
        title="ReMe Daily Paper",
    )(context)

    assert response.metadata["skipped"] is True
    assert response.metadata["dingtalk_sent_count"] == 1
    assert seen_payloads == [
        {
            "robotCode": "robot-code",
            "openConversationId": "existing-group",
            "msgKey": "sampleMarkdown",
            "msgParam": json.dumps(
                {"title": "ReMe Daily Paper", "text": "# 已有日报\n\n复用正文"},
                ensure_ascii=False,
            ),
        },
    ]
