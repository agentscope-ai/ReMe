"""Focused tests for the daily-paper cookbook workflow."""

from contextlib import asynccontextmanager
import datetime as dt
import importlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock

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
from reme.utils import arxiv as arxiv_utils
from reme.utils import huggingface_papers as hf_utils
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


@pytest.mark.asyncio
async def test_hf_client_uses_configured_ssh_proxy(monkeypatch):
    """The owned HTTP client lives inside the temporary SSH proxy context."""
    monkeypatch.setenv("REME_PROXY_IP", "proxy.example.com")
    monkeypatch.setenv("REME_PROXY_ACCOUNT", "reme-user")
    events: list[str] = []
    client_kwargs: dict = {}
    logger = MagicMock()

    @asynccontextmanager
    async def fake_proxy(*, connect_timeout):
        assert connect_timeout == 12.0
        events.append("proxy-start")
        yield "socks5://127.0.0.1:43123"
        events.append("proxy-stop")

    class FakeAsyncClient:
        """Capture construction and close ordering without network access."""

        def __init__(self, **kwargs):
            client_kwargs.update(kwargs)

        async def aclose(self):
            """Record deterministic client cleanup."""
            events.append("client-close")

    monkeypatch.setattr(hf_utils, "ssh_socks_proxy", fake_proxy)
    monkeypatch.setattr(hf_utils.httpx, "AsyncClient", FakeAsyncClient)

    client = hf_utils.HuggingFacePapersClient(timeout=12.0, logger=logger)
    assert client.client is None
    async with client:
        assert client_kwargs["proxy"] == "socks5://127.0.0.1:43123"

    assert events == ["proxy-start", "client-close", "proxy-stop"]
    info_messages = [call.args[0] for call in logger.info.call_args_list]
    assert any("network mode=ssh_socks destination=reme-user@proxy.example.com" in message for message in info_messages)
    assert info_messages[-1] == "[HuggingFacePapersClient] SSH proxy closed"


@pytest.mark.asyncio
async def test_hf_client_logs_retry_after_http_error(monkeypatch):
    """Transient HTTP failures report the attempt before retrying."""
    attempts = 0
    logger = MagicMock()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectTimeout("timed out", request=request)
        return httpx.Response(200, json=[])

    sleep = AsyncMock()
    monkeypatch.setattr(hf_utils.asyncio, "sleep", sleep)
    async with httpx.AsyncClient(
        base_url="https://huggingface.co",
        transport=httpx.MockTransport(handler),
    ) as raw_client:
        client = hf_utils.HuggingFacePapersClient(client=raw_client, max_retries=2, logger=logger)
        paper_ids = await client.fetch_daily_ids("2026-07-22")

    assert paper_ids == set()
    assert attempts == 2
    sleep.assert_awaited_once_with(0.25)
    warning = logger.warning.call_args.args[0]
    assert "request retry path=/api/daily_papers attempt=1/2" in warning
    assert "error=ConnectTimeout detail=timed out" in warning


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


@pytest.mark.asyncio
async def test_arxiv_pdf_downloads_missing_cache_once(tmp_path: Path, monkeypatch):
    """A missing PDF is downloaded atomically and then reused on the next lookup."""
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=b"%PDF-downloaded")

    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient
    monkeypatch.setattr(
        arxiv_utils.httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=transport, **kwargs),
    )
    target = tmp_path / "resource" / "papers" / "2607.10001.pdf"
    logger = MagicMock()
    client = arxiv_utils.ArxivPdfClient(logger=logger)

    assert await client.download("2607.10001", target) == target
    assert await client.download("2607.10001", target) == target

    assert target.read_bytes() == b"%PDF-downloaded"
    assert [str(request.url) for request in requests] == ["https://arxiv.org/pdf/2607.10001"]
    info_messages = [call.args[0] for call in logger.info.call_args_list]
    assert "download start arxiv_id=2607.10001" in info_messages[0]
    assert "download done arxiv_id=2607.10001" in info_messages[1]
    assert "cache hit arxiv_id=2607.10001" in logger.debug.call_args.args[0]


@pytest.mark.asyncio
async def test_arxiv_pdf_uses_configured_ssh_proxy(tmp_path: Path, monkeypatch):
    """The arXiv HTTP client closes before its temporary SSH proxy stops."""
    monkeypatch.setenv("REME_PROXY_IP", "proxy.example.com")
    monkeypatch.setenv("REME_PROXY_ACCOUNT", "reme-user")
    events: list[str] = []
    client_kwargs: dict = {}
    logger = MagicMock()

    @asynccontextmanager
    async def fake_proxy(*, connect_timeout):
        assert connect_timeout == 12.0
        events.append("proxy-start")
        yield "socks5://127.0.0.1:43123"
        events.append("proxy-stop")

    class FakeResponse:
        """Return one small, valid PDF stream."""

        headers = {"content-length": "12"}

        def raise_for_status(self):
            """Match the successful httpx response interface."""

        async def aiter_bytes(self):
            """Yield the fake PDF body."""
            yield b"%PDF-proxied"

    class FakeStream:
        """Async response context returned by the HTTP client."""

        async def __aenter__(self):
            return FakeResponse()

        async def __aexit__(self, exc_type, exc_value, traceback):
            return None

    class FakeAsyncClient:
        """Capture client construction and cleanup without network access."""

        def __init__(self, **kwargs):
            """Capture HTTPX construction arguments."""
            client_kwargs.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            events.append("client-close")

        def stream(self, method, url):
            """Return the fake streaming response context."""
            assert (method, url) == ("GET", "https://arxiv.org/pdf/2607.10001")
            return FakeStream()

    monkeypatch.setattr(arxiv_utils, "ssh_socks_proxy", fake_proxy)
    monkeypatch.setattr(arxiv_utils.httpx, "AsyncClient", FakeAsyncClient)
    target = tmp_path / "2607.10001.pdf"

    assert await arxiv_utils.ArxivPdfClient(timeout=12.0, logger=logger).download("2607.10001", target) == target

    assert target.read_bytes() == b"%PDF-proxied"
    assert client_kwargs["proxy"] == "socks5://127.0.0.1:43123"
    assert events == ["proxy-start", "client-close", "proxy-stop"]
    info_messages = [call.args[0] for call in logger.info.call_args_list]
    assert any("network mode=ssh_socks destination=reme-user@proxy.example.com" in message for message in info_messages)
    assert any("SSH proxy closed arxiv_id=2607.10001" in message for message in info_messages)


def test_standalone_config_wires_daily_paper_and_memory_jobs(monkeypatch):
    """The standalone config schedules daily paper and wires the memory/search stack."""
    for name in (
        "DINGTALK_APP_KEY",
        "DINGTALK_APP_SECRET",
        "DINGTALK_ROBOT_CODE",
        "DINGTALK_CONVERSATION_IDS",
        "LLM_MODEL_NAME",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "EMBEDDING_MODEL_NAME",
        "EMBEDDING_API_KEY",
        "EMBEDDING_BASE_URL",
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
    assert config["components"]["agent_wrapper"]["claude_code"]["job_tools"] == ["search"]
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
    assert {
        "auto_dream",
        "auto_memory",
        "index_update_loop",
        "reindex",
        "search",
    }.issubset(config["jobs"])
    assert config["jobs"]["auto_memory"]["steps"] == [
        {"backend": "auto_memory_step", "agent_wrapper": "memory"},
    ]
    dream_steps = config["jobs"]["auto_dream"]["steps"]
    assert {
        step["backend"]: step.get("agent_wrapper") for step in dream_steps if step["backend"] != "dream_finish_step"
    } == {
        "dream_extract_step": "memory",
        "dream_integrate_step": "memory",
        "dream_topics_step": "memory",
    }
    assert set(config["components"]["agent_wrapper"]) == {"claude_code", "memory"}
    memory_llm = config["components"]["as_llm"]["memory"]
    assert memory_llm["backend"] == "anthropic"
    assert memory_llm["model"] == "qwen3.7-max"
    assert memory_llm["credential"]["base_url"] == "https://dashscope.aliyuncs.com/apps/anthropic"
    assert config["components"]["agent_wrapper"]["memory"]["as_llm"] == "memory"
    assert config["components"]["agent_wrapper"]["memory"]["builtin_tools"] is False
    assert config["components"]["file_store"]["default"] == {
        "backend": "local",
        "store_name": "local",
        "embedding_store": "default",
        "keyword_index": "default",
        "file_graph": "default",
    }
    assert config["components"]["as_embedding"]["default"]["model"] == "text-embedding-v4"
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


def test_reme_import_does_not_require_optional_dingtalk_stream():
    """Importing ReMe must not eagerly load the core-only DingTalk dependency."""
    script = """
import builtins

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "dingtalk_stream":
        raise ModuleNotFoundError("blocked optional dependency")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import reme
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


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
        resource_dir="external-assets",
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
    note = frontmatter.load(note_path)
    assert note.metadata["arxiv_id"] == "2607.10001"
    assert note.metadata["source_pdf"] == "[[external-assets/papers/2607.10001.pdf]]"
    assert (tmp_path / "external-assets" / "papers" / "2607.10001.pdf").is_file()
    assert "[[daily/2026-07-21/paper-2607.10001.md]]" in digest_path.read_text(
        encoding="utf-8",
    )
    assert not (tmp_path / "metadata" / "daily_paper" / "2026-07-21.json").exists()
    digest = frontmatter.load(digest_path)
    assert digest.metadata["selection_reasoning"] == "Best remaining ranked paper."
    assert digest.metadata["arxiv_ids"] == ["2607.10001"]
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
    assert rerun.response.metadata["selection"] == {
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
    }
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

    dingtalk_stream = importlib.import_module("dingtalk_stream")
    monkeypatch.setattr(dingtalk_stream.DingTalkStreamClient, "get_access_token", get_access_token)
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

    dingtalk_stream = importlib.import_module("dingtalk_stream")
    monkeypatch.setattr(
        dingtalk_stream.DingTalkStreamClient,
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
