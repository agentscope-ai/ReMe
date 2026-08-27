"""Focused tests for the Daily Paper plugin."""

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
import yaml

from reme_daily_paper import (
    AnalyzedPaper,
    DailyPaperMarkdownOutput,
    DailyPaperAnalyzeStep,
    DailyPaperCollectStep,
    DailyPaperDigestStep,
    DailyPaperRankStep,
    DailyPaperSelectStep,
    PaperInfo,
    PaperPick,
    PaperPickList,
)
from reme_daily_paper import analyze, collect
from reme_daily_paper import arxiv as arxiv_utils
from reme_daily_paper import huggingface_papers as hf_utils
from reme_daily_paper.base import (
    normalize_chinese_title,
    now,
    replace_surrogates,
    write_atomic,
    write_markdown,
)
from reme_daily_paper.huggingface_papers import paper_ids_from_html, paper_info_from_payload
from reme_daily_paper.rank import build_candidate_pool, rrf_score
from reme.components import ApplicationContext
from reme.components.agent_wrapper.base_agent_wrapper import BaseAgentWrapper
from reme.components.runtime_context import RuntimeContext
from reme.config import expand_env_vars
from reme.steps.cookbook.dingtalk import DingTalkMarkdownSendStep
from reme.steps.cookbook.dingtalk import send as dingtalk_send

PLUGIN_MANIFEST = yaml.safe_load(
    (Path(__file__).parents[1] / "src" / "reme_daily_paper" / "plugin.yaml").read_text(encoding="utf-8"),
)


def _plugin_config() -> dict:
    """Load application defaults with the same environment expansion as ReMe."""
    return expand_env_vars(PLUGIN_MANIFEST["application_defaults"])


def test_plugin_manifest_declares_complete_runtime_surface():
    """Keep backend registration and both public Jobs inside the distribution."""
    assert set(PLUGIN_MANIFEST["backends"]) == {
        "daily_paper_collect_step",
        "daily_paper_rank_step",
        "daily_paper_select_step",
        "daily_paper_analyze_step",
        "daily_paper_digest_step",
    }
    assert set(_plugin_config()["jobs"]) == {"daily_paper", "daily_paper_cron"}


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


def test_daily_paper_replaces_surrogates_in_text_and_titles():
    """Invalid surrogate code points become visible replacement characters."""
    assert replace_surrogates("before\ud800middle\udfffafter") == "before\ufffdmiddle\ufffdafter"
    assert normalize_chinese_title("论文\ud800标题", "fallback") == "论文\ufffd标题"


@pytest.mark.parametrize("timezone", ["/etc/localtime", "../UTC", "invalid\x00timezone"])
def test_daily_paper_invalid_timezone_falls_back_to_local_time(timezone: str):
    """Invalid IANA timezone keys retain the workflow's local-time fallback."""
    assert now(timezone).tzinfo is None


@pytest.mark.asyncio
async def test_daily_paper_atomic_write_replaces_surrogates(tmp_path: Path):
    """Markdown writes always produce valid UTF-8 even when model output is malformed."""
    target = tmp_path / "note.md"

    await write_atomic(target, "before\ud800after")

    assert target.read_text(encoding="utf-8") == "before\ufffdafter"


@pytest.mark.asyncio
async def test_daily_paper_markdown_write_replaces_surrogates_in_frontmatter(tmp_path: Path):
    """Frontmatter serialization sanitizes nested metadata before YAML encoding."""
    target = tmp_path / "note.md"

    await write_markdown(
        target,
        "body\ud800text",
        {"description": "meta\udffftext", "authors": ["safe", "author\ud800name"]},
    )

    post = frontmatter.load(target)
    assert post.content == "body\ufffdtext"
    assert post.metadata == {
        "description": "meta\ufffdtext",
        "authors": ["safe", "author\ufffdname"],
    }


def test_daily_paper_pdf_extraction_replaces_surrogates(monkeypatch, tmp_path: Path):
    """Malformed PDF text is sanitized before it enters an agent prompt."""

    class FakePage:
        """Return text containing one unpaired surrogate."""

        @staticmethod
        def extract_text():
            """Return the malformed page text."""
            return "before\ud800after"

    class FakeReader:
        """Expose one fake PDF page."""

        def __init__(self, _path: str):
            self.pages = [FakePage()]

    import pypdf

    monkeypatch.setattr(pypdf, "PdfReader", FakeReader)

    content, page_count, truncated = DailyPaperAnalyzeStep._extract_pdf_text_sync(  # pylint: disable=protected-access
        tmp_path / "paper.pdf",
        20,
        300_000,
    )

    assert content == "--- PAGE 1 ---\n\nbefore\ufffdafter"
    assert page_count == 1
    assert truncated is False


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
async def test_hf_client_uses_configured_mirror(monkeypatch):
    """The owned HTTP client uses HF_MIRROR_URL once the mirror is enabled."""
    events: list[str] = []
    client_kwargs: dict = {}
    logger = MagicMock()

    class FakeAsyncClient:
        """Capture construction and close ordering without network access."""

        def __init__(self, **kwargs):
            client_kwargs.update(kwargs)

        async def aclose(self):
            """Record deterministic client cleanup."""
            events.append("client-close")

    monkeypatch.setattr(hf_utils.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setenv("HF_MIRROR_URL", "https://hf-mirror.com/")

    client = hf_utils.HuggingFacePapersClient(
        timeout=12.0,
        logger=logger,
        use_mirror=True,
    )
    assert client.client is None
    async with client:
        assert client_kwargs["base_url"] == "https://hf-mirror.com"
        assert "trust_env" not in client_kwargs
        assert "proxy" not in client_kwargs

    assert events == ["client-close"]
    info_messages = [call.args[0] for call in logger.info.call_args_list]
    assert info_messages == ["[HuggingFacePapersClient] source=mirror"]


def test_hf_client_mirror_switch_controls_source(monkeypatch):
    """The switch alone decides the source, even with HF_MIRROR_URL configured."""
    monkeypatch.setenv("HF_MIRROR_URL", "https://relay.example/hf")

    official = hf_utils.HuggingFacePapersClient(use_mirror=False)
    configured_mirror = hf_utils.HuggingFacePapersClient(use_mirror=True)

    assert official.base_url == "https://huggingface.co"
    assert configured_mirror.base_url == "https://relay.example/hf"


def test_hf_client_uses_default_mirror_when_enabled(monkeypatch):
    """The mirror switch is useful without requiring another setting."""
    monkeypatch.delenv("HF_MIRROR_URL", raising=False)

    client = hf_utils.HuggingFacePapersClient(use_mirror=True)

    assert client.base_url == "https://hf-mirror.com"


@pytest.mark.asyncio
async def test_hf_client_warns_when_mirror_url_is_ignored(monkeypatch):
    """A mirror-only setup learns why traffic still reaches the official site."""
    logger = MagicMock()

    class FakeAsyncClient:
        """Stand in for the owned client without touching the network."""

        def __init__(self, **kwargs):
            pass

        async def aclose(self):
            """Match the owned-client cleanup contract."""

    monkeypatch.setattr(hf_utils.httpx, "AsyncClient", FakeAsyncClient)
    secret_mirror_url = "https://user:password@relay.example/hf?token=secret"
    monkeypatch.setenv("HF_MIRROR_URL", secret_mirror_url)

    async with hf_utils.HuggingFacePapersClient(logger=logger) as client:
        assert client.base_url == "https://huggingface.co"

    warning = logger.warning.call_args.args[0]
    assert "ignoring configured HF_MIRROR_URL" in warning
    assert "use_hf_mirror=true" in warning
    assert secret_mirror_url not in warning
    assert "password" not in warning
    assert "secret" not in warning


@pytest.mark.asyncio
async def test_hf_client_preserves_mirror_path_prefix(monkeypatch):
    """Relative requests retain a path-prefixed HF_MIRROR_URL."""
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[])

    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient
    monkeypatch.setattr(
        hf_utils.httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=transport, **kwargs),
    )
    monkeypatch.setenv("HF_MIRROR_URL", "http://relay.example:18080/hf/")

    async with hf_utils.HuggingFacePapersClient(use_mirror=True) as client:
        assert await client.fetch_daily_ids("2026-07-22") == set()

    assert [str(request.url) for request in requests] == [
        "http://relay.example:18080/hf/api/daily_papers?date=2026-07-22&limit=100",
    ]


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
        async with hf_utils.HuggingFacePapersClient(
            client=raw_client,
            max_retries=2,
            logger=logger,
        ) as client:
            paper_ids = await client.fetch_daily_ids("2026-07-22")
        assert raw_client.is_closed is False

    assert paper_ids == set()
    assert attempts == 2
    sleep.assert_awaited_once_with(0.25)
    warning = logger.warning.call_args.args[0]
    assert "request retry path=/api/daily_papers attempt=1/2" in warning
    assert "error=ConnectTimeout detail=timed out" in warning


def test_rrf_candidate_pool_has_no_topic_preference():
    """The candidate pool follows RRF without reserving slots for a topic."""
    higher = _paper("2607.10001", title="General model", upvotes=100)
    lower = _paper("2607.10002", title="Long-term memory for agents", upvotes=1)
    higher.fused_score = rrf_score(1, None, rrf_k=60, weekly_weight=0.7)
    lower.fused_score = rrf_score(100, None, rrf_k=60, weekly_weight=0.7)

    candidates = build_candidate_pool([lower, higher], limit=1)

    assert candidates == [higher]
    assert higher.fused_score == pytest.approx(1 / 61)


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

    async with arxiv_utils.ArxivPdfClient(logger=logger) as client:
        assert await client.download("2607.10001", target) == target
        assert await client.download("2607.10001", target) == target

    assert target.read_bytes() == b"%PDF-downloaded"
    assert [str(request.url) for request in requests] == [
        "https://arxiv.org/pdf/2607.10001",
    ]
    info_messages = [call.args[0] for call in logger.info.call_args_list]
    assert any("download start arxiv_id=2607.10001" in message for message in info_messages)
    assert any("download done arxiv_id=2607.10001" in message for message in info_messages)
    assert "cache hit arxiv_id=2607.10001" in logger.debug.call_args.args[0]


@pytest.mark.asyncio
async def test_arxiv_pdf_uses_configured_mirror(tmp_path: Path, monkeypatch):
    """The owned arXiv client downloads from ARXIV_MIRROR_URL."""
    events: list[str] = []
    client_kwargs: dict = {}
    logger = MagicMock()

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

        async def aclose(self):
            """Record deterministic client cleanup."""
            events.append("client-close")

        def stream(self, method, url):
            """Return the fake streaming response context."""
            assert (method, url) == ("GET", "https://export.arxiv.org/pdf/2607.10001")
            return FakeStream()

    monkeypatch.setattr(arxiv_utils.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setenv("ARXIV_MIRROR_URL", "https://export.arxiv.org/")
    target = tmp_path / "2607.10001.pdf"

    async with arxiv_utils.ArxivPdfClient(
        timeout=12.0,
        logger=logger,
    ) as client:
        assert await client.download("2607.10001", target) == target

    assert target.read_bytes() == b"%PDF-proxied"
    assert "trust_env" not in client_kwargs
    assert "proxy" not in client_kwargs
    assert events == ["client-close"]
    info_messages = [call.args[0] for call in logger.info.call_args_list]
    assert info_messages[0] == "[ArxivPdfClient] base_url=https://export.arxiv.org"


@pytest.mark.asyncio
async def test_paper_clients_enforce_context_and_unambiguous_ownership(tmp_path: Path):
    """Requests require context entry and injected clients remain caller-owned."""
    with pytest.raises(RuntimeError, match="async context manager"):
        await hf_utils.HuggingFacePapersClient().fetch_daily_ids("2026-07-22")
    with pytest.raises(RuntimeError, match="async context manager"):
        await arxiv_utils.ArxivPdfClient().download(
            "2607.10001",
            tmp_path / "paper.pdf",
        )

    async with httpx.AsyncClient() as raw_client:
        async with arxiv_utils.ArxivPdfClient(client=raw_client):
            pass
        assert raw_client.is_closed is False


@pytest.mark.asyncio
async def test_daily_paper_steps_construct_source_clients_without_proxy(
    tmp_path: Path,
    monkeypatch,
):
    """Collection and analysis construct direct source clients."""
    papers = [_paper(f"2607.1000{index}", title=f"Managed proxy paper {index}") for index in range(1, 4)]
    hf_kwargs: list[dict] = []
    arxiv_kwargs: list[dict] = []

    class FakeHfClient:
        """Return one eligible paper while recording construction."""

        def __init__(self, **kwargs):
            hf_kwargs.append(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def fetch_scope(self, _scope: str, _value: str):
            """Return one ranked paper."""
            return papers

        async def fetch_daily_ids(self, _day: str):
            """Return no yesterday exclusions."""
            return set()

    class FakeArxivClient:
        """Write a fake PDF while recording one shared downloader."""

        def __init__(self, **kwargs):
            arxiv_kwargs.append(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def download(self, _arxiv_id: str, target: Path):
            """Write one minimal PDF fixture."""
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"%PDF-fake")
            return target

    monkeypatch.setattr(collect, "HuggingFacePapersClient", FakeHfClient)
    monkeypatch.setattr(analyze, "ArxivPdfClient", FakeArxivClient)
    monkeypatch.setattr(
        analyze.DailyPaperAnalyzeStep,
        "_extract_pdf_text_sync",
        lambda *_args: ("--- PAGE 1 ---\nPaper content", 1, False),
    )

    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    context = RuntimeContext(date="2026-07-21")
    agent = _QueuedAgentWrapper(
        [
            {
                "title": f"代理论文解读{index}",
                "desc": "Detailed note",
                "body": "# Detailed reading\n\nEvidence [p. 1].",
            }
            for index in range(1, 4)
        ],
    )

    await DailyPaperCollectStep(app_context=app_context)(context)
    context["daily_paper_selected"] = [PaperPick(arxiv_id=paper.arxiv_id, reasoning="Relevant") for paper in papers]
    context["daily_paper_candidates"] = papers
    await DailyPaperAnalyzeStep(app_context=app_context, agent_wrapper=agent)(context)

    assert "proxy_url" not in hf_kwargs[0]
    assert len(arxiv_kwargs) == 1
    assert "proxy_url" not in arxiv_kwargs[0]


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

    step = _plugin_config()["jobs"]["daily_paper"]["steps"][-1]

    assert {key: step[key] for key in ("app_key", "app_secret", "robot_code", "conversation_ids")} == {
        "app_key": "app-key",
        "app_secret": "app-secret",
        "robot_code": "robot-code",
        "conversation_ids": "group-one,group-two",
    }


def test_daily_paper_uses_agentscope_without_tools():
    """Daily Paper uses the shared tool-free agent."""
    config = _plugin_config()
    assert "agent_wrapper" not in config["jobs"]["daily_paper"]["steps"][2]


def test_daily_paper_topics_parameter_defaults_to_empty():
    """Topics are an optional selection preference in the public job schema."""
    topics = _plugin_config()["jobs"]["daily_paper"]["parameters"]["properties"]["topics"]

    assert topics == {
        "type": "string",
        "description": "Optional topics to prioritize when selecting papers.",
        "default": "",
    }


def test_daily_paper_cron_prioritizes_long_term_llm_memory():
    """The scheduled workflow prioritizes papers about long-term LLM memory."""
    assert _plugin_config()["jobs"]["daily_paper_cron"]["topics"] == "大模型长期记忆"


def test_daily_paper_hf_mirror_parameter_defaults_to_disabled():
    """The public job schema exposes an explicit Hugging Face mirror switch."""
    use_hf_mirror = _plugin_config()["jobs"]["daily_paper"]["parameters"]["properties"]["use_hf_mirror"]

    assert use_hf_mirror == {
        "type": "boolean",
        "description": "Use the Hugging Face mirror configured by HF_MIRROR_URL, or hf-mirror.com when unset.",
        "default": False,
    }


def test_daily_paper_cron_hf_mirror_defaults_enabled_with_environment_override(monkeypatch):
    """The scheduled workflow uses the mirror by default and supports an explicit override."""
    monkeypatch.delenv("DAILY_PAPER_USE_HF_MIRROR", raising=False)
    assert _plugin_config()["jobs"]["daily_paper_cron"]["use_hf_mirror"] is True

    monkeypatch.setenv("DAILY_PAPER_USE_HF_MIRROR", "false")
    assert _plugin_config()["jobs"]["daily_paper_cron"]["use_hf_mirror"] is False


def test_paper_pick_list_uses_an_object_root_for_tool_output():
    """AgentScope function arguments require an object-root JSON schema."""
    schema = PaperPickList.model_json_schema()

    assert schema["type"] == "object"
    assert schema["required"] == ["papers"]
    assert schema["properties"]["papers"]["type"] == "array"


def test_daily_paper_selects_three_papers_and_bounds_pdf_context():
    """The public job has no paper-count option and bounds extracted PDF text."""
    job = _plugin_config()["jobs"]["daily_paper"]

    assert "top_k" not in job
    assert "top_k" not in job["parameters"]["properties"]
    assert job["max_pdf_pages"] == 35
    assert job["max_pdf_chars"] == 300_000


@pytest.mark.asyncio
async def test_selection_retries_invalid_id_and_keeps_three_candidates(tmp_path: Path):
    """Selection retries an out-of-pool id and stores three validated candidate ids."""
    candidates = [
        _paper("2607.10001", title="First paper"),
        _paper("2607.10002", title="Second paper"),
        _paper("2607.10003", title="Third paper"),
    ]
    agent = _QueuedAgentWrapper(
        [
            {
                "papers": [
                    {"arxiv_id": "not-a-candidate", "reasoning": "Invalid"},
                    {"arxiv_id": "2607.10002", "reasoning": "Second"},
                    {"arxiv_id": "2607.10003", "reasoning": "Third"},
                ],
            },
            {
                "papers": [
                    {"arxiv_id": " 2607.10001 ", "reasoning": "First"},
                    {"arxiv_id": "2607.10002", "reasoning": "Second"},
                    {"arxiv_id": "2607.10003", "reasoning": "Third"},
                ],
            },
        ],
    )
    context = RuntimeContext(topics="context engineering")
    context["daily_paper_candidates"] = candidates

    response = await DailyPaperSelectStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        agent_wrapper=agent,
    )(context)

    selected = context["daily_paper_selected"]
    assert [item.arxiv_id for item in selected] == [
        "2607.10001",
        "2607.10002",
        "2607.10003",
    ]
    assert [item.reasoning for item in selected] == ["First", "Second", "Third"]
    assert response.answer == "Selected 3 papers with an agent"
    assert len(agent.calls) == 2
    assert "outside the candidate pool" in agent.calls[1]["inputs"]
    assert "用户明确感兴趣的主题：context engineering" in agent.calls[1]["inputs"]
    assert "仅将这些 topics 作为主题偏好" in agent.calls[1]["inputs"]


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
        cwd=Path(__file__).parents[3],
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
        "2607.10004": _paper("2607.10004", title="Second eligible paper", upvotes=70),
        "2607.10005": _paper("2607.10005", title="Third eligible paper", upvotes=60),
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
            return [
                papers["2607.10001"],
                papers["2607.10002"],
                papers["2607.10004"],
                papers["2607.10005"],
            ]

        async def fetch_daily_ids(self, day: str):
            """Record and return the exact requested day."""
            self.requested_daily.append(day)
            return {"2607.10002"}

    async def fake_download(_self, _arxiv_id: str, target: Path):
        """Create a minimal cached-PDF fixture."""
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"%PDF-fake")
        return target

    extraction_limits: list[tuple[int, int]] = []

    def fake_extract(_self, _path: Path, max_pages: int, max_chars: int):
        """Return deterministic extracted text."""
        extraction_limits.append((max_pages, max_chars))
        return "--- PAGE 1 ---\nPaper content", 1, False

    monkeypatch.setattr(collect, "HuggingFacePapersClient", _FakeHfClient)
    monkeypatch.setattr(analyze.ArxivPdfClient, "download", fake_download)
    monkeypatch.setattr(
        analyze.DailyPaperAnalyzeStep,
        "_extract_pdf_text_sync",
        fake_extract,
    )

    cc_wrapper = _QueuedAgentWrapper(
        [
            {
                "papers": [
                    {"arxiv_id": "2607.10001", "reasoning": "Strong result"},
                    {"arxiv_id": "2607.10004", "reasoning": "Useful method"},
                    {"arxiv_id": "2607.10005", "reasoning": "Clear evidence"},
                ],
            },
            {
                "title": "记忆代理研究",
                "desc": "Detailed note\ud800 one",
                "body": "Evidence\udfff one [p. 1].",
            },
            {
                "title": "上下文压缩研究",
                "desc": "Detailed note two",
                "body": "Evidence two [p. 1].",
            },
            {
                "title": "持续学习研究",
                "desc": "Detailed note three",
                "body": "Evidence three [p. 1].",
            },
            {
                "title": "今日智能体论文简报",
                "desc": "Five-minute brief",
                "body": "# 今日一句话\n\nSummary.",
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
        candidate_limit=3,
    )

    await DailyPaperCollectStep(app_context=app_context)(context)
    await DailyPaperRankStep(app_context=app_context)(context)
    await DailyPaperSelectStep(app_context=app_context, agent_wrapper=cc_wrapper)(
        context,
    )
    await DailyPaperAnalyzeStep(app_context=app_context, agent_wrapper=cc_wrapper)(
        context,
    )
    await DailyPaperDigestStep(
        app_context=app_context,
        agent_wrapper=cc_wrapper,
        job_tools=["search", "read"],
    )(context)

    assert _FakeHfClient.requested_daily == ["2026-07-20"]
    assert context.response.metadata["selected_arxiv_ids"] == [
        "2607.10001",
        "2607.10004",
        "2607.10005",
    ]
    assert context.response.metadata["excluded_yesterday_count"] == 1
    assert context.response.metadata["excluded_history_count"] == 1
    note_path = tmp_path / "daily" / "2026-07-21" / "记忆代理研究.md"
    digest_path = tmp_path / "daily" / "2026-07-21" / "今日智能体论文简报.md"
    note = frontmatter.load(note_path)
    assert note.metadata["arxiv_id"] == "2607.10001"
    assert note.metadata["source_pdf"] == "[[external-assets/papers/2607.10001.pdf]]"
    assert (tmp_path / "external-assets" / "papers" / "2607.10001.pdf").is_file()
    assert note.metadata["title"] == "记忆代理研究"
    assert extraction_limits == [(35, 300_000)] * 3
    assert "[[daily/2026-07-21/记忆代理研究.md]]" in digest_path.read_text(
        encoding="utf-8",
    )
    assert not (tmp_path / "metadata" / "daily_paper" / "2026-07-21.json").exists()
    digest = frontmatter.load(digest_path)
    assert digest.metadata["title"] == "今日智能体论文简报"
    assert digest.metadata["selection_reasoning"] == [
        "Strong result",
        "Useful method",
        "Clear evidence",
    ]
    assert digest.metadata["arxiv_ids"] == ["2607.10001", "2607.10004", "2607.10005"]
    assert cc_wrapper.calls[0]["kwargs"] == {"output_schema": PaperPickList}
    assert "用户感兴趣的主题" not in cc_wrapper.calls[0]["inputs"]
    assert "用户未提供明确的 topic 倾向" in cc_wrapper.calls[0]["inputs"]
    assert "优先选择 fused_score 更高的论文" in cc_wrapper.calls[0]["inputs"]
    assert "memory_keyword_score" not in cc_wrapper.calls[0]["inputs"]
    assert "Agent 长期记忆" not in cc_wrapper.calls[0]["inputs"]
    assert all(call["kwargs"] == {"output_schema": DailyPaperMarkdownOutput} for call in cc_wrapper.calls[1:-1])
    assert cc_wrapper.calls[-1]["kwargs"] == {
        "output_schema": DailyPaperMarkdownOutput,
        "job_tools": ["search", "read"],
    }
    assert [call["kwargs"]["output_schema"] for call in cc_wrapper.calls] == [
        PaperPickList,
        DailyPaperMarkdownOutput,
        DailyPaperMarkdownOutput,
        DailyPaperMarkdownOutput,
        DailyPaperMarkdownOutput,
    ]
    analysis_prompt = cc_wrapper.calls[1]["inputs"]
    assert "长期记忆相关性初筛" not in analysis_prompt
    assert "ReMe" not in analysis_prompt
    assert "# PDF 分页文本" in analysis_prompt
    digest_prompt = cc_wrapper.calls[-1]["inputs"]
    assert "Evidence\ufffd one [p. 1]." in digest_prompt
    assert "Detailed note\ufffd one" in digest_prompt
    assert not any("\ud800" <= character <= "\udfff" for character in digest_prompt)
    assert "调用 Read" not in digest_prompt
    assert "daily/2026-07-21" not in digest_prompt
    assert "长期记忆" not in digest_prompt
    assert "先调用 `search` 检索以前的文章" in digest_prompt
    assert "end_date" not in digest_prompt
    assert "Wikilink" in digest_prompt

    rerun = RuntimeContext(date="2026-07-21")
    await DailyPaperCollectStep(app_context=app_context)(rerun)
    assert rerun.response.metadata["skipped"] is True
    assert rerun.get("daily_paper_digest_path") == "daily/2026-07-21/今日智能体论文简报.md"
    assert _FakeHfClient.requested_daily == ["2026-07-20"]


@pytest.mark.asyncio
async def test_digest_force_migrates_old_fixed_filename_to_chinese_title(tmp_path: Path):
    """A forced regeneration replaces the old generated brief without leaving a stale copy."""
    old_path = tmp_path / "daily" / "2026-07-21" / "daily-paper-brief.md"
    old_path.parent.mkdir(parents=True)
    old_path.write_text(
        frontmatter.dumps(frontmatter.Post("old", kind="daily-paper-brief")),
        encoding="utf-8",
    )
    analyses = [
        AnalyzedPaper(
            arxiv_id=f"2607.1000{index}",
            reasoning=f"Reason {index}",
            title=f"论文解读{index}",
            desc=f"Description {index}",
            body=f"Body {index}",
            note_path=f"daily/2026-07-21/论文解读{index}.md",
            pdf_path=f"resource/papers/2607.1000{index}.pdf",
        )
        for index in range(1, 4)
    ]
    context = RuntimeContext()
    context["daily_paper_run_date"] = "2026-07-21"
    context["daily_paper_existing_digest_path"] = "daily/2026-07-21/daily-paper-brief.md"
    context["daily_paper_analyses"] = analyses
    agent = _QueuedAgentWrapper(
        [{"title": "全新论文简报", "desc": "Digest", "body": "Digest body"}],
    )

    await DailyPaperDigestStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        agent_wrapper=agent,
    )(context)

    new_path = tmp_path / "daily" / "2026-07-21" / "全新论文简报.md"
    assert new_path.is_file()
    assert not old_path.exists()
    assert context["daily_paper_digest_path"] == "daily/2026-07-21/全新论文简报.md"


@pytest.mark.asyncio
async def test_digest_validates_model_generated_historical_wikilinks(tmp_path: Path):
    """Only existing daily Markdown from before the run date survives as a model-generated wikilink."""
    valid = tmp_path / "daily" / "2026-07-20" / "历史论文.md"
    valid.parent.mkdir(parents=True)
    valid.write_text("historical", encoding="utf-8")
    today = tmp_path / "daily" / "2026-07-21" / "当日文章.md"
    today.parent.mkdir(parents=True)
    today.write_text("today", encoding="utf-8")
    digest_node = tmp_path / "digest" / "wiki" / "相关概念.md"
    digest_node.parent.mkdir(parents=True)
    digest_node.write_text("digest", encoding="utf-8")
    analyses = [
        AnalyzedPaper(
            arxiv_id=f"2607.1000{index}",
            reasoning=f"Reason {index}",
            title=f"论文解读{index}",
            desc=f"Description {index}",
            body=f"Body {index}",
            note_path=f"daily/2026-07-21/论文解读{index}.md",
            pdf_path=f"resource/papers/2607.1000{index}.pdf",
        )
        for index in range(1, 4)
    ]
    context = RuntimeContext(
        daily_paper_run_date="2026-07-21",
        daily_paper_analyses=analyses,
    )
    agent = _QueuedAgentWrapper(
        [
            {
                "title": "每日简报",
                "desc": "Digest",
                "body": (
                    "保留 [[daily/2026-07-20/历史论文.md|历史论文]]；"
                    "移除 [[daily/2026-07-19/缺失.md|缺失文章]]、"
                    "[[daily/2026-07-21/当日文章.md|当日文章]]、"
                    "[[daily/2026-07-21/每日简报.md|自引用]]、"
                    "[[digest/wiki/相关概念.md|非 daily 节点]] 和 "
                    "[[../outside.md|越界路径]]。"
                ),
            },
        ],
    )

    await DailyPaperDigestStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        agent_wrapper=agent,
    )(context)

    rendered = (tmp_path / "daily" / "2026-07-21" / "每日简报.md").read_text(encoding="utf-8")
    assert "[[daily/2026-07-20/历史论文.md|历史论文]]" in rendered
    assert "缺失文章" in rendered and "daily/2026-07-19/缺失.md" not in rendered
    assert "当日文章" in rendered and "[[daily/2026-07-21/当日文章.md" not in rendered
    assert "自引用" in rendered and "[[daily/2026-07-21/每日简报.md" not in rendered
    assert "非 daily 节点" in rendered and "[[digest/wiki/相关概念.md" not in rendered
    assert "越界路径" in rendered and "../outside.md" not in rendered
    assert "[[daily/2026-07-21/论文解读1.md]]" in rendered


@pytest.mark.asyncio
async def test_dingtalk_markdown_sends_groups_serially_in_configured_order(
    tmp_path: Path,
    monkeypatch,
):
    """The notifier gets one app token and posts once per group in list order."""
    digest_path = tmp_path / "daily" / "2026-07-21" / "daily-paper-brief.md"
    digest_path.parent.mkdir(parents=True)
    digest_path.write_text(
        frontmatter.dumps(
            frontmatter.Post("# 今日论文\n\n测试内容", name="daily-paper-brief"),
        ),
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
        return httpx.Response(
            200,
            json={"processQueryKey": f"query-{len(seen_payloads)}"},
        )

    transport = httpx.MockTransport(handler)
    transport_kwargs: dict = {}

    def ipv4_transport(**kwargs):
        transport_kwargs.update(kwargs)
        return transport

    dingtalk_stream = importlib.import_module("dingtalk_stream")
    monkeypatch.setattr(
        dingtalk_stream.DingTalkStreamClient,
        "get_access_token",
        get_access_token,
    )
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
    assert [payload["openConversationId"] for payload in seen_payloads] == [
        "group-one",
        "group-two",
    ]
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

    response = await DingTalkMarkdownSendStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
    )(context)

    assert response.success is True
    assert response.metadata["dingtalk_configured_count"] == 0
    assert response.metadata["dingtalk_sent_count"] == 0


@pytest.mark.asyncio
async def test_existing_daily_paper_is_reused_and_sent_to_dingtalk(
    tmp_path: Path,
    monkeypatch,
):
    """An idempotent daily-paper run skips generation but still notifies DingTalk."""
    digest_path = tmp_path / "daily" / "2026-07-22" / "daily-paper-brief.md"
    digest_path.parent.mkdir(parents=True)
    digest_path.write_text(
        frontmatter.dumps(
            frontmatter.Post("# 已有日报\n\n复用正文", name="daily-paper-brief"),
        ),
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

    monkeypatch.setattr(
        dingtalk_send.httpx,
        "AsyncHTTPTransport",
        lambda **_kwargs: transport,
    )
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
