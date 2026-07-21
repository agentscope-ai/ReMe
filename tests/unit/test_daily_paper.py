"""Focused tests for the daily-paper cookbook workflow."""

import datetime as dt
import json
from pathlib import Path

import frontmatter
import pytest

from reme.components import ApplicationContext
from reme.components.agent_wrapper.base_agent_wrapper import BaseAgentWrapper
from reme.components.runtime_context import RuntimeContext
from reme.config.config_parser import _load_config
from reme.schema import PaperInfo
from reme.steps.cookbook.daily_paper import (
    DailyPaperAnalyzeStep,
    DailyPaperCollectStep,
    DailyPaperDigestStep,
    DailyPaperRankStep,
    DailyPaperSelectStep,
)
from reme.steps.cookbook.daily_paper import analyze, collect
from reme.steps.cookbook.daily_paper.collect import load_historical_arxiv_ids
from reme.steps.cookbook.daily_paper.rank import build_candidate_pool, rrf_score
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

    found = load_historical_arxiv_ids(
        tmp_path,
        dt.date(2026, 7, 21),
        30,
        "daily",
    )

    assert found == {"2607.10001"}


def test_standalone_config_has_backend_split_and_eight_am_cron():
    """The standalone config schedules 08:00 and routes work to AS/CC."""
    config = _load_config("daily_cookbook")

    assert config.get("extends") is None
    assert config["jobs"]["daily_paper_cron"]["cron"] == "0 8 * * *"
    steps = config["jobs"]["daily_paper"]["steps"]
    assert steps[2]["agent_wrapper"] == "default"
    assert steps[3]["agent_wrapper"] == "claude_code"
    assert steps[4]["agent_wrapper"] == "claude_code"
    assert set(config["components"]["agent_wrapper"]) == {"default", "claude_code"}


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

    async def fake_extract(_path: Path, _max_pages: int, _max_chars: int):
        """Return deterministic extracted text."""
        return "--- PAGE 1 ---\nPaper content", 1, False

    monkeypatch.setattr(collect, "HuggingFacePapersClient", _FakeHfClient)
    monkeypatch.setattr(analyze.ArxivPdfClient, "download", fake_download)
    monkeypatch.setattr(analyze, "extract_pdf_text", fake_extract)

    select_wrapper = _QueuedAgentWrapper(
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
        ],
    )
    cc_wrapper = _QueuedAgentWrapper(
        [
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
    await DailyPaperSelectStep(app_context=app_context, agent_wrapper=select_wrapper)(context)
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

    rerun = RuntimeContext(date="2026-07-21")
    await DailyPaperCollectStep(app_context=app_context)(rerun)
    assert rerun.response.metadata["skipped"] is True
    assert _FakeHfClient.requested_daily == ["2026-07-20"]
