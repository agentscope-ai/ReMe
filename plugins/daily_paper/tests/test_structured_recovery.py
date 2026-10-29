"""Offline recovery tests through the actual AgentScope structured-output parser."""

# pylint: disable=protected-access

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from reme_daily_paper.analyze import DailyPaperAnalyzeStep
from reme_daily_paper.schema import PaperInfo, PaperPick
from reme.components import ApplicationContext, RuntimeContext
from reme.components.agent_wrapper import AsAgentWrapper

# The plugin also supports wrappers without the optional AgentScope installation.
ChatModelBase = pytest.importorskip("agentscope.model").ChatModelBase
ChatResponse = pytest.importorskip("agentscope.model").ChatResponse
ChatUsage = pytest.importorskip("agentscope.model").ChatUsage
OpenAIChatFormatter = pytest.importorskip("agentscope.formatter").OpenAIChatFormatter
TextBlock = pytest.importorskip("agentscope.message").TextBlock
ToolCallBlock = pytest.importorskip("agentscope.message").ToolCallBlock


class _Parameters(BaseModel):
    """No provider parameters are needed for this offline model."""


class _TruncatedModel(ChatModelBase):
    """Emit malformed tool arguments until the plugin asks for a shorter note."""

    def __init__(self, recover):
        super().__init__(credential=None, model="offline", parameters=_Parameters(), stream=False)
        self.formatter = OpenAIChatFormatter()
        self.recover = recover
        self.retry_instruction = ""
        self.analysis_calls = 0
        self.structured_calls = 0

    async def _call_api(self, *_args, **kwargs):
        """Replace provider I/O while retaining real agent and parser behavior."""
        messages = kwargs.get("messages") or []
        for message in messages:
            if message.role == "user":
                message.id = "forced-user-id-800"
        tools = kwargs.get("tools") or []
        structured = any(item.get("function", {}).get("name") == "generate_structured_output" for item in tools)
        if structured:
            self.structured_calls += 1
            shortened = any(
                message.role == "user" and self.retry_instruction in message.get_text_content() for message in messages
            )
            raw = json.dumps({"title": "论文摘要", "desc": "summary", "body": "Supported result"})
            if not (self.recover and shortened):
                raw = '{"body":"' + "unfinished note " * 2000
            content = [ToolCallBlock(id="offline-call", name="generate_structured_output", input=raw)]
        else:
            self.analysis_calls += 1
            content = [TextBlock(text="Supported analysis")]
        return ChatResponse(
            content=content,
            is_last=True,
            usage=ChatUsage(input_tokens=10, output_tokens=10, time=0.01),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [True, False])
async def test_shorter_retry_uses_real_structured_parser(tmp_path, monkeypatch, recover):
    """A shorter second request recovers or fails without starting a third attempt."""
    app = ApplicationContext(workspace_dir=str(tmp_path))
    model = _TruncatedModel(recover)
    agent = AsAgentWrapper(app_context=app, as_llm="")
    agent.as_llm = SimpleNamespace(model=model)
    step = DailyPaperAnalyzeStep(app_context=app, agent_wrapper=agent)
    model.retry_instruction = step.prompt_format("analyze_short_retry")
    step.context = RuntimeContext(daily_paper_run_date="2026-07-21")
    monkeypatch.setattr(step, "_extract_pdf_text_sync", lambda *_: ("Supported result [p. 1]", 1, False))
    downloader = SimpleNamespace(download=AsyncMock())
    paper = PaperInfo(arxiv_id="2607.10001")
    pick = PaperPick(arxiv_id=paper.arxiv_id, reasoning="test")
    if recover:
        result = await step._analyze_one(downloader, paper, pick, set())
        assert (tmp_path / result.note_path).is_file()
    else:
        from agentscope.exception import StructuredOutputError

        with pytest.raises(StructuredOutputError):
            await step._analyze_one(downloader, paper, pick, set())
        assert not list(tmp_path.rglob("*.md"))
    assert model.analysis_calls == 2
    assert model.structured_calls >= 2
    assert downloader.download.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [True, False])
async def test_codex_json_parse_failure_gets_one_shorter_retry(tmp_path, monkeypatch, recover):
    """Use the real Codex reply parser with only SDK/lifecycle boundaries replaced."""
    from reme.components.agent_wrapper import CodexAgentWrapper

    app = ApplicationContext(workspace_dir=str(tmp_path))
    wrapper = CodexAgentWrapper(app_context=app)
    valid = json.dumps({"title": "论文摘要", "desc": "summary", "body": "Supported result"})
    thread = SimpleNamespace(
        id="offline-thread",
        run=AsyncMock(
            side_effect=[
                SimpleNamespace(final_response='{"body":"unfinished', usage=None),
                SimpleNamespace(final_response=valid if recover else '{"body":"unfinished', usage=None),
            ],
        ),
    )
    monkeypatch.setattr(wrapper, "start", AsyncMock())
    monkeypatch.setattr(wrapper, "_get_codex", AsyncMock(return_value=object()))
    monkeypatch.setattr(wrapper, "_open_thread", AsyncMock(return_value=thread))
    monkeypatch.setattr(wrapper, "_release_thread", AsyncMock())
    monkeypatch.setattr(wrapper, "_serialize", lambda _: {})
    step = DailyPaperAnalyzeStep(app_context=app, agent_wrapper=wrapper)
    step.context = RuntimeContext(daily_paper_run_date="2026-07-21")
    monkeypatch.setattr(step, "_extract_pdf_text_sync", lambda *_: ("Evidence [p. 1]", 1, False))
    downloader = SimpleNamespace(download=AsyncMock())
    paper = PaperInfo(arxiv_id="2607.10001")
    pick = PaperPick(arxiv_id=paper.arxiv_id, reasoning="test")
    if recover:
        result = await step._analyze_one(downloader, paper, pick, set())
        assert (tmp_path / result.note_path).is_file()
    else:
        with pytest.raises(ValueError, match="Codex returned invalid JSON"):
            await step._analyze_one(downloader, paper, pick, set())
    assert thread.run.await_count == 2
    instruction = step.prompt_format("analyze_short_retry")
    assert instruction not in thread.run.await_args_list[0].args[0]
    assert instruction in thread.run.await_args_list[1].args[0]
    downloader.download.assert_awaited_once()
