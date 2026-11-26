"""Standalone judge Jobs preserve parsers, usage and generation defaults."""

# pylint: disable=missing-function-docstring,protected-access
from pathlib import Path
from unittest.mock import AsyncMock

from agentscope.message import TextBlock
from agentscope.model import ChatResponse, ChatUsage
import pytest
import yaml

from reme.core.context import RuntimeContext
from reme.plugin.manifest import parse_plugin_manifest


@pytest.fixture(name="judge")
def fixture_judge(monkeypatch):
    root = Path(__file__).resolve().parents[2] / "plugins/memlens-judge/src"
    monkeypatch.syspath_prepend(str(root))
    from judge_memlens import protocol
    from judge_memlens.llm_judge import MemLensJudgeStep

    step = MemLensJudgeStep()
    step.context = RuntimeContext.from_context(
        None,
        question="QUESTION",
        reference="REFERENCE",
        prediction="PREDICTION",
        evidence=["first", "second"],
        category="information_extraction",
    )
    step.as_llm = AsyncMock(
        return_value=ChatResponse(
            content=[TextBlock(text='{"answer_score":1}')],
            is_last=True,
            usage=ChatUsage(100, 10, 0.01, cache_input_tokens=30),
        ),
    )
    return step, protocol, root


@pytest.mark.asyncio
async def test_judge_job_preserves_scoring_usage_and_generation_defaults(judge):
    step, _, root = judge
    response = await step.execute()
    assert response.success and response.metadata["llm_judge_score"] == 1
    assert response.metadata["usage"]["cache_input_tokens"] == 30
    text = (root / "judge_memlens/plugin.yaml").read_text()
    parse_plugin_manifest(text, plugin_name="memlens-judge")
    manifest = yaml.safe_load(text)["application_defaults"]
    assert manifest["jobs"]["answer_judge"]["steps"] == [{"backend": "memlens_judge_step", "as_llm": "judge"}]
    model = manifest["components"]["as_llm"]["judge"]
    assert model["extra_body"] == {"enable_thinking": False} and model["parameters"]["temperature"] == 0
    assert model["parameters"]["max_tokens"] == 256
    step.as_llm.return_value = ChatResponse(content=[TextBlock(text="malformed")], is_last=True)
    assert (await step.execute()).metadata["parse_default_zero"] is True
    step.context.data["prediction"] = "word " * 501
    assert (await step.execute()).metadata["auto_zero"] is True
    assert step.as_llm.await_count == 2
    step.context.data["prediction"] = "PREDICTION"
    step.as_llm.side_effect = TimeoutError("provider")
    with pytest.raises(TimeoutError):
        await step.execute()


def test_memlens_task_routing_and_parser(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "plugins/memlens-judge/src"))
    from judge_memlens import protocol

    assert protocol.task_key("multi_session_reasoning", "entity_resolution", "2") == "MSR_Counting"
    assert protocol.parse_judge_response('{"answer_score":0} [Score]:1')["acc"] == 0
