"""Standalone judge Jobs preserve parsers, usage and generation defaults."""

# pylint: disable=missing-function-docstring,protected-access
from importlib import import_module
from pathlib import Path
from unittest.mock import AsyncMock

from agentscope.message import TextBlock
from agentscope.model import ChatResponse, ChatUsage
import pytest
import yaml

from reme.components.runtime_context import RuntimeContext
from reme.plugin_manifest import parse_plugin_manifest


@pytest.fixture(name="judge")
def fixture_judge(monkeypatch):
    name = "mobilemem"
    root = Path(__file__).resolve().parents[2] / "plugins" / f"{name}-judge" / "src"
    monkeypatch.syspath_prepend(str(root))
    module = import_module(f"judge_{name}.llm_judge")
    prefix = "MobileMem"
    step = getattr(module, f"{prefix}JudgeStep")()
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
            content=[TextBlock(text='{"label":"CORRECT"}')],
            is_last=True,
            usage=ChatUsage(100, 10, 0.01, cache_input_tokens=30),
        ),
    )
    return name, step, import_module(f"judge_{name}.protocol"), root


@pytest.mark.asyncio
async def test_judge_job_preserves_scoring_usage_and_generation_defaults(judge):
    name, step, _, root = judge
    response = await step.execute()
    assert response.success and response.metadata["llm_judge_score"] == 1
    assert response.metadata["usage"]["cache_input_tokens"] == 30
    text = (root / f"judge_{name}" / "plugin.yaml").read_text()
    parse_plugin_manifest(text, plugin_name=f"{name}-judge")
    manifest = yaml.safe_load(text)["application_defaults"]
    assert manifest["jobs"]["answer_judge"]["steps"] == [{"backend": f"{name}_judge_step", "as_llm": "judge"}]
    model = manifest["components"]["as_llm"]["judge"]
    assert model["extra_body"] == {"enable_thinking": False} and model["parameters"]["temperature"] == 0
    assert model["parameters"]["max_tokens"] == 4096
    step.as_llm.return_value = ChatResponse(content=[TextBlock(text="malformed")], is_last=True)
    assert (await step.execute()).metadata["parse_default_zero"] is True
    step.as_llm.side_effect = TimeoutError("provider")
    with pytest.raises(TimeoutError):
        await step.execute()


def test_mobilemem_parser_contracts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "plugins/mobilemem-judge/src"))
    protocol = import_module("judge_mobilemem.protocol")
    for raw, score, default in [('{"label":"CORRECT"}', 1, False), ("CORRECT WRONG", 0, False), ("bad", 0, True)]:
        parsed = protocol.parse_judge(raw)
        assert (parsed["llm_judge_score"], parsed["parse_default_zero"]) == (score, default)
