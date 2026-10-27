"""Standalone judge Jobs preserve official prompts, parsers and generation defaults."""

# pylint: disable=missing-function-docstring,protected-access
import hashlib
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
async def test_judge_job_uses_only_official_prompt_and_separate_usage(judge):
    name, step, protocol, root = judge
    response = await step.execute()
    assert response.success and response.metadata["llm_judge_score"] == 1
    assert response.metadata["usage"]["cache_input_tokens"] == 30
    kwargs = {"evidence": ["first", "second"]}
    expected = protocol.judge_prompt("QUESTION", "REFERENCE", "PREDICTION", **kwargs)
    assert step.as_llm.call_args.args[0][0].get_text_content() == expected
    assert len(step.as_llm.call_args.args[0]) == 1 and not step.as_llm.call_args.kwargs
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


def test_mobilemem_official_prompt_and_parser_golden_contracts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "plugins/mobilemem-judge/src"))
    protocol = import_module("judge_mobilemem.protocol")
    for mode, digest in [
        ("membase_default", "7880f17ac2335307094dc8b02fa2c0093e1bec7b335215e3362e95a785a0f495"),
        ("native_strict", "8710fcee94ffaa1b40911073b1f6404b62b93d36a4cc8236f3a10fef73622ca7"),
    ]:
        prompt = protocol.judge_prompt(
            "QUESTION $prediction",
            "REFERENCE",
            "PREDICTION $question",
            ["first", "second"],
            judge_protocol=mode,
        )
        assert hashlib.sha256(prompt.encode()).hexdigest() == digest
    for raw, score, default in [('{"label":"CORRECT"}', 1, False), ("CORRECT WRONG", 0, False), ("bad", 0, True)]:
        parsed = protocol.parse_judge(raw)
        assert (parsed["llm_judge_score"], parsed["parse_default_zero"]) == (score, default)
