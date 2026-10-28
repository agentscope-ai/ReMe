"""LongMemEval judging retries invalid outputs without treating them as scores."""

# pylint: disable=missing-function-docstring
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from reme.components.runtime_context import RuntimeContext


@pytest.fixture(name="judge")
def fixture_judge(monkeypatch):
    root = Path(__file__).resolve().parents[2] / "plugins/lme-judge/src"
    monkeypatch.syspath_prepend(str(root))
    from judge_lme.llm_judge import LmeAnswerJudgeStep

    step = LmeAnswerJudgeStep()
    step.context = RuntimeContext.from_context(None, query="When?", agent_answer="Today", golden_answer="Today")
    step.agent_wrapper = SimpleNamespace(reply=AsyncMock())
    return step


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,expected", [("YES", "yes"), (" No, incorrect.", "no")])
async def test_valid_judgement_does_not_retry(judge, raw, expected):
    judge.agent_wrapper.reply.return_value = {"result": raw}
    response = await judge.execute()
    assert response.success
    assert response.answer == expected
    assert response.metadata["raw_answer_judgement"] == raw.strip()
    assert judge.agent_wrapper.reply.await_count == 1


@pytest.mark.asyncio
async def test_invalid_output_retries_once_and_recovers(judge):
    judge.agent_wrapper.reply.side_effect = [{"result": "mostly correct"}, {"result": "yes"}]
    response = await judge.execute()
    assert response.success and response.answer == "yes"
    assert response.metadata["raw_answer_judgement"] == "yes"
    calls = judge.agent_wrapper.reply.await_args_list
    assert len(calls) == 2 and calls[0] == calls[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["", "maybe", "yesterday", "correct"])
async def test_two_invalid_outputs_fail_without_a_score(judge, raw):
    judge.agent_wrapper.reply.return_value = {"result": raw}
    with pytest.raises(ValueError, match="invalid yes/no judgement after 2 attempts"):
        await judge.execute()
    assert judge.agent_wrapper.reply.await_count == 2
    assert "answer_judgement" not in judge.context.response.metadata


@pytest.mark.asyncio
async def test_transport_errors_are_not_invalid_output_retries(judge):
    judge.agent_wrapper.reply.side_effect = TimeoutError("provider")
    with pytest.raises(TimeoutError):
        await judge.execute()
    assert judge.agent_wrapper.reply.await_count == 1
