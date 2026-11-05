"""Review report recovery without network or provider dependencies."""

# pylint: disable=missing-function-docstring,missing-class-docstring,protected-access,unused-argument

import asyncio
from unittest.mock import AsyncMock

import pytest
from agentscope.agent import ReActConfig
from agentscope.message import AssistantMsg, UserMsg
from agentscope.model import ChatUsage, StructuredResponse
from reme_home.agent.candidate_gate import CandidateGateReport
from reme_home.agent.structured_report import StructuredReportAgent, parse_json_object

from test_workflows import ScriptedModel


@pytest.mark.parametrize(
    "text",
    [
        '{"decision":"reject","reason":"Mechanism unchanged"}',
        '<think>private analysis</think>\n{"decision":"reject","reason":"Mechanism unchanged"}',
        'Conclusion:\n```json\n{"decision":"reject","reason":"Mechanism unchanged",}\n```',
        'Conclusion: {"decision":"reject","reason":"Mechanism unchanged",} Done.',
    ],
)
def test_parse_review_formats(text):
    assert parse_json_object(AssistantMsg(name="review", content=text), report_name="review") == {
        "decision": "reject",
        "reason": "Mechanism unchanged",
    }


@pytest.mark.parametrize(
    "text",
    [
        "invalid",
        "[]",
        '[{"decision":"accept","reason":"Valid schema inside an invalid array"}]',
        'Result: [{"decision":"accept","reason":"Still an array"}]',
        '{"decision":"accept"} {"decision":"reject"}',
        '```json\n{"decision":"accept"}\n```\n{"decision":"reject"}',
        '```json\n{"decision":"accept"}\n```\n```json\n{"decision":"reject"}\n```',
        '```text\nexample\n```\n```json\n{"decision":"accept"}\n```',
        '```python\n{"decision":"accept"}\n```',
        '```json\n{"decision":"accept"}',
    ],
)
def test_invalid_or_ambiguous_reports_require_correction(text):
    with pytest.raises(ValueError):
        parse_json_object(AssistantMsg(name="review", content=text), report_name="review")


@pytest.mark.parametrize("template", ["{}", "Completed.\n```json\n{}\n```"])
def test_text_input_preserves_literal_formatting_inside_json_strings(template):
    text = template.format('{"reason":"literal ```json"}')
    assert parse_json_object(text, report_name="review") == {
        "reason": "literal ```json",
    }


def test_bare_json_preserves_literal_thinking_tags():
    text = '{"reason":"literal <think>evidence</think>"}'
    assert parse_json_object(text, report_name="review") == {"reason": "literal <think>evidence</think>"}


def test_repairs_preserve_strings_and_nested_json():
    text = 'Report: {"reason":"literal ,} and escaped \\"quote\\" {", "items":[{"value":1,},],}'
    assert parse_json_object(AssistantMsg(name="review", content=text), report_name="review") == {
        "reason": 'literal ,} and escaped "quote" {',
        "items": [{"value": 1}],
    }


def make_agent(model):
    return StructuredReportAgent(
        name="review",
        system_prompt="Review committed code.",
        model=model,
        report_model=CandidateGateReport,
        react_config=ReActConfig(max_iters=3),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_usage", [False, True])
@pytest.mark.parametrize(
    "initial",
    ["Reject: mechanism unchanged.", '{"decision":"reject","reason":"x","extra":true}'],
)
async def test_one_correction_retains_history_disables_tools_and_accounts_usage(missing_usage, initial):
    model = ScriptedModel(final=initial)
    model.extra_body = {"enable_thinking": True, "thinking_budget": 100, "host_option": "retained"}
    model.generate_structured_output = AsyncMock(
        return_value=StructuredResponse(
            content={"decision": "reject", "reason": "Mechanism unchanged"},
            usage=None if missing_usage else ChatUsage(input_tokens=7, output_tokens=11, time=0),
        ),
    )
    agent = make_agent(model)
    agent.state.summary = "Earlier investigation summary."
    reply = await agent.reply(UserMsg(name="host", content="Review this candidate."))
    assert CandidateGateReport.model_validate_json(reply.get_text_content()).decision == "reject"
    assert len(model.calls) == 1
    model.generate_structured_output.assert_awaited_once()
    options = model.generate_structured_output.call_args.kwargs
    assert set(options) == {"messages", "structured_model", "extra_body"}
    assert options["structured_model"] is CandidateGateReport
    assert options["extra_body"] == {
        "reasoning_effort": "none",
        "preserve_thinking": False,
        "host_option": "retained",
    }
    text = "\n".join(message.get_text_content() or "" for message in options["messages"])
    for expected in (
        "Review committed code.",
        "Earlier investigation summary.",
        "Review this candidate.",
        initial,
        "Do not inspect more files, call tools",
    ):
        assert expected in text
    assert model.extra_body["enable_thinking"] is True
    assert agent.state.context[-1] is reply
    if missing_usage:
        assert reply.usage is None
    else:
        assert reply.usage.input_tokens == 9 and reply.usage.output_tokens == 14


@pytest.mark.asyncio
async def test_tolerant_parse_avoids_model_correction():
    model = ScriptedModel(final='Decision: {"decision":"accept","reason":"Mechanism changes",}')
    model.generate_structured_output = AsyncMock()
    reply = await make_agent(model).reply(UserMsg(name="host", content="Review."))
    assert CandidateGateReport.model_validate_json(reply.get_text_content()).decision == "accept"
    model.generate_structured_output.assert_not_awaited()
    assert reply.usage.input_tokens == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["invalid", "provider", "cancel"])
async def test_correction_is_not_retried_and_cancellation_propagates(failure):
    model = ScriptedModel(final="invalid")
    model.generate_structured_output = AsyncMock(
        return_value=StructuredResponse(content={"decision": "unknown", "reason": "Invalid"}),
    )
    expected = ValueError
    if failure == "provider":
        model.generate_structured_output.side_effect = RuntimeError("offline")
        expected = RuntimeError
    elif failure == "cancel":
        model.generate_structured_output.side_effect = asyncio.CancelledError()
        expected = asyncio.CancelledError
    with pytest.raises(expected):
        await make_agent(model).reply(UserMsg(name="host", content="Review."))
    model.generate_structured_output.assert_awaited_once()
