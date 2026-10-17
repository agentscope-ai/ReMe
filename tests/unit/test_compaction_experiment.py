"""Whole-source selection, complete-input budgets and compressor data boundaries."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from benchmark.compaction.cases import event_units, render
from benchmark.compaction.run import Experiment, compose, decision_input, messages, top_n


class CharacterTokenizer:
    """Count characters so budget boundaries require no external tokenizer assets."""

    @staticmethod
    def encode(text, **_kwargs):
        """Match the tokenizer's small counting interface."""
        return SimpleNamespace(ids=list(text))


def experiment(tmp_path, cap=10000):
    """Use real packing with no live model clients."""
    return Experiment(None, None, tmp_path, CharacterTokenizer(), asyncio.Semaphore(1), "mock", cap)


def fixture():
    """Keep the oracle/exam separate from the task and evidence sent to compressors."""
    case = {
        "messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}],
        "exam": "PRIVATE_EXAM",
        "gold": {"value": "PRIVATE_GOLD"},
    }
    units = [
        {"id": str(i), "file": "file.txt", "text": f"source-{i}", "spans": [{"start": i, "end": i + 1}]}
        for i in range(3)
    ]
    return case, units


def test_top_n_is_strict_nested_score_prefix_with_source_order():
    """Retain even low-ranked evidence; do not threshold, skip or reorder it."""
    _, units = fixture()
    scores = {"0": 0.01, "1": 0.8, "2": 0.9}
    assert top_n(units, scores, 1) == units[2:]
    assert top_n(units, scores, 2) == units[1:]
    assert top_n(units, scores, 3) == units
    assert top_n(units, dict.fromkeys(scores, 0.1), 1) == units[:1]
    with pytest.raises(ValueError):
        top_n(units, scores, 4)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, -0.1, 1.1])
def test_invalid_rankings_never_become_top_n(value):
    """Malformed probabilities must not silently determine retained evidence."""
    _, units = fixture()
    with pytest.raises(ValueError):
        top_n(units, {"0": value, "1": 0.5, "2": 0.5}, 1)


def test_grouping_preserves_every_source_character_and_correlated_events():
    """Evidence grouping uses source correlations, never expected payment identifiers."""
    text = "".join(f"healthcheck {i}\n" for i in range(100))
    text += "event request_id=trace-a step=1\nevent request_id=trace-a step=2\n"
    history = [
        {"role": "assistant", "tool_calls": [{"id": "c", "function": {"arguments": '{"path":"log"}'}}]},
        {"role": "tool", "tool_call_id": "c", "content": json.dumps({"content": text})},
    ]
    units = event_units(history)
    spans = sorted([s for u in units for s in u["spans"]], key=lambda s: s["start"])
    assert "".join(s["text"] for s in spans) == text
    assert all(text[s["start"] : s["end"]] == s["text"] for s in spans)
    assert any("step=1" in u["text"] and "step=2" in u["text"] for u in units)


def test_compressors_see_task_and_source_but_no_exam_or_gold(tmp_path):
    """Usefulness scoring and summary generation cannot consult the follow-up oracle."""
    captured = []

    class FakeExperiment(Experiment):
        """Capture the real summary request without network calls."""

        async def ask(self, _label, context, *, summary=False):
            """Return a bounded summary."""
            captured.append(context)
            return "S", {"prompt_tokens": 10}

    case, units = fixture()
    state, _ = decision_input(case["messages"][1]["content"], units)
    exp = FakeExperiment(None, None, tmp_path, CharacterTokenizer(), asyncio.Semaphore(1), "mock", 10000)
    asyncio.run(exp.summarize("summary", "task", units, 100))
    assert "PRIVATE_" not in json.dumps([state, captured])
    supplied = json.loads(captured[0][1]["content"])
    assert supplied["remaining_evidence"] == units and supplied["retained_evidence"] == []


def test_frozen_s_is_a_literal_prefix_of_every_fusion_context():
    """Adding original evidence must preserve the exact baseline summary and protected prompts."""
    case, units = fixture()
    baseline = messages(case, compose("Fixed S", ""))
    for n in range(1, 4):
        chosen = top_n(units, {"0": 0.1, "1": 0.8, "2": 0.9}, n)
        context = messages(case, compose("Fixed S", render(chosen)))
        assert context[2]["content"].startswith(baseline[2]["content"])
        assert context[:2] == baseline[:2] and context[-1] == baseline[-1]


def test_complete_input_budget_counts_protected_prompts_and_followup(tmp_path):
    """The last token is shared by summary, original evidence and all input framing."""
    case, _ = fixture()
    exp = experiment(tmp_path)
    payload = compose("S", "raw evidence")
    exp.cap = exp.input_tokens(messages(case, payload)) + 64
    assert exp.fits(case, payload)
    assert not exp.fits(case, payload + "x")
