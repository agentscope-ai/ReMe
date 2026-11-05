"""Evidence selection keeps source history, tool pairing, and host budget contracts intact."""

# pylint: disable=protected-access,redefined-outer-name
import asyncio
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from reme.components import ApplicationContext
from reme.components.as_decision_model import BaseAsDecisionModel
from reme.components.job import BaseJob
from reme.enumeration import ComponentEnum
from reme.schema.compaction import CompactionOptions, CompactionRequest
from reme.steps.compaction import CompactContextStep, collect_pairs
from reme.steps.compaction.state import build_batches, estimate_decision_tokens, serialize


def transcript(count=3, output_size=2000):
    """Include same-message pairs and metadata that must survive rebuilding."""
    messages = [{"id": "first", "role": "user", "content": [{"type": "text", "text": "Keep paths exact."}]}]
    for i in range(count):
        messages.append(
            {
                "id": f"m{i}",
                "role": "assistant",
                "metadata": {"i": i},
                "content": [
                    {"type": "text", "text": f"Reading file {i}."},
                    {
                        "type": "tool_call",
                        "id": f"c{i}",
                        "name": "read_file",
                        "input": '{"path":"a.py"}',
                        "state": "finished",
                    },
                    {
                        "type": "tool_result",
                        "id": f"c{i}",
                        "name": "read_file",
                        "output": "x" * output_size,
                        "state": "success",
                        "metadata": {"source": "disk"},
                    },
                ],
            },
        )
    messages.append({"id": "last", "role": "user", "content": [{"type": "text", "text": "Now fix it."}]})
    return messages


class DecisionModel(BaseAsDecisionModel):
    """A provider boundary fake; source and output handling use real Steps."""

    def __init__(self, values=0.9, delay=0, fail=False, usage=None):
        super().__init__()
        self.values, self.delay, self.fail, self.usage = values, delay, fail, usage
        self.calls, self.active, self.peak = [], 0, 0

    async def __call__(self, state, questions):
        self.calls.append((deepcopy(state), questions))
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
            if self.fail:
                raise RuntimeError("provider unavailable")
            return SimpleNamespace(
                content={
                    name: SimpleNamespace(
                        type="binary_answer",
                        probability=self.values[int(name[1:]) - 1] if isinstance(self.values, list) else self.values,
                    )
                    for name in questions
                },
                usage=self.usage,
            )
        finally:
            self.active -= 1


async def compact(messages, model, *, source_refs=None, token_counter=None, **options):
    """Archive refs in fixtures represent an explicit host-provided recovery contract."""
    if source_refs is None:
        source_refs = {f"c{i}": f"archive/run/c{i}" for i in range(len(messages))}
    return await CompactContextStep(as_decision_model=model, token_counter=token_counter)(
        messages=messages,
        source_refs=source_refs,
        eligible_tool_names=["read_file"],
        options={"preserve_recent_messages": 0, **options},
    )


def test_only_clearly_unneeded_results_become_recoverable_previews():
    """Uncertainty keeps the evidence; both ends and tool structure survive replacements."""
    messages = transcript()
    original = deepcopy(messages)
    messages[1]["content"][-1]["output"] = "START" + "x" * 2000 + "END"
    original = deepcopy(messages)
    result = asyncio.run(compact(messages, DecisionModel([0.05, 0.5, 0.9]))).answer
    assert messages == original
    assert [d["action"] for d in result["decisions"]] == ["preview", "keep", "keep"]
    assert result["messages"][2:] == original[2:]
    before, after = original[1]["content"], result["messages"][1]["content"]
    assert before[:2] == after[:2]
    assert after[-1]["metadata"] == before[-1]["metadata"]
    assert after[-1]["output"].startswith("START")
    assert "END" in after[-1]["output"] and "archive/run/c0" in after[-1]["output"]
    assert result["schema_version"] == 2 and result["budget_met"] is None


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "missing", "running", "error", "media", "bad_input", "unknown", "system"],
)
def test_unsafe_pairs_are_never_changed(mutation):
    """Deterministic guards override any semantic score."""
    messages = transcript(1)
    content = messages[1]["content"]
    if mutation == "duplicate":
        content.append(deepcopy(content[-1]))
    elif mutation == "missing":
        content.pop()
    elif mutation == "bad_input":
        content[1]["input"] = "invalid"
    elif mutation == "unknown":
        content[1]["name"] = "write_file"
    elif mutation == "media":
        content[-1]["output"] = [{"type": "data", "data": "image"}]
    elif mutation == "system":
        messages[1]["role"] = "system"
    else:
        content[-1]["state"] = mutation
    model = DecisionModel(0)
    result = asyncio.run(compact(messages, model)).answer
    assert result["messages"] == messages and not model.calls


def test_archive_reference_is_required_and_not_opened():
    """A missing archive cannot be replaced by a fictional recovery notice."""
    model = DecisionModel(0)
    result = asyncio.run(compact(transcript(1), model, source_refs={})).answer
    assert not result["changed"] and not model.calls


def test_cross_message_pair_and_boundary_protection():
    """Protection of either endpoint protects the complete exchange."""
    messages = transcript(1)
    block = messages[1]["content"].pop()
    messages.insert(2, {"id": "result", "role": "assistant", "content": [block]})
    request = CompactionRequest(
        messages=messages,
        source_refs={"c0": "archive/c0"},
        eligible_tool_names=["read_file"],
        options={"preserve_recent_messages": 0},
    )
    assert not collect_pairs(request)[0].protected
    request.protected_message_ids = ["result"]
    assert collect_pairs(request)[0].protected
    request.protected_message_ids = []
    request.options.preserve_recent_messages = 2
    assert collect_pairs(request)[0].protected


def test_batches_carry_their_own_visible_evidence_and_questions():
    """Model observations contain real result snippets, no thinking, and no unrelated old calls."""
    messages = transcript(12)
    messages[1]["content"].append({"type": "thinking", "thinking": "PRIVATE_THOUGHT"})
    request = CompactionRequest(
        messages=messages,
        goal="Fix retries",
        next_action="Review the patch",
        source_refs={f"c{i}": f"a/{i}" for i in range(12)},
        eligible_tool_names=["read_file"],
        options={"preserve_recent_messages": 0, "max_request_tokens": 1100},
    )
    batches = build_batches(request, collect_pairs(request))
    assert len(batches) > 1
    seen = []
    for state, questions in batches:
        assert {u["id"] for u in state["evidence_units"]} == set(questions)
        assert state["next_action"] == "Review the patch"
        assert "PRIVATE_THOUGHT" not in serialize(state)
        assert all("x" * 100 in u["result_preview"] for u in state["evidence_units"])
        assert estimate_decision_tokens({"state": state, "questions": questions}) + 256 <= 1100
        seen.extend(questions)
    assert len(seen) == len(set(seen)) == 12


def test_oversized_evidence_is_kept_without_a_blind_model_call():
    """Insufficient decision budget preserves source, rather than erasing observations."""
    model = DecisionModel(0)
    result = asyncio.run(compact(transcript(1), model, max_state_tokens=1)).answer
    assert not model.calls and not result["changed"]
    assert result["decisions"][0]["reason"] == "unobserved"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True, None])
def test_invalid_probability_keeps_only_the_affected_evidence(value):
    """Invalid answers cannot authorize removal; other valid low scores remain usable."""
    result = asyncio.run(compact(transcript(2), DecisionModel([value, 0]))).answer
    assert result["decisions"][0]["reason"] == "invalid_answer"
    assert [d["action"] for d in result["decisions"]] == ["keep", "preview"]


def test_failed_provider_keeps_original():
    """Provider outages return an unchanged, usable proposal."""
    messages = transcript()
    result = asyncio.run(compact(messages, DecisionModel(fail=True))).answer
    assert result["messages"] == messages
    assert all(d["reason"] == "model_error" for d in result["decisions"])


def test_timeout_and_external_cancellation_drain_requests():
    """Timeout keeps evidence, while explicit cancellation propagates after cleanup."""

    async def run():
        model = DecisionModel(delay=100)
        result = (await compact(transcript(), model, total_timeout_seconds=0.01)).answer
        assert not result["changed"] and model.active == 0
        assert all(d["reason"] == "timeout" for d in result["decisions"])
        task = asyncio.create_task(compact(transcript(), model))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert model.active == 0

    asyncio.run(run())


def test_output_budget_previews_lowest_scores_first_and_stops_when_met():
    """A deterministic stand-in counter verifies budget control, not a vendor token estimate."""
    messages = transcript()
    budget = len(serialize(messages)) - 1000
    result = asyncio.run(
        compact(
            messages,
            DecisionModel([0.08, 0.01, 0.05]),
            token_counter=lambda rows: len(serialize(rows)),
            max_output_tokens=budget,
        ),
    ).answer
    assert [d["action"] for d in result["decisions"]] == ["keep", "preview", "keep"]
    assert result["budget_met"] and result["stats"]["tokens_after"] <= budget


def test_budget_cannot_override_needed_evidence():
    """Protected and uncertain evidence stays intact even if the budget cannot be met."""
    result = asyncio.run(
        compact(transcript(), DecisionModel(0.5), token_counter=lambda rows: len(serialize(rows)), max_output_tokens=1),
    ).answer
    assert result["reason"] == "budget_exceeded" and result["budget_met"] is False
    assert not result["changed"]


def test_already_within_budget_needs_no_model_call():
    """A satisfied host budget avoids inference and preserves the snapshot."""
    model = DecisionModel(0)
    result = asyncio.run(compact(transcript(), model, token_counter=lambda _: 10, max_output_tokens=20)).answer
    assert result["budget_met"] and not result["changed"] and not model.calls


@pytest.mark.parametrize("counter", [None, lambda _: True, lambda _: -1, lambda _: 1.5])
def test_budget_requires_a_valid_host_counter(counter):
    """Reject missing or invalid counters before model inference."""
    model = DecisionModel(0)
    with pytest.raises(ValueError):
        asyncio.run(compact(transcript(), model, token_counter=counter, max_output_tokens=10))
    assert not model.calls


def test_short_outputs_are_not_expanded_by_recovery_markers():
    """Recovery markers must not inflate small outputs."""
    result = asyncio.run(compact(transcript(1, 20), DecisionModel(0))).answer
    assert not result["changed"] and result["decisions"][0]["reason"] == "no_savings"


def test_real_job_registration_and_opt_in(tmp_path):
    """The registered job keeps tools intact unless explicitly eligible."""

    async def run():
        context = ApplicationContext(workspace_dir=str(tmp_path))
        context.components[ComponentEnum.AS_DECISION_MODEL] = {"default": DecisionModel()}
        async with BaseJob(app_context=context, steps=[{"backend": "compact_context_step"}]) as job:
            response = await job(messages=transcript())
            assert response.success and response.answer["reason"] == "no_candidates"

    asyncio.run(run())


@pytest.mark.parametrize(
    "options",
    [
        {"drop_threshold": float("nan")},
        {"max_concurrency": 0},
        {"preview_chars": -1},
        {"max_output_tokens": 0},
        {"surprise": 1},
    ],
)
def test_invalid_options(options):
    """Reject invalid options at the public schema boundary."""
    with pytest.raises(ValidationError):
        CompactionOptions(**options)


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_nonfinite_tool_inputs_do_not_poison_other_candidates(number):
    """Invalid JSON numbers protect that pair without blocking valid candidates."""
    messages = transcript(2)
    messages[1]["content"][1]["input"] = '{"value":' + number + "}"
    result = asyncio.run(compact(messages, DecisionModel(0))).answer
    assert result["messages"][1] == messages[1]
    assert [d["call_id"] for d in result["decisions"]] == ["c1"]


@dataclass
class Usage:
    """Providers can return dataclass usage records."""

    input_tokens: int = 12


@pytest.mark.parametrize("usage", [{"input_tokens": 12}, Usage()])
def test_usage_accepts_json_and_dataclasses(usage):
    """Normalize usage from both supported provider representations."""
    result = asyncio.run(compact(transcript(1), DecisionModel(usage=usage))).answer
    assert result["stats"]["usage"] == [{"input_tokens": 12}]


def test_fallback_goal_and_pinned_context_survive_tool_only_tail():
    """The current task can precede the last assistant/tool-only message."""
    messages = transcript(2)[:-1]
    messages.insert(
        2,
        {"id": "constraint", "role": "user", "content": [{"type": "text", "text": "Preserve receipt IDs"}]},
    )
    request = CompactionRequest(
        messages=messages,
        source_refs={"c0": "a/0", "c1": "a/1"},
        eligible_tool_names=["read_file"],
        protected_message_ids=["constraint"],
        options={"preserve_recent_messages": 0},
    )
    state, _ = build_batches(request, collect_pairs(request))[0]
    assert state["goal"] == "Preserve receipt IDs"
    assert any(entry["text"] == "Preserve receipt IDs" for entry in state["task_context"])


def test_concurrent_batches_isolate_failure_and_respect_limit():
    """One failed batch does not erase successful judgments or remove its own evidence."""

    class PartialModel(DecisionModel):
        """Fail only the request containing the first evidence unit."""

        async def __call__(self, state, questions):
            if "t1" in questions:
                raise RuntimeError("one batch failed")
            return await super().__call__(state, questions)

    async def run():
        model = PartialModel(0, delay=0.01)
        result = (await compact(transcript(8), model, max_request_tokens=1100, max_concurrency=2)).answer
        assert len(model.calls) > 1 and model.peak <= 2
        assert result["decisions"][0]["action"] == "keep"
        assert result["decisions"][0]["reason"] == "model_error"
        assert any(d["action"] == "preview" for d in result["decisions"])

    asyncio.run(run())


def test_previews_must_reduce_host_tokens_as_well_as_characters():
    """A pathological tokenizer can make a shorter recovery marker more expensive."""

    def counter(rows):
        return 1000 if "Compacted output" in serialize(rows) else 100

    result = asyncio.run(compact(transcript(1), DecisionModel(0), token_counter=counter, max_output_tokens=50)).answer
    assert not result["changed"] and result["reason"] == "budget_exceeded"
    assert result["decisions"][0]["reason"] == "no_savings"


def test_rejected_preview_restores_only_its_result():
    """A token-increasing replacement cannot undo an earlier accepted preview or affect later ones."""
    messages = transcript(3)
    original = deepcopy(messages)

    def counter(rows):
        changed = ["Compacted output" in rows[i + 1]["content"][-1]["output"] for i in range(3)]
        if changed[1]:
            return 1000
        return 20 if changed[2] else 80 if changed[0] else 100

    result = asyncio.run(
        compact(messages, DecisionModel([0, 0.01, 0.02]), token_counter=counter, max_output_tokens=30),
    ).answer
    assert result["budget_met"] and result["stats"]["tokens_after"] == 20
    assert [d["action"] for d in result["decisions"]] == ["preview", "keep", "preview"]
    assert result["messages"][2] == original[2] and messages == original


def test_incomplete_provider_response_does_not_publish_partial_judgments():
    """A response failing during normalization retains every result in its batch."""

    class BrokenResponseModel(DecisionModel):
        """A malformed usage object raises after valid binary answers have been read."""

        async def __call__(self, state, questions):
            response = await super().__call__(state, questions)

            class BrokenResponse:
                """Expose valid answers but fail to finish response decoding."""

                content = response.content

                @property
                def usage(self):
                    """Simulate a provider response normalization failure."""
                    raise ValueError("bad response usage")

            return BrokenResponse()

    messages = transcript(2)
    result = asyncio.run(compact(messages, BrokenResponseModel(0))).answer
    assert result["messages"] == messages
    assert all(d["reason"] == "model_error" for d in result["decisions"])
