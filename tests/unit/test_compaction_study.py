"""Rule grading, source provenance and equal-task aggregation for the fixture study."""

import hashlib
import json
from argparse import Namespace

import pytest

from benchmark.compaction.cases import event_units, write
from benchmark.compaction.suite_native import prepare_native, validate_base
from benchmark.compaction.suite_fusion import prepare
from benchmark.compaction.suite import context, define, exam, grade
from benchmark.compaction.suite_run import ARMS, load_cases, metrics


def sample():
    """Define independent expected values, including strict nested types."""
    return define(
        "example",
        "Read the project files.",
        {"source.txt": "evidence"},
        {"number": 7, "flag": True, "mapping": {"a": 2}, "ordered": ["a", "b"]},
        {"number": "integer value", "flag": "boolean state", "mapping": "mapping", "ordered": "ordered values"},
        "fixture/example",
    )


def test_followup_never_includes_expected_answers():
    """Only the requested field labels and descriptions enter the evaluation prompt."""
    case = sample()
    case["gold"] = {"hidden": "PRIVATE_GOLD_8391"}
    assert "PRIVATE_GOLD" not in exam(case)
    assert "integer value" in exam(case)


@pytest.mark.parametrize("value", [True, 7.0, "7", None])
def test_rules_do_not_confuse_numeric_types(value):
    """No model judge or JSON coercion can turn a wrong type into a correct fact."""
    case = sample()
    assert not grade(case, {"facts": {**case["gold"], "number": value}})["number"]


def test_mapping_shape_and_array_order_are_part_of_the_contract():
    """Require an object for named test counts and preserve meaningful list ordering."""
    case = sample()
    assert all(grade(case, {"facts": case["gold"]}).values())
    result = grade(case, {"facts": {**case["gold"], "mapping": [2], "ordered": ["b", "a"]}})
    assert not result["mapping"] and not result["ordered"]
    assert not any(grade(case, []).values())


def test_format_clarifications_are_case_specific_and_do_not_reveal_counts():
    """Clarify ambiguous output shapes without changing the oracle or unaffected questions."""
    case = sample()
    payment = {**case, "id": "payment"}
    notes = {**case, "id": "meeting_notes"}
    assert "must be a JSON object" in exam(payment)
    assert "timeout_then_success" in exam(payment)
    assert "leading list marker" in exam(notes)
    assert "must be a JSON object" not in exam(case)


def test_matching_line_views_retain_original_line_numbers_and_context():
    """The same source query selects a deterministic view without answer-based trimming."""
    text = "before\nERROR exact-id\nafter\nlast\n"
    assert context(text, r"ERROR", radius=1) == "1|before\n2|ERROR exact-id\n3|after\n"
    assert context(text, r"absent", radius=1) == ""


def test_denied_reads_remain_lossless_observations_in_the_trajectory():
    """A recovered failed read cannot crash evidence extraction or disappear from history."""
    text = json.dumps({"error": "Only listed source files may be read"})
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "c", "function": {"arguments": '{"path":"missing"}'}}]},
        {"role": "tool", "tool_call_id": "c", "content": text},
    ]
    units = event_units(messages)
    assert units[0]["text"] == text
    assert units[0]["spans"][0]["text"] == text


def test_task_weights_and_failed_conditions_have_fixed_denominators():
    """A task with many checked fields cannot outweigh a task with only one."""
    rows = []
    for arm in ARMS:
        for task, checks in [("a", dict.fromkeys(["x", "y", "z"], True)), ("b", {"x": False})]:
            rows.append(
                {
                    "case": task,
                    "arm": arm,
                    "valid": True,
                    "trials": [{"checks": checks, "usage": {"prompt_tokens": 10}}],
                },
            )
    assert metrics(rows, ["a", "b"], 1)[0]["facts"] == 0.5
    rows[0].update(valid=False, trials=[])
    assert metrics(rows, ["a", "b"], 1)[0]["facts"] == 0


def test_corpus_loader_excludes_final_reports_and_checks_hashes(tmp_path):
    """Internal DB metadata and the completed answer cannot leak into scoring history."""
    task = tmp_path / "example"
    work = task / "workspace"
    work.mkdir(parents=True)
    (work / "source.txt").write_text("evidence")
    source = {
        "messages": [
            {"role": "system", "content": "real system"},
            {"role": "user", "content": "task", "_db_row_snapshot": {"content": "PRIVATE_COPY"}},
            {"role": "assistant", "content": "FINAL_REPORT"},
        ],
    }
    write(task / "trajectory.full.json", source)
    source_bytes = (task / "trajectory.full.json").read_bytes()
    write(task / "collection.json", {"completed": True, "trajectory_sha256": hashlib.sha256(source_bytes).hexdigest()})
    case = {
        **sample(),
        "source_hashes": {"source.txt": hashlib.sha256(b"evidence").hexdigest()},
        "control_reason": None,
    }
    write(tmp_path / "corpus.json", {"cases": [case]})
    _, cases = load_cases(tmp_path / "corpus.json")
    assert len(cases["example"]["messages"]) == 2
    assert "PRIVATE_COPY" not in json.dumps(cases)
    assert "FINAL_REPORT" not in json.dumps(cases)
    (work / "source.txt").write_text("changed")
    with pytest.raises(ValueError, match="Changed source"):
        load_cases(tmp_path / "corpus.json")


def test_native_rejects_incompatible_or_duplicate_base_panels(tmp_path):
    """Paid native calls must not silently combine different experiment protocols."""
    corpus = tmp_path / "corpus.json"
    corpus.write_text("frozen")
    cases = [str(i) for i in range(14)]
    rows = [{"case": key, "arm": arm, "repeat": i} for key in cases for arm in ARMS for i in range(3)]
    protocol = {
        "model": "qwen3.8-max",
        "compressions": 3,
        "answer_repeats": 2,
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
    }
    write(tmp_path / "protocol.json", protocol)
    write(tmp_path / "results.json", rows)
    assert validate_base(tmp_path, corpus, cases) == rows
    write(tmp_path / "results.json", rows + [rows[0]])
    with pytest.raises(ValueError, match="exactly once"):
        validate_base(tmp_path, corpus, cases)
    write(tmp_path / "results.json", rows)
    protocol["model"] = "different-model"
    write(tmp_path / "protocol.json", protocol)
    with pytest.raises(ValueError, match="matching fourteen-task"):
        validate_base(tmp_path, corpus, cases)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change,error",
    [
        ("none", None),
        ("hash", "source mismatch"),
        ("missing_validation", "incomplete"),
        ("missing_hash", "source mismatch"),
        ("missing_summaries", "incomplete"),
        ("duplicate_repeat", "incomplete"),
        ("exam_implicit", "source mismatch"),
        ("exam_explicit", None),
        ("messages", "evidence changed"),
        ("gold", "evidence changed"),
        ("units", "evidence changed"),
    ],
)
async def test_native_cache_provenance_before_provider_calls(tmp_path, monkeypatch, change, error):
    """Reject stale caches without generating or answering, except explicit exam-only amendments."""
    evaluation = tmp_path / "evaluation"
    source = evaluation / "example" / "source.json"
    source.parent.mkdir(parents=True)
    original = {
        "messages": [{"role": "user", "content": "original task"}],
        "exam": "original exam",
        "gold": {"answer": 42},
        "units": [{"id": "u", "text": "original evidence"}],
    }
    write(source, original)
    generation = tmp_path / "native" / "example" / "generation"
    generation.mkdir(parents=True)
    validation_path = generation / "validation.json"
    summaries_path = generation / "summaries.json"
    write(validation_path, {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    write(summaries_path, [{"repeat": i, "summary": f"frozen {i}"} for i in range(3)])
    args = Namespace(native=tmp_path / "native", evaluation=evaluation, native_evaluation=None)
    if change == "hash":
        write(validation_path, {"source_sha256": "different-source"})
    elif change == "missing_validation":
        validation_path.unlink()
    elif change == "missing_hash":
        write(validation_path, {})
    elif change == "missing_summaries":
        summaries_path.unlink()
    elif change == "duplicate_repeat":
        write(summaries_path, [{"repeat": 0}] * 3)
    elif change in ("exam_implicit", "exam_explicit", "messages", "gold", "units"):
        native_source = tmp_path / "original" / "example" / "source.json"
        native_source.parent.mkdir(parents=True)
        write(native_source, original)
        changed = {**original, "exam": "amended exam"}
        if change in ("messages", "gold", "units"):
            changed[change] = []
        write(source, changed)
        if change != "exam_implicit":
            args.native_evaluation = tmp_path / "original"

    def no_provider_calls(*_args, **_kwargs):
        pytest.fail("Invalid cached sources must be rejected before provider calls")

    monkeypatch.setattr("benchmark.compaction.suite_native.asyncio.create_subprocess_exec", no_provider_calls)
    monkeypatch.setattr("benchmark.compaction.suite_native.AsyncOpenAI", no_provider_calls)
    if error:
        with pytest.raises(ValueError, match=error):
            await prepare_native(args, ["example"], tmp_path / "repo")
    else:
        await prepare_native(args, ["example"], tmp_path / "repo")


def test_hermes_fusion_reuses_summary_and_verified_jev_prefix(tmp_path, monkeypatch):
    """Appending evidence cannot rewrite the frozen native summary or rerank units."""
    case = {
        "messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}],
        "exam": "held-out",
        "gold": {"answer": 42},
    }
    units = [{"id": "u", "file": "source", "text": "evidence", "spans": []}]
    base = []
    native = []
    for i in range(3):
        for arm in ARMS:
            base.append(
                {
                    "case": "example",
                    "arm": arm,
                    "repeat": i,
                    "summary": "S",
                    "selected": ["u"] if arm.startswith("fusion") else [],
                    "input_cap": 1000,
                    "trials": [{"usage": {"prompt_tokens": 200}}],
                },
            )
        native.append(
            {
                "case": "example",
                "arm": "hermes_summary",
                "repeat": i,
                "summary": f"frozen H {i}",
                "valid": True,
            },
        )
    evaluation = tmp_path / "evaluation"
    generation = tmp_path / "native" / "example" / "generation"
    source = evaluation / "example" / "source.json"
    source.parent.mkdir(parents=True)
    generation.mkdir(parents=True)
    write(source, {**case, "units": units})
    for i in range(3):
        write(source.parent / f"paired.c{i}.decision0.response.json", {"scores": {"u": 0.9}})
    write(generation / "validation.json", {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    write(generation / "summaries.json", [{"repeat": i, "summary": f"frozen H {i}", "valid": True} for i in range(3)])
    combined = tmp_path / "combined"
    combined.mkdir()
    write(combined / "results.json", base + native)
    monkeypatch.setattr("benchmark.compaction.suite_fusion.validate_base", lambda *_: base)
    args = Namespace(
        evaluation=evaluation,
        native=tmp_path / "native",
        combined=combined,
        corpus=tmp_path / "corpus.json",
        native_evaluation=None,
    )
    reused, prepared = prepare(args, {"example": case})
    assert reused == base + native and len(prepared) == 15
    for row in prepared:
        assert row["summary"] == f"frozen H {row['repeat']}"
        assert row["selected"] == ["u"] and row["actual_n"] == 1
        assert row["context"][-1]["content"] == "held-out"
        assert "evidence" in row["context"][-2]["content"]
    write(source.parent / "paired.c0.decision0.response.json", {"scores": {"wrong": 0.9}})
    with pytest.raises(ValueError, match="every unit needs a score"):
        prepare(args, {"example": case})
