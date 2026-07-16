"""Focused tests for the disputed LongMemEval final-answer workflow."""

import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchmark.longmemeval import run_final_answer_review as driver_module
from benchmark.longmemeval.run_final_answer_review import (
    REFERENCE_PATHS_ENV,
    atomic_write_results,
    merge_references,
    select_question_ids,
)
from reme.components.agent_wrapper.base_agent_wrapper import BaseAgentWrapper
from reme.components.agent_wrapper.cc_agent_wrapper import CcAgentWrapper
from reme.components.application_context import ApplicationContext
from reme.config import resolve_app_config
from reme.steps.benchmark.lme import final_answer_review as review_module
from reme.steps.benchmark.lme.final_answer_review import FinalAnswerReviewStep


class _FakeAgentWrapper(BaseAgentWrapper):
    """Return queued ordinary text replies and retain every prompt call."""

    def __init__(self, replies: list[str]):
        super().__init__()
        self.replies = list(replies)
        self.calls: list[tuple[str, dict]] = []

    async def reply(self, inputs, **kwargs) -> dict:
        """Return the next queued agent response."""
        self.calls.append((inputs, kwargs))
        return {
            "session_id": f"attempt-{len(self.calls)}",
            "result": self.replies.pop(0),
        }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _session(session_id: str, date: str, marker: str) -> dict:
    return {
        "haystack_session_id": session_id,
        "haystack_date": date,
        "messages": [{"role": "user", "content": marker}],
        "other_session_field": f"full-{marker}",
    }


def test_final_answer_review_keeps_raw_sessions_out_of_prompt_and_retries_plain_json(
    tmp_path,
    monkeypatch,
):
    """Raw session messages stay on disk, and invalid ordinary replies are retried."""
    query = {
        "question_id": "question-1",
        "question": "What happened?",
        "question_type": "single-session-user",
        "question_date": "2024/01/02 (Tue) 10:00",
        "extra_query_field": "keep-me",
    }
    golden = {
        "answer": "old answer",
        "answer_session_ids": ["past", "future"],
        "extra_answer_field": "keep-me-too",
    }
    _write_json(tmp_path / "query.json", query)
    _write_json(tmp_path / "answer.json", golden)
    _write_json(
        tmp_path / "session" / "past.json",
        _session("past", "2024/01/02 (Tue) 09:59", "past-evidence"),
    )
    _write_json(
        tmp_path / "session" / "equal.json",
        _session("equal", "2024/01/02 (Tue) 10:00", "equal-evidence"),
    )
    _write_json(
        tmp_path / "session" / "future.json",
        _session("future", "2024/01/02 (Tue) 10:01", "future-secret"),
    )
    _write_jsonl(
        tmp_path / "first.jsonl",
        [
            {
                "question_id": "question-1",
                "answer": "reference one",
                "reason": "first reason",
            },
        ],
    )
    _write_jsonl(
        tmp_path / "second.jsonl",
        [
            {
                "question_id": "question-1",
                "answer": "reference two",
                "reason": "second reason",
            },
        ],
    )

    wrapper = _FakeAgentWrapper(
        [
            '{"reason":"missing fence","golden_answer_correct":false,"answer":"invalid",'
            '"is_session_time_wrong":false}',
            '```json\n{"reason":"deprecated timestamp verdict","golden_answer_correct":false,'
            '"answer":"still invalid","is_session_time_wrong":true}\n```',
            "补充分析可以放在代码块外。\n"
            '```json\n{"reason":"由 past 和 equal 两个 session 支持 golden answer。",'
            '"golden_answer_correct":true,"answer":"","is_session_time_wrong":false}\n```\n'
            "审核完成。",
        ],
    )
    sleep = AsyncMock()
    monkeypatch.setattr(review_module.asyncio, "sleep", sleep)
    app_context = ApplicationContext(
        workspace_dir=str(tmp_path),
        resource_dir="session",
    )
    step = FinalAnswerReviewStep(
        app_context=app_context,
        agent_wrapper=wrapper,
        reference_paths=["first.jsonl", "second.jsonl"],
        retry_initial_seconds=0.01,
        retry_max_seconds=0.02,
    )

    response = asyncio.run(step())

    assert response.success is True
    assert json.loads(response.answer) == {
        "reason": "由 past 和 equal 两个 session 支持 golden answer。",
        "golden_answer_correct": True,
        "answer": "",
        "is_session_time_wrong": False,
    }
    assert response.metadata["attempts"] == 3
    assert response.metadata["num_sessions"] == 3
    assert response.metadata["num_future_sessions"] == 1
    assert response.metadata["future_sessions"] == [
        {
            "session_id": "future",
            "session_date": "2024/01/02 (Tue) 10:01",
            "session_file": "future.json",
        },
    ]
    assert len(wrapper.calls) == 3
    prompt, reply_kwargs = wrapper.calls[0]
    assert "past-evidence" not in prompt
    assert "equal-evidence" not in prompt
    assert "full-past-evidence" not in prompt
    assert "future-secret" not in prompt
    assert "extra_query_field" in prompt
    assert "extra_answer_field" in prompt
    assert "reference one" in prompt and "reference two" in prompt
    assert '"session_time_check"' in prompt
    assert '"sessions_after_question_date": [' in prompt
    assert '"answer_session_ids_after_question_date"' not in prompt
    assert '"future"' in prompt
    assert "output_schema" not in reply_kwargs
    assert [call.args for call in sleep.await_args_list] == [(0.01,), (0.02,)]


# pylint: disable=protected-access
def test_final_answer_review_reference_paths_env_overrides_config(tmp_path, monkeypatch):
    """The batch driver can pass its selected reference files into the job process."""
    configured = tmp_path / "configured.jsonl"
    selected = tmp_path / "selected.jsonl"
    _write_jsonl(
        configured,
        [{"question_id": "question-1", "answer": "configured", "reason": "configured reason"}],
    )
    _write_jsonl(
        selected,
        [{"question_id": "question-1", "answer": "selected", "reason": "selected reason"}],
    )
    monkeypatch.setenv(REFERENCE_PATHS_ENV, json.dumps([str(selected)]))
    step = FinalAnswerReviewStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        reference_paths=[str(configured)],
    )

    references = step._load_references("question-1")

    assert len(references) == 1
    assert references[0]["answer"] == "selected"
    assert references[0]["source"] == selected.name


def test_final_answer_review_allows_question_without_reference_answer(tmp_path):
    """Samples outside the disputed lists are reviewed from answer.json alone."""
    references_path = tmp_path / "references.jsonl"
    _write_jsonl(
        references_path,
        [{"question_id": "another-question", "answer": "other", "reason": "other reason"}],
    )
    step = FinalAnswerReviewStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        reference_paths=[str(references_path)],
    )

    assert not step._load_references("question-without-reference")


# pylint: enable=protected-access


def test_final_answer_review_agent_cwd_is_sample_session_directory(tmp_path):
    """The configured relative cwd resolves inside each selected LME workspace."""
    config = resolve_app_config(config="jinli_lme", log_config=False)
    agent_config = config["components"]["agent_wrapper"]["lme_final_answer_review"]
    assert agent_config["cwd"] == "session"

    wrapper = CcAgentWrapper(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        cwd=agent_config["cwd"],
    )
    assert wrapper.cwd == tmp_path / "session"


# pylint: disable=protected-access
def test_final_answer_review_requires_empty_answer_when_golden_is_correct():
    """Correct golden answers are collected without duplicating their answer text."""
    parsed = FinalAnswerReviewStep._parse_reply(
        '```json\n{"reason":"golden is supported","golden_answer_correct":true,"answer":"",'
        '"is_session_time_wrong":false}\n```',
    )
    assert parsed == {
        "reason": "golden is supported",
        "golden_answer_correct": True,
        "answer": "",
        "is_session_time_wrong": False,
    }

    with pytest.raises(ValueError, match="answer.*must be empty"):
        FinalAnswerReviewStep._parse_reply(
            '```json\n{"reason":"bad duplicate","golden_answer_correct":true,"answer":"duplicate",'
            '"is_session_time_wrong":false}\n```',
        )

    with pytest.raises(ValueError, match="deprecated and must be false"):
        FinalAnswerReviewStep._parse_reply(
            '```json\n{"reason":"legacy session id verdict",'
            '"golden_answer_correct":false,"answer":"corrected","is_session_time_wrong":true}\n```',
        )

    with pytest.raises(ValueError, match="must not evaluate answer_session_ids"):
        FinalAnswerReviewStep._parse_reply(
            '```json\n{"reason":"answer_session_ids contains a future session",'
            '"golden_answer_correct":false,"answer":"corrected","is_session_time_wrong":false}\n```',
        )


# pylint: enable=protected-access


def test_final_answer_review_rejects_unparseable_session_time_before_agent(tmp_path):
    """An unknown session time is never silently admitted across the time boundary."""
    _write_json(
        tmp_path / "query.json",
        {
            "question_id": "question-1",
            "question": "Q",
            "question_date": "2024/01/02 (Tue) 10:00",
        },
    )
    _write_json(tmp_path / "answer.json", {"answer": "A"})
    _write_json(
        tmp_path / "session" / "bad.json",
        _session("bad", "unknown", "must-not-reach-agent"),
    )
    _write_jsonl(
        tmp_path / "refs.jsonl",
        [{"question_id": "question-1", "answer": "reference", "reason": "reason"}],
    )
    valid_reply = "".join(
        [
            '```json\n{"reason":"y","golden_answer_correct":false,',
            '"answer":"x","is_session_time_wrong":false}\n```',
        ],
    )
    wrapper = _FakeAgentWrapper([valid_reply])
    step = FinalAnswerReviewStep(
        app_context=ApplicationContext(
            workspace_dir=str(tmp_path),
            resource_dir="session",
        ),
        agent_wrapper=wrapper,
        reference_paths=["refs.jsonl"],
    )

    with pytest.raises(ValueError, match="Invalid LongMemEval datetime"):
        asyncio.run(step())
    assert not wrapper.calls


def test_driver_merges_references_and_atomically_rewrites_in_input_order(tmp_path):
    """The batch checkpoint contains one stable row per completed question."""
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_jsonl(
        first,
        [
            {"question_id": "q2", "answer": "a2", "reason": "r2"},
            {"question_id": "q1", "answer": "a1", "reason": "r1"},
        ],
    )
    _write_jsonl(second, [{"question_id": "q1", "answer": "a1b", "reason": "r1b"}])

    merged = merge_references([first, second])

    assert list(merged) == ["q2", "q1"]
    assert len(merged["q2"]) == 1
    assert len(merged["q1"]) == 2
    output = tmp_path / "result.jsonl"
    atomic_write_results(
        output,
        list(merged),
        {
            "q1": {
                "reason": "reason-1",
                "golden_answer_correct": False,
                "answer": "final-1",
                "is_session_time_wrong": False,
            },
            "q2": {
                "reason": "reason-2",
                "golden_answer_correct": False,
                "answer": "final-2",
                "is_session_time_wrong": True,
            },
        },
    )
    rows = _read_output(output)
    assert [row["question_id"] for row in rows] == ["q2", "q1"]
    assert driver_module.load_existing(output)["q2"]["is_session_time_wrong"] is False


def test_driver_selects_all_or_explicit_question_ids(tmp_path):
    """Explicit IDs may select samples that have no reference-answer row."""
    mapping = {
        "q1": tmp_path / "0",
        "q2": tmp_path / "1",
        "q3": tmp_path / "2",
    }

    assert select_question_ids(mapping, None) == ["q1", "q2", "q3"]
    assert select_question_ids(mapping, ["q3", "q1"]) == ["q3", "q1"]
    assert select_question_ids(mapping, None, {"q1", "q3"}) == ["q2"]
    assert select_question_ids(mapping, ["q3", "q2"], {"q3"}) == ["q2"]
    with pytest.raises(ValueError, match="No dataset workspace"):
        select_question_ids(mapping, ["unknown"])
    with pytest.raises(ValueError, match="Duplicate"):
        select_question_ids(mapping, ["q1", "q1"])


def test_driver_limits_concurrency_and_spaces_submissions(tmp_path, monkeypatch):
    """Concurrent jobs never exceed the cap and are not submitted in a burst."""
    mapping = {f"q{index}": tmp_path / str(index) for index in range(4)}
    starts: list[float] = []
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_run_one(question_id, workspace, log_dir, reference_paths):
        del question_id, workspace, log_dir, reference_paths
        nonlocal active, max_active
        with lock:
            starts.append(time.monotonic())
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.055)
        with lock:
            active -= 1
        return {
            "reason": "reviewed",
            "golden_answer_correct": True,
            "answer": "",
            "is_session_time_wrong": False,
        }

    monkeypatch.setattr(driver_module, "workspace_map", lambda: mapping)
    monkeypatch.setattr(driver_module, "merge_references", lambda paths: {})
    monkeypatch.setattr(driver_module, "load_existing", lambda path: {})
    monkeypatch.setattr(driver_module, "atomic_write_results", lambda *args: None)
    monkeypatch.setattr(driver_module, "run_one", fake_run_one)
    monkeypatch.setattr(driver_module, "MIN_SUBMIT_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_final_answer_review.py",
            "--concurrency",
            "3",
            "--submit-interval-seconds",
            "0.02",
            "--output",
            str(tmp_path / "output.jsonl"),
        ],
    )

    assert driver_module.main() == 0
    assert max_active == 3
    assert len(starts) == 4
    assert all(later - earlier >= 0.015 for earlier, later in zip(starts, starts[1:]))


def _read_output(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
