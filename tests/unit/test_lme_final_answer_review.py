"""Focused tests for the disputed LongMemEval final-answer workflow."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchmark.longmemeval.run_final_answer_review import (
    atomic_write_results,
    merge_references,
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
            '"is_session_time_wrong":true}',
            '```json\n{"reason":"wrong timestamp verdict","golden_answer_correct":false,'
            '"answer":"still invalid","is_session_time_wrong":false}\n```',
            "补充分析可以放在代码块外。\n"
            '```json\n{"reason":"由 past 和 equal 两个 session 支持。","golden_answer_correct":false,'
            '"answer":"final answer","is_session_time_wrong":true}\n```\n'
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
        "reason": "由 past 和 equal 两个 session 支持。",
        "golden_answer_correct": False,
        "answer": "final answer",
        "is_session_time_wrong": True,
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
    assert '"answer_session_ids_after_question_date": [' in prompt
    assert '"future"' in prompt
    assert "output_schema" not in reply_kwargs
    assert [call.args for call in sleep.await_args_list] == [(0.01,), (0.02,)]


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
        expected_session_time_wrong=False,
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
            expected_session_time_wrong=False,
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


def _read_output(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
