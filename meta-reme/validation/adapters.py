"""Translate normalized Meta-ReMe cases into sandbox requests."""

from __future__ import annotations

from datetime import datetime
import re

from models import CaseSpec, DatasetName, QuerySpec, SessionSpec
from sandbox import EvaluationQuery, JobRequest


class AdapterError(ValueError):
    """Raised when normalized benchmark data cannot form sandbox requests."""


def build_jobs(dataset: DatasetName, case: CaseSpec) -> list[JobRequest]:
    """Construct memory in time order and refresh the index after every session."""

    jobs: list[JobRequest] = []
    sessions = sorted(case.sessions, key=_session_timestamp)
    if dataset is DatasetName.LONGMEMEVAL:
        question_dates = [query.metadata.get("question_date") for query in case.queries]
        if any(question_dates):
            if len(case.queries) != 1 or not question_dates[0]:
                raise AdapterError(f"LongMemEval case {case.case_id!r} requires one question_date")
            query_time = _longmemeval_iso(str(question_dates[0]))
            sessions = [session for session in sessions if _session_timestamp(session) <= query_time]
    for session in sessions:
        jobs.extend(
            [
                JobRequest(
                    "auto_memory",
                    {
                        "session_id": session.session_id,
                        "messages": session.messages,
                        "date": _session_timestamp(session)[:10],
                    },
                ),
                JobRequest("index_update"),
            ],
        )
    if not jobs:
        raise AdapterError(f"case {case.case_id!r} has no sessions to construct")
    return jobs


def _session_timestamp(session: SessionSpec) -> str:
    timestamps = [message.get("created_at") for message in session.messages if message.get("created_at")]
    if not timestamps or not isinstance(timestamps[0], str):
        raise AdapterError(f"session {session.session_id!r} has no created_at timestamp")
    return timestamps[0]


def evaluation_queries(dataset: DatasetName, case: CaseSpec) -> list[EvaluationQuery]:
    """Create dataset-specific answer and judge requests for one case."""

    if dataset is DatasetName.LONGMEMEVAL:
        return [_longmemeval_query(query) for query in case.queries]
    if dataset is DatasetName.BEAM:
        return [_beam_query(query) for query in case.queries]
    raise AdapterError(f"unsupported validation dataset: {dataset}")


def _longmemeval_query(query: QuerySpec) -> EvaluationQuery:
    metadata = query.metadata
    question_type = str(metadata.get("question_type", ""))
    answer_arguments: dict[str, str] = {}
    question_date = metadata.get("question_date")
    if question_date:
        answer_arguments["query_time"] = _longmemeval_iso(str(question_date))
    return EvaluationQuery(
        query_id=query.query_id,
        question=query.question,
        golden_answer=query.golden_answer,
        answer_arguments=answer_arguments,
        judge_arguments={
            "query": query.question,
            "golden_answer": query.golden_answer,
            "question_type": question_type,
        },
    )


def _beam_query(query: QuerySpec) -> EvaluationQuery:
    metadata = query.metadata
    rubric = metadata.get("rubric", query.golden_answer)
    if not isinstance(rubric, list):
        raise AdapterError(f"BEAM query {query.query_id!r} has a non-list rubric")
    question_type = str(metadata.get("question_type", ""))
    return EvaluationQuery(
        query_id=query.query_id,
        question=query.question,
        golden_answer=query.golden_answer,
        judge_answer_argument="llm_response",
        judge_arguments={
            "rubric": rubric,
            "probing_question": query.question,
            "question_type": question_type,
        },
        score_path="metadata.llm_judge_score",
        score_mapping=None,
    )


def _longmemeval_iso(value: str) -> str:
    match = re.match(r"(\d{4}/\d{2}/\d{2})\s+\(\w+\)\s+(\d{2}:\d{2})", value)
    if not match:
        raise AdapterError(f"invalid LongMemEval question_date: {value!r}")
    parsed = datetime.strptime(f"{match.group(1)} {match.group(2)}", "%Y/%m/%d %H:%M")
    return parsed.strftime("%Y-%m-%dT%H:%M:%S")
