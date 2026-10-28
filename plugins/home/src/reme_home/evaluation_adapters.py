"""Translate normalized Home cases into existing benchmark Docker contracts."""

from datetime import datetime
from typing import Any

from deploy.benchmark_docker import EvaluationQuery, JobRequest

from .data_preparation.models import CaseSpec, QuerySpec, SessionSpec


def session_time(session: SessionSpec) -> str:
    """Match Meta-ReMe's timestamp ordering and date inputs."""
    times = [message.get("created_at") for message in session.messages if message.get("created_at")]
    if not times or not isinstance(times[0], str):
        raise ValueError(f"Session {session.session_id!r} has no timestamp")
    datetime.fromisoformat(times[0])
    return times[0]


def lme_time(value: str) -> str:
    """Convert the source question date to ReMe's query-time format."""
    return datetime.strptime(value, "%Y/%m/%d (%a) %H:%M").strftime("%Y-%m-%dT%H:%M:%S")


def build_jobs(dataset: str, case: CaseSpec) -> tuple[JobRequest, ...]:
    """Build in time order, checkpointing memory after each session's index refresh."""
    sessions = sorted(case.sessions, key=session_time)
    if dataset == "longmemeval":
        dates = [query.metadata.get("question_date") for query in case.queries]
        if any(dates):
            if len(dates) != 1 or not dates[0]:
                raise ValueError("LongMemEval requires one question_date per case")
            cutoff = lme_time(str(dates[0]))
            sessions = [session for session in sessions if session_time(session) <= cutoff]
    jobs = []
    for session in sessions:
        jobs.extend(
            [
                JobRequest(
                    "auto_memory",
                    {
                        "session_id": session.session_id,
                        "messages": session.messages,
                        "date": session_time(session)[:10],
                    },
                ),
                JobRequest("index_update", memory_checkpoint=f"session: {session.session_id}"),
            ],
        )
    if not jobs:
        raise ValueError(f"Case {case.case_id!r} has no sessions to construct")
    return tuple(jobs)


def evaluation_query(dataset: str, query: QuerySpec) -> EvaluationQuery:
    """Only answer inputs cross the sandbox boundary; metadata stays on the host."""
    arguments: dict[str, Any] = {}
    if dataset == "longmemeval" and query.metadata.get("question_date"):
        arguments["query_time"] = lme_time(query.metadata["question_date"])
    return EvaluationQuery(query.query_id, query.question, query.golden_answer, answer_arguments=arguments)


def accepts_build(build: dict, threshold: float) -> bool:
    """Preserve Meta-ReMe's strict-below-threshold partial-construction rule."""
    jobs = build.get("jobs", [])
    failed = sum(not isinstance(job.get("result"), dict) or not job["result"].get("success") for job in jobs)
    return failed / len(jobs) < threshold if failed else build.get("success") is True
