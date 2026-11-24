"""Translate normalized Home cases into existing benchmark Docker contracts."""

from datetime import datetime

from ..preparation.data.basic import sessions_through_cutoff
from ..preparation.data.models import CaseSpec, QuerySpec, SessionSpec
from .artifacts import construction_exception, failure_kind
from .docker import EvaluationQuery, JobRequest


def session_time(session: SessionSpec) -> str:
    """Return the first available message timestamp for session ordering and date inputs."""
    timestamp = session.first_timestamp
    if not isinstance(timestamp, str):
        raise ValueError(f"Session {session.session_id!r} has no timestamp")
    datetime.fromisoformat(timestamp)
    return timestamp


def build_jobs(case: CaseSpec) -> tuple[JobRequest, ...]:
    """Build in time order, checkpointing memory after each session's index refresh."""
    sessions = sessions_through_cutoff(sorted(case.sessions, key=session_time), case.construction_cutoff)
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


def evaluation_query(query: QuerySpec) -> EvaluationQuery:
    """Only answer inputs cross the sandbox boundary; metadata stays on the host."""
    arguments = {} if query.query_time is None else {"query_time": query.query_time}
    return EvaluationQuery(query.query_id, query.question, None, answer_arguments=arguments)


def accepts_build(build: dict, threshold: float) -> bool:
    """Accept partial builds below the failure threshold if no infrastructure error occurred."""
    jobs = build.get("jobs", [])
    failed = sum(not isinstance(job.get("result"), dict) or not job["result"].get("success") for job in jobs)
    if failed and failure_kind(construction_exception(build), "construction") == "infra_error":
        return False
    return failed / len(jobs) < threshold if failed else build.get("success") is True
