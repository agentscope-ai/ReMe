"""Normalize LongMemEval cases into the shared Meta-ReMe contracts."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from .basic import select_case_ids
from .models import CaseSpec, QuerySpec, SessionSpec


def _load_evidence_sessions(source: Path | None) -> dict[str, dict[str, Any]]:
    """Load optional, independently reviewed evidence-session annotations."""

    if source is None:
        return {}
    annotations: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSONL evidence-session annotation on line {line_number}",
            ) from error
        if not isinstance(item, dict):
            raise ValueError(f"Evidence-session annotation on line {line_number} must be a JSON object")
        case_id = item.get("case_id")
        session_ids = item.get("evidence_sessions")
        is_certain = item.get("is_certain")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Each evidence-session annotation must have a non-empty case_id")
        if case_id in annotations:
            raise ValueError(f"Duplicate evidence-session annotation for case {case_id!r}")
        if not isinstance(session_ids, list) or any(not isinstance(value, str) or not value for value in session_ids):
            raise ValueError(f"Evidence sessions for case {case_id!r} must be a list of non-empty strings")
        if not isinstance(is_certain, bool):
            raise ValueError(f"Evidence certainty for case {case_id!r} must be a boolean")
        annotations[case_id] = {
            "evidence_session_ids": session_ids,
            "evidence_is_certain": is_certain,
        }
    return annotations


def iter_cases(
    source: Path,
    train_case_ids: Iterable[str] | None,
    evidence_source: Path | None = None,
) -> Iterator[CaseSpec]:
    """Yield selected LongMemEval question IDs as unified cases."""

    raw_items = json.loads(source.read_text(encoding="utf-8"))
    by_id = {str(item["question_id"]): item for item in raw_items}
    if len(by_id) != len(raw_items):
        raise ValueError("Duplicate LongMemEval question IDs")
    evidence_by_id = _load_evidence_sessions(evidence_source)
    for case_id in select_case_ids(list(by_id), train_case_ids):
        item = by_id[case_id]
        sessions = []
        for date, session_id, messages in zip(
            item["haystack_dates"],
            item["haystack_session_ids"],
            item["haystack_sessions"],
            strict=True,
        ):
            timestamp = datetime.strptime(date, "%Y/%m/%d (%a) %H:%M").strftime("%Y-%m-%dT%H:%M:%S")
            normalized_messages = [
                {**message, "name": message["role"], "created_at": timestamp} for message in messages
            ]
            sessions.append(
                SessionSpec(session_id=str(session_id), messages=normalized_messages, metadata={"date": date}),
            )
        query_metadata = {
            key: value
            for key, value in item.items()
            if key
            not in {
                "question_id",
                "question",
                "answer",
                "haystack_dates",
                "haystack_session_ids",
                "haystack_sessions",
            }
        }
        if case_id in evidence_by_id:
            annotation = evidence_by_id[case_id]
            known_session_ids = {session.session_id for session in sessions}
            unknown_session_ids = sorted(set(annotation["evidence_session_ids"]) - known_session_ids)
            if unknown_session_ids:
                raise ValueError(
                    f"Evidence annotation for case {case_id!r} references unknown sessions: "
                    f"{', '.join(unknown_session_ids)}",
                )
            query_metadata.update(annotation)
            query_metadata["evidence_annotation_source"] = evidence_source.name
        yield CaseSpec(
            case_id=case_id,
            sessions=sessions,
            queries=[
                QuerySpec(
                    query_id=case_id,
                    question=item["question"],
                    golden_answer=item["answer"],
                    metadata=query_metadata,
                ),
            ],
            metadata={"dataset": "longmemeval"},
        )


def load_cases(
    source: Path,
    train_case_ids: Iterable[str] | None,
    evidence_source: Path | None = None,
) -> list[CaseSpec]:
    """Load selected LongMemEval cases into the unified contracts."""

    return list(iter_cases(source, train_case_ids, evidence_source))
