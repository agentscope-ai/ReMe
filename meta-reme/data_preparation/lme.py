"""Normalize LongMemEval cases into the shared Meta-ReMe contracts."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Iterable, Iterator

from data_preparation.basic import select_case_ids
from models import CaseSpec, QuerySpec, SessionSpec


def iter_cases(source: Path, train_case_ids: Iterable[str] | None) -> Iterator[CaseSpec]:
    """Yield selected LongMemEval question IDs as unified cases."""

    raw_items = json.loads(source.read_text(encoding="utf-8"))
    by_id = {str(item["question_id"]): item for item in raw_items}
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


def load_cases(source: Path, train_case_ids: Iterable[str] | None) -> list[CaseSpec]:
    """Load selected LongMemEval cases into the unified contracts."""

    return list(iter_cases(source, train_case_ids))
