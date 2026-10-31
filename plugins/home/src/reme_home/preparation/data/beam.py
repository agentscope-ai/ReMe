"""Normalize BEAM cases into the shared Meta-ReMe contracts."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from .basic import select_case_ids
from .models import CaseSpec, QuerySpec, SessionSpec

VARIANTS = ("100K", "500K", "1M", "10M")


def _parse_date(value: str) -> datetime:
    for date_format in ("%B-%d-%Y", "%b-%d-%Y"):
        try:
            return datetime.strptime(value, date_format)
        except ValueError:
            continue
    raise ValueError(f"Invalid BEAM time anchor: {value!r}")


def _sessions(raw_batches: list[dict[str, Any]], variant: str, case_id: str) -> list[SessionSpec]:
    sessions: list[SessionSpec] = []
    for batch in raw_batches:
        fallback_anchor = batch.get("time_anchor") or "January-1-2024"
        previous_time: datetime | None = None
        first_time: datetime | None = None
        messages: list[dict[str, Any]] = []
        for turn in batch["turns"]:
            anchor = next((message.get("time_anchor") for message in turn if message.get("time_anchor")), None)
            current_time = _parse_date(anchor or fallback_anchor) if previous_time is None or anchor else previous_time
            first_time = first_time or current_time
            previous_time = current_time
            for message in turn:
                # BEAM raw messages carry integer ids while AgentScope Msg
                # requires a string id, so normalize before ingestion.
                if isinstance(message.get("id"), int):
                    message = {**message, "id": str(message["id"])}
                messages.append(
                    {
                        **message,
                        "name": message["role"],
                        "created_at": current_time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                )
        batch_number = str(batch["batch_number"])
        sessions.append(
            SessionSpec(
                session_id=f"beam_{variant}_{case_id}_batch{batch_number}",
                messages=messages,
                metadata={"batch_number": batch["batch_number"], "date": first_time.strftime("%Y-%m-%d")},
            ),
        )
    return sessions


def _source_chat_ids(value: Any) -> list[int]:
    """Flatten BEAM's list- or group-shaped source-chat references."""

    if isinstance(value, dict):
        return [chat_id for nested in value.values() for chat_id in _source_chat_ids(nested)]
    if isinstance(value, list):
        return [chat_id for nested in value for chat_id in _source_chat_ids(nested)]
    # One upstream 10M annotation contains the placeholder ``"--"``. It is not
    # a chat reference, so retain only the integer IDs used by chat messages.
    return [value] if isinstance(value, int) and not isinstance(value, bool) else []


def _queries(raw_questions: dict[str, list[dict[str, Any]]], sessions: list[SessionSpec]) -> list[QuerySpec]:
    queries: list[QuerySpec] = []
    session_ids_by_chat_id: dict[int, list[str]] = {}
    for session in sessions:
        for message in session.messages:
            chat_id = message.get("id")
            if isinstance(chat_id, str) and chat_id.isdigit():
                chat_id = int(chat_id)
            if isinstance(chat_id, int) and not isinstance(chat_id, bool):
                session_ids = session_ids_by_chat_id.setdefault(chat_id, [])
                if session.session_id not in session_ids:
                    session_ids.append(session.session_id)
    for question_type in sorted(raw_questions):
        for index, raw in enumerate(raw_questions[question_type], start=1):
            golden_answer = raw.get("ideal_answer", raw.get("ideal_response", raw.get("answer")))
            if golden_answer is None:
                golden_answer = raw.get("rubric", [])
            metadata = {
                key: value
                for key, value in raw.items()
                if key not in {"question", "ideal_answer", "ideal_response", "answer"}
            }
            metadata["question_type"] = question_type
            source_chat_ids = list(dict.fromkeys(_source_chat_ids(raw.get("source_chat_ids"))))
            if source_chat_ids:
                unknown_chat_ids = [chat_id for chat_id in source_chat_ids if chat_id not in session_ids_by_chat_id]
                if unknown_chat_ids:
                    raise ValueError(
                        f"BEAM query {question_type}:{index} references unknown chat IDs: {unknown_chat_ids}",
                    )
                metadata["evidence_session_ids"] = list(
                    dict.fromkeys(
                        session_id for chat_id in source_chat_ids for session_id in session_ids_by_chat_id[chat_id]
                    ),
                )
                metadata["evidence_is_certain"] = all(
                    len(session_ids_by_chat_id[chat_id]) == 1 for chat_id in source_chat_ids
                )
            queries.append(
                QuerySpec(
                    query_id=f"{question_type}:{index}",
                    question=raw["question"],
                    golden_answer=golden_answer,
                    metadata=metadata,
                ),
            )
    return queries


def selected_case_ids(source: Path, variant: str, requested: Iterable[str] | None) -> list[str]:
    """Select and validate case IDs for one BEAM variant."""

    cases_root = source / "chats" / variant
    if not cases_root.is_dir():
        raise FileNotFoundError(f"BEAM variant does not exist: {cases_root}")
    available = sorted(
        (path.name for path in cases_root.iterdir() if path.is_dir()),
        key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
    )
    return select_case_ids(available, requested)


def iter_cases(source: Path, variant: str, case_ids: Iterable[str]) -> Iterator[CaseSpec]:
    """Yield normalized BEAM cases one at a time."""

    cases_root = source / "chats" / variant
    for case_id in case_ids:
        case_root = cases_root / case_id
        raw_chat = json.loads((case_root / "chat.json").read_text(encoding="utf-8"))
        raw_questions = json.loads(
            (case_root / "probing_questions/probing_questions.json").read_text(encoding="utf-8"),
        )
        sessions = _sessions(raw_chat, variant, case_id)
        yield CaseSpec(
            case_id=case_id,
            sessions=sessions,
            queries=_queries(raw_questions, sessions),
            metadata={"dataset": "beam", "variant": variant},
        )


def load_cases(source: Path, variant: str, train_case_ids: Iterable[str] | None) -> list[CaseSpec]:
    """Load selected BEAM cases into the unified contracts."""

    return list(iter_cases(source, variant, selected_case_ids(source, variant, train_case_ids)))
