#!/usr/bin/env python3
"""Simulate a LongMemEval session-retrieval benchmark run.

The script reads one exported question directory:

    longmemeval/{question_index}/query.json
    longmemeval/{question_index}/answer.json
    longmemeval/{question_index}/session/*.json

It ranks haystack sessions for the query, returns the top-k session ids, and
scores them by the fraction of gold answer_session_ids covered by top-k.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


DEFAULT_QUESTION_DIR = Path("/Users/yuli/workspace/ReMe/longmemeval/0")
DEFAULT_DATASET_DIR = Path("/Users/yuli/workspace/ReMe/longmemeval")
TOP_K = 5

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "you",
    "your",
}


@dataclass(frozen=True)
class SessionDoc:
    path: Path
    session_id: str
    date: str
    all_text: str
    user_text: str
    all_tokens: list[str]
    user_tokens: list[str]
    parsed_date: datetime | None


@dataclass(frozen=True)
class TimeSignal:
    kind: str
    weight: float
    target_start: datetime | None = None
    target_end: datetime | None = None
    description: str = ""


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    if keep_stopwords:
        return tokens
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(message)


def parse_dataset_datetime(value: str) -> datetime | None:
    match = re.search(r"(\d{4})/(\d{2})/(\d{2}).*?(\d{2}):(\d{2})", value)
    if not match:
        return None
    year, month, day, hour, minute = map(int, match.groups())
    return datetime(year, month, day, hour, minute)


def parse_reme_date(value: Any, *, end_of_day: bool = False) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    value = value.strip().strip("\"'")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None
    if end_of_day:
        return parsed.replace(hour=23, minute=59, second=59)
    return parsed


def normalize_reme_memory_time_range(memory_time_range: Any) -> dict[str, str]:
    if not isinstance(memory_time_range, dict):
        return {}

    normalized = {}
    thinking = memory_time_range.get("thinking")
    if isinstance(thinking, str) and thinking.strip():
        normalized["thinking"] = thinking.strip()
    for key in ("start_dt", "end_dt"):
        value = memory_time_range.get(key)
        if not isinstance(value, str) or not value:
            continue
        value = value.strip().strip("\"'")
        if parse_reme_date(value) is not None:
            normalized[key] = value
    return normalized


def reme_range_to_signal(memory_time_range: Any) -> TimeSignal | None:
    memory_time_range = normalize_reme_memory_time_range(memory_time_range)
    if not isinstance(memory_time_range, dict):
        return None

    start = parse_reme_date(memory_time_range.get("start_dt"))
    end = parse_reme_date(memory_time_range.get("end_dt"), end_of_day=True)
    if start is None and end is None:
        return None

    if start and end:
        description = f"reme memory_time_range {start.date()} to {end.date()}"
    elif start:
        description = f"reme memory_time_range after {start.date()}"
    else:
        description = f"reme memory_time_range before {end.date()}"

    return TimeSignal(
        kind="reme_range",
        target_start=start,
        target_end=end,
        weight=2.5,
        description=description,
    )


def parse_reme_stdout(stdout: str) -> dict[str, Any]:
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("✅"):
            line = line.removeprefix("✅").strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            if "memory_time_range" in payload and isinstance(
                payload["memory_time_range"], dict
            ):
                return normalize_reme_memory_time_range(payload["memory_time_range"])
            if "structured_output" in payload and isinstance(
                payload["structured_output"], dict
            ):
                return normalize_reme_memory_time_range(payload["structured_output"])
            if "start_dt" in payload or "end_dt" in payload:
                return normalize_reme_memory_time_range(payload)
    return {}


def request_reme_memory_time_range(question: str, question_date: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "reme",
            "memory_time_range",
            f"question={question}",
            f"question_date={question_date}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "reme memory_time_range failed with exit code "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )
    return parse_reme_stdout(completed.stdout)


def get_or_cache_reme_memory_time_range(
    query_path: Path, query: dict[str, Any], *, refresh: bool = False
) -> dict[str, Any]:
    existing = query.get("memory_time_range")
    if isinstance(existing, dict) and not refresh:
        normalized = normalize_reme_memory_time_range(existing)
        if normalized != existing:
            query["memory_time_range"] = normalized
            write_json(query_path, query)
        return normalized

    memory_time_range = request_reme_memory_time_range(
        str(query["question"]), str(query["question_date"])
    )
    query["memory_time_range"] = memory_time_range
    write_json(query_path, query)
    return memory_time_range


def read_session(path: Path) -> SessionDoc:
    session = load_json(path)
    messages = session.get("messages", [])
    all_parts: list[str] = []
    user_parts: list[str] = []

    for message in messages:
        content = message_content(message)
        all_parts.append(content)
        if isinstance(message, dict) and message.get("role") == "user":
            user_parts.append(content)

    all_text = "\n".join(all_parts)
    user_text = "\n".join(user_parts)
    return SessionDoc(
        path=path,
        session_id=str(session["haystack_session_id"]),
        date=str(session.get("haystack_date", "")),
        all_text=all_text,
        user_text=user_text,
        all_tokens=tokenize(all_text),
        user_tokens=tokenize(user_text),
        parsed_date=parse_dataset_datetime(str(session.get("haystack_date", ""))),
    )


def idf_by_token(docs: list[list[str]]) -> dict[str, float]:
    doc_count = len(docs)
    document_frequency: Counter[str] = Counter()
    for tokens in docs:
        document_frequency.update(set(tokens))

    return {
        token: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
        for token, freq in document_frequency.items()
    }


def bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    idf: dict[str, float],
    avg_doc_len: float,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0

    counts = Counter(doc_tokens)
    doc_len = len(doc_tokens)
    score = 0.0
    for token in query_tokens:
        term_frequency = counts.get(token, 0)
        if term_frequency == 0:
            continue
        denominator = term_frequency + k1 * (1 - b + b * doc_len / avg_doc_len)
        score += idf.get(token, 0.0) * term_frequency * (k1 + 1) / denominator
    return score


def phrase_overlap_score(query_tokens: list[str], text: str) -> float:
    """Small bonus for exact query-token phrase fragments in the session text."""
    if not query_tokens:
        return 0.0

    text = text.lower()
    score = 0.0
    for token in set(query_tokens):
        if re.search(rf"\b{re.escape(token)}\b", text):
            score += 0.15

    for size in (2, 3):
        for i in range(0, len(query_tokens) - size + 1):
            phrase = " ".join(query_tokens[i : i + size])
            if phrase in text:
                score += size * 0.4
    return score


def temporal_signal(question: str, question_date: str | None) -> TimeSignal | None:
    if not question_date:
        return None

    anchor = parse_dataset_datetime(question_date)
    if anchor is None:
        return None

    text = question.lower()
    if re.search(r"\byesterday\b", text):
        target = anchor - timedelta(days=1)
        return TimeSignal(
            kind="target_day",
            target_start=target.replace(hour=0, minute=0),
            target_end=target.replace(hour=23, minute=59),
            weight=2.0,
            description="query mentions yesterday",
        )

    if re.search(r"\btoday\b", text):
        return TimeSignal(
            kind="target_day",
            target_start=anchor.replace(hour=0, minute=0),
            target_end=anchor.replace(hour=23, minute=59),
            weight=1.5,
            description="query mentions today",
        )

    match = re.search(r"\b(\d+)\s+days?\s+ago\b", text)
    if match:
        target = anchor - timedelta(days=int(match.group(1)))
        return TimeSignal(
            kind="target_day",
            target_start=target.replace(hour=0, minute=0),
            target_end=target.replace(hour=23, minute=59),
            weight=2.0,
            description=f"query mentions {match.group(1)} days ago",
        )

    match = re.search(r"\b(\d+)\s+weeks?\s+ago\b", text)
    if match:
        target = anchor - timedelta(weeks=int(match.group(1)))
        return TimeSignal(
            kind="target_week",
            target_start=(target - timedelta(days=3)).replace(hour=0, minute=0),
            target_end=(target + timedelta(days=3)).replace(hour=23, minute=59),
            weight=1.6,
            description=f"query mentions {match.group(1)} weeks ago",
        )

    match = re.search(r"\b(\d+)\s+months?\s+ago\b", text)
    if match:
        target = anchor - timedelta(days=30 * int(match.group(1)))
        return TimeSignal(
            kind="target_month",
            target_start=(target - timedelta(days=15)).replace(hour=0, minute=0),
            target_end=(target + timedelta(days=15)).replace(hour=23, minute=59),
            weight=1.2,
            description=f"query mentions {match.group(1)} months ago",
        )

    if re.search(r"\blast\s+(week|month|year|visited|time)\b", text):
        return TimeSignal(
            kind="prefer_recent_before_question",
            target_end=anchor,
            weight=0.8,
            description="query mentions last/recent previous event",
        )

    if re.search(r"\bpast\s+three\s+months\b|\bpast\s+3\s+months\b", text):
        return TimeSignal(
            kind="window",
            target_start=anchor - timedelta(days=90),
            target_end=anchor,
            weight=1.0,
            description="query mentions past three months",
        )

    if re.search(r"\b(first|last|earliest|latest|order|between|before|after)\b", text):
        return TimeSignal(
            kind="weak_temporal",
            target_end=anchor,
            weight=0.3,
            description="query asks for temporal reasoning",
        )

    return None


def time_score(session_date: datetime | None, signal: TimeSignal | None) -> float:
    if session_date is None or signal is None:
        return 0.0

    if signal.kind == "reme_range":
        if signal.target_start and session_date < signal.target_start:
            distance_days = (signal.target_start - session_date).total_seconds() / 86400
            return signal.weight / (1.0 + distance_days)
        if signal.target_end and session_date > signal.target_end:
            distance_days = (session_date - signal.target_end).total_seconds() / 86400
            return signal.weight / (1.0 + distance_days)
        return signal.weight

    if signal.kind in {"target_day", "target_week", "target_month", "window"}:
        if signal.target_start and signal.target_end:
            if signal.target_start <= session_date <= signal.target_end:
                return signal.weight
            if session_date < signal.target_start:
                distance_days = (signal.target_start - session_date).total_seconds() / 86400
            else:
                distance_days = (session_date - signal.target_end).total_seconds() / 86400
            return signal.weight / (1.0 + distance_days)

    if signal.kind == "prefer_recent_before_question" and signal.target_end:
        if session_date > signal.target_end:
            return 0.0
        distance_days = (signal.target_end - session_date).total_seconds() / 86400
        return signal.weight / (1.0 + distance_days / 7)

    if signal.kind == "weak_temporal" and signal.target_end:
        if session_date > signal.target_end:
            return 0.0
        distance_days = (signal.target_end - session_date).total_seconds() / 86400
        return signal.weight / (1.0 + distance_days / 30)

    return 0.0


def session_in_time_range(session: SessionDoc, signal: TimeSignal | None) -> bool:
    if signal is None or signal.kind != "reme_range" or session.parsed_date is None:
        return True
    if signal.target_start and session.parsed_date < signal.target_start:
        return False
    if signal.target_end and session.parsed_date > signal.target_end:
        return False
    return True


def rank_sessions(
    question: str,
    sessions: list[SessionDoc],
    *,
    question_date: str | None = None,
    time_aware: bool = False,
    memory_time_range: dict[str, Any] | None = None,
    filter_by_time_range: bool = False,
) -> list[dict[str, Any]]:
    signal = reme_range_to_signal(memory_time_range) if memory_time_range else None
    if signal and filter_by_time_range:
        filtered_sessions = [session for session in sessions if session_in_time_range(session, signal)]
        if filtered_sessions:
            sessions = filtered_sessions

    query_tokens = tokenize(question)
    all_docs = [session.all_tokens for session in sessions]
    user_docs = [session.user_tokens for session in sessions]
    all_idf = idf_by_token(all_docs)
    user_idf = idf_by_token(user_docs)
    avg_all_len = max(1.0, sum(len(tokens) for tokens in all_docs) / len(all_docs))
    avg_user_len = max(1.0, sum(len(tokens) for tokens in user_docs) / len(user_docs))
    if signal is None and time_aware:
        signal = temporal_signal(question, question_date)

    ranked = []
    for session in sessions:
        all_score = bm25_score(query_tokens, session.all_tokens, all_idf, avg_all_len)
        user_score = bm25_score(query_tokens, session.user_tokens, user_idf, avg_user_len)
        phrase_score = phrase_overlap_score(query_tokens, session.all_text)
        session_time_score = time_score(session.parsed_date, signal)
        score = all_score + 0.6 * user_score + phrase_score + session_time_score
        ranked.append(
            {
                "haystack_session_id": session.session_id,
                "haystack_date": session.date,
                "score": score,
                "text_score": all_score + 0.6 * user_score + phrase_score,
                "time_score": session_time_score,
                "path": str(session.path),
            }
        )

    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def evaluate_question(
    question_dir: Path,
    top_k: int,
    *,
    time_aware: bool,
    use_reme_time_range: bool = False,
    refresh_reme_time_range: bool = False,
    filter_by_time_range: bool = False,
) -> dict[str, Any]:
    query_path = question_dir / "query.json"
    query = load_json(query_path)
    answer = load_json(question_dir / "answer.json")
    session_paths = sorted((question_dir / "session").glob("*.json"))
    sessions = [read_session(path) for path in session_paths]

    memory_time_range = None
    if use_reme_time_range:
        memory_time_range = get_or_cache_reme_memory_time_range(
            query_path, query, refresh=refresh_reme_time_range
        )

    reme_signal = reme_range_to_signal(memory_time_range)
    time_signal = reme_signal or temporal_signal(
        str(query["question"]), str(query["question_date"])
    )
    ranked = rank_sessions(
        str(query["question"]),
        sessions,
        question_date=str(query["question_date"]),
        time_aware=time_aware,
        memory_time_range=memory_time_range,
        filter_by_time_range=filter_by_time_range,
    )
    top = ranked[:top_k]
    predicted_ids = [item["haystack_session_id"] for item in top]
    gold_ids = [str(session_id) for session_id in answer["answer_session_ids"]]
    hit_ids = [session_id for session_id in gold_ids if session_id in predicted_ids]
    score = len(hit_ids) / len(gold_ids) if gold_ids else 0.0

    return {
        "question_dir": str(question_dir),
        "question_id": query["question_id"],
        "question_type": query["question_type"],
        "question": query["question"],
        "gold_answer": answer["answer"],
        "gold_answer_session_ids": gold_ids,
        "top_k": top_k,
        "predicted_haystack_session_ids": predicted_ids,
        "hit_answer_session_ids": hit_ids,
        "score": score,
        "time_aware": time_aware,
        "use_reme_time_range": use_reme_time_range,
        "filter_by_time_range": filter_by_time_range,
        "memory_time_range": memory_time_range,
        "time_signal": None
        if time_signal is None
        else {
            "kind": time_signal.kind,
            "description": time_signal.description,
            "target_start": time_signal.target_start.isoformat()
            if time_signal.target_start
            else None,
            "target_end": time_signal.target_end.isoformat()
            if time_signal.target_end
            else None,
            "weight": time_signal.weight,
        },
        "ranked_top": top,
    }


def evaluate_all(
    dataset_dir: Path,
    top_k: int,
    *,
    time_aware: bool,
    use_reme_time_range: bool = False,
    refresh_reme_time_range: bool = False,
    filter_by_time_range: bool = False,
) -> dict[str, Any]:
    question_dirs = sorted(
        [path for path in dataset_dir.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )
    results = [
        evaluate_question(
            question_dir,
            top_k,
            time_aware=time_aware,
            use_reme_time_range=use_reme_time_range,
            refresh_reme_time_range=refresh_reme_time_range,
            filter_by_time_range=filter_by_time_range,
        )
        for question_dir in question_dirs
    ]
    scores = [result["score"] for result in results]
    full_hits = sum(1 for score in scores if score == 1.0)
    partial_hits = sum(1 for score in scores if 0.0 < score < 1.0)

    return {
        "dataset_dir": str(dataset_dir),
        "questions": len(results),
        "top_k": top_k,
        "time_aware": time_aware,
        "use_reme_time_range": use_reme_time_range,
        "filter_by_time_range": filter_by_time_range,
        "average_score": sum(scores) / len(scores) if scores else 0.0,
        "full_hit_questions": full_hits,
        "partial_hit_questions": partial_hits,
        "zero_hit_questions": len(scores) - full_hits - partial_hits,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate LongMemEval top-k haystack-session retrieval scoring."
    )
    parser.add_argument("--question-dir", type=Path, default=DEFAULT_QUESTION_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate every numeric question directory under --dataset-dir.",
    )
    parser.add_argument(
        "--time-aware",
        action="store_true",
        help="Parse query/session dates and add a temporal ranking score.",
    )
    parser.add_argument(
        "--use-reme-time-range",
        action="store_true",
        help="Use cached query.json memory_time_range or call reme to populate it.",
    )
    parser.add_argument(
        "--refresh-reme-time-range",
        action="store_true",
        help="Call reme even when query.json already has memory_time_range.",
    )
    parser.add_argument(
        "--filter-by-reme-time-range",
        action="store_true",
        help="Filter candidate sessions to the reme memory_time_range before ranking.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="When using --all, omit per-question ranked results from stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.all:
        result = evaluate_all(
            args.dataset_dir,
            args.top_k,
            time_aware=args.time_aware,
            use_reme_time_range=args.use_reme_time_range,
            refresh_reme_time_range=args.refresh_reme_time_range,
            filter_by_time_range=args.filter_by_reme_time_range,
        )
        if args.summary_only:
            result.pop("results", None)
    else:
        result = evaluate_question(
            args.question_dir,
            args.top_k,
            time_aware=args.time_aware,
            use_reme_time_range=args.use_reme_time_range,
            refresh_reme_time_range=args.refresh_reme_time_range,
            filter_by_time_range=args.filter_by_reme_time_range,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
