#!/usr/bin/env python3
"""Copy matching Claude Code session JSONL files into LongMemEval sample dirs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


DEFAULT_DATASET_DIR = Path("/Users/yuli/workspace/ReMe/longmemeval")
DEFAULT_CLAUDE_SESSION_DIR = Path("/Users/yuli/workspace/ReMe/.reme/session/claude_code")
DEFAULT_ANSWER_NAME = "longmemeval_session_answer.json"
DEFAULT_OUTPUT_NAME = "claude_code_session.jsonl"

QUERY_PATH_RE = re.compile(r"/longmemeval/(\d+)/query\.json")


@dataclass(frozen=True)
class ClaudeSession:
    path: Path
    indices: frozenset[int]
    final_text: str
    last_timestamp: datetime


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def iter_question_dirs(dataset_dir: Path) -> list[Path]:
    return sorted(
        [path for path in dataset_dir.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )


def parse_indices(value: str) -> set[int]:
    indices: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid index range: {part}")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    return indices


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.removesuffix("Z") + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunks)
    return ""


def read_claude_session(path: Path) -> ClaudeSession:
    indices: set[int] = set()
    assistant_texts: list[str] = []
    last_timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            for match in QUERY_PATH_RE.finditer(line):
                indices.add(int(match.group(1)))

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            parsed_time = parse_timestamp(payload.get("timestamp"))
            if parsed_time is not None and parsed_time > last_timestamp:
                last_timestamp = parsed_time

            message = payload.get("message")
            if not isinstance(message, dict):
                continue
            if message.get("role") != "assistant":
                continue
            text = extract_text_from_content(message.get("content")).strip()
            if text:
                assistant_texts.append(text)

    return ClaudeSession(
        path=path,
        indices=frozenset(indices),
        final_text=assistant_texts[-1] if assistant_texts else "",
        last_timestamp=last_timestamp,
    )


def build_session_index(session_dir: Path) -> dict[int, list[ClaudeSession]]:
    by_index: dict[int, list[ClaudeSession]] = {}
    for path in sorted(session_dir.glob("*.jsonl")):
        session = read_claude_session(path)
        for index in session.indices:
            by_index.setdefault(index, []).append(session)
    return by_index


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def answer_similarity(answer: str, candidate: str) -> float:
    answer_norm = normalize_text(answer)
    candidate_norm = normalize_text(candidate)
    if not answer_norm or not candidate_norm:
        return 0.0
    if answer_norm in candidate_norm:
        return 1.0
    if candidate_norm in answer_norm:
        return min(1.0, len(candidate_norm) / len(answer_norm))

    # Compare suffixes because the final answer is often preceded by reasoning text.
    answer_tail = answer_norm[-2000:]
    candidate_tail = candidate_norm[-2000:]
    return SequenceMatcher(None, answer_tail, candidate_tail).ratio()


def choose_session(
    candidates: list[ClaudeSession],
    answer_payload: dict[str, Any],
) -> tuple[ClaudeSession, float, bool]:
    answer = answer_payload.get("longmemeval_session_answer")
    answer_text = answer if isinstance(answer, str) else ""
    scored = [
        (answer_similarity(answer_text, candidate.final_text), candidate.last_timestamp, candidate)
        for candidate in candidates
    ]
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score, _, best = scored[0]
    ambiguous = len(scored) > 1 and scored[0][0] == scored[1][0]
    return best, best_score, ambiguous


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find the Claude Code JSONL session used for each "
            "longmemeval_session_answer.json and copy it into that index dir."
        ),
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--claude-session-dir", type=Path, default=DEFAULT_CLAUDE_SESSION_DIR)
    parser.add_argument("--answer-name", default=DEFAULT_ANSWER_NAME)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--indices", default="", help="Comma-separated indices/ranges, e.g. 0,3,10-20.")
    parser.add_argument("--copy", action="store_true", help="Actually copy files. Default is dry-run.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = parse_indices(args.indices) if args.indices else None
    sessions_by_index = build_session_index(args.claude_session_dir)

    stats = {
        "copied": 0,
        "would_copy": 0,
        "skipped_existing": 0,
        "missing_answer": 0,
        "missing_session": 0,
        "ambiguous": 0,
    }

    for question_dir in iter_question_dirs(args.dataset_dir):
        index = int(question_dir.name)
        if selected is not None and index not in selected:
            continue

        answer_path = question_dir / args.answer_name
        if not answer_path.exists():
            stats["missing_answer"] += 1
            print(f"[missing_answer] {index}: {answer_path}")
            continue

        candidates = sessions_by_index.get(index, [])
        if not candidates:
            stats["missing_session"] += 1
            print(f"[missing_session] {index}")
            continue

        chosen, score, ambiguous = choose_session(candidates, load_json(answer_path))
        if ambiguous:
            stats["ambiguous"] += 1

        output_path = question_dir / args.output_name
        if output_path.exists() and not args.overwrite:
            stats["skipped_existing"] += 1
            if not args.quiet:
                print(f"[skip_existing] {index}: {output_path}")
            continue

        action = "copy" if args.copy else "dry_run"
        if args.copy:
            shutil.copy2(chosen.path, output_path)
            stats["copied"] += 1
        else:
            stats["would_copy"] += 1

        if not args.quiet:
            suffix = " ambiguous" if ambiguous else ""
            print(
                f"[{action}] {index}: {chosen.path.name} -> "
                f"{output_path.name} score={score:.3f} candidates={len(candidates)}{suffix}"
            )

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
