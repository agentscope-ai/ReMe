#!/usr/bin/env python3
"""Summarize LongMemEval result JSON files for one answer id."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "longmemeval"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"result must be a JSON object: {path}")
    return data


def _pct(count: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{count / total * 100:.2f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-id", required=True, help="answer id saved as result/{answer_id}.json")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=499)
    parser.add_argument("--show-failures", action="store_true", help="print non-yes completed indices")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    indices = list(range(args.start_index, args.end_index + 1))
    completed: list[tuple[int, Path, dict[str, Any]]] = []
    unreadable: list[tuple[int, Path, str]] = []
    missing: list[int] = []

    for index in indices:
        path = args.dataset_root / str(index) / "result" / f"{args.answer_id}.json"
        if not path.is_file():
            missing.append(index)
            continue
        try:
            completed.append((index, path, _read_json(path)))
        except Exception as exc:  # pylint: disable=broad-except
            unreadable.append((index, path, str(exc)))

    yes: list[int] = []
    no: list[int] = []
    other: list[tuple[int, str]] = []
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"completed": 0, "yes": 0, "no": 0, "other": 0})

    for index, _path, data in completed:
        judgement = str(data.get("answer_judgement") or "").strip().lower()
        question_type = str(data.get("question_type") or "unknown").strip() or "unknown"
        by_type[question_type]["completed"] += 1
        if judgement == "yes":
            yes.append(index)
            by_type[question_type]["yes"] += 1
        elif judgement == "no":
            no.append(index)
            by_type[question_type]["no"] += 1
        else:
            other.append((index, judgement))
            by_type[question_type]["other"] += 1

    completed_count = len(completed)
    total = len(indices)
    print(f"answer_id: {args.answer_id}")
    print(f"range: {args.start_index}-{args.end_index}")
    print(f"completed: {completed_count}/{total} ({_pct(completed_count, total)})")
    print(f"accuracy_completed: {len(yes)}/{completed_count} ({_pct(len(yes), completed_count)})")
    print(f"accuracy_total: {len(yes)}/{total} ({_pct(len(yes), total)})")
    print(f"yes: {len(yes)}")
    print(f"no: {len(no)}")
    print(f"other_judgement: {len(other)}")
    print(f"missing: {len(missing)}")
    print(f"unreadable: {len(unreadable)}")
    print()
    print("by_question_type:")
    if by_type:
        print("question_type\tcompleted\tyes\tno\tother\taccuracy_completed")
        for question_type in sorted(by_type):
            row = by_type[question_type]
            print(
                f"{question_type}\t{row['completed']}\t{row['yes']}\t{row['no']}\t"
                f"{row['other']}\t{_pct(row['yes'], row['completed'])}",
            )
    else:
        print("(no completed results)")

    if args.show_failures:
        failed = no + [index for index, _judgement in other]
        print("failed_indices:", ",".join(map(str, sorted(failed))) if failed else "")
        if unreadable:
            print("unreadable_indices:", ",".join(str(index) for index, _path, _err in unreadable))
    return 1 if unreadable else 0


if __name__ == "__main__":
    raise SystemExit(main())
