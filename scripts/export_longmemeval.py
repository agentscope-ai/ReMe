#!/usr/bin/env python3
"""Export LongMemEval samples into per-query benchmark directories."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "/Users/yuli/.cache/huggingface/hub/"
    "datasets--xiaowu0162--longmemeval-cleaned/"
    "snapshots/98d7416c24c778c2fee6e6f3006e7a073259d48f/"
    "longmemeval_s_cleaned.json"
)
DEFAULT_OUTPUT = Path("/Users/yuli/workspace/ReMe/longmemeval")


def safe_filename(value: str) -> str:
    """Make a stable, readable filename component from dataset ids/dates."""
    value = value.strip()
    value = value.replace("/", "-").replace(":", "-")
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9_.@()+-]", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("._") or "unknown"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def export_dataset(input_path: Path, output_dir: Path, clean: bool) -> dict[str, int]:
    with input_path.open(encoding="utf-8") as f:
        samples = json.load(f)

    if not isinstance(samples, list):
        raise TypeError(f"Expected top-level JSON list, got {type(samples).__name__}")

    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session_count = 0
    for question_index, sample in enumerate(samples):
        sample_dir = output_dir / str(question_index)
        session_dir = sample_dir / "session"

        query = {
            "question_id": sample["question_id"],
            "question_type": sample["question_type"],
            "question": sample["question"],
            "question_date": sample["question_date"],
        }
        answer = {
            "answer": sample["answer"],
            "answer_session_ids": sample["answer_session_ids"],
        }
        write_json(sample_dir / "query.json", query)
        write_json(sample_dir / "answer.json", answer)

        haystack_dates = sample["haystack_dates"]
        haystack_session_ids = sample["haystack_session_ids"]
        haystack_sessions = sample["haystack_sessions"]
        if not (
            len(haystack_dates)
            == len(haystack_session_ids)
            == len(haystack_sessions)
        ):
            raise ValueError(
                f"Misaligned haystack fields at question_index={question_index}"
            )

        used_filenames: set[str] = set()
        for haystack_date, haystack_session_id, messages in zip(
            haystack_dates, haystack_session_ids, haystack_sessions, strict=True
        ):
            base_name = (
                f"{safe_filename(haystack_date)}@"
                f"{safe_filename(haystack_session_id)}.json"
            )
            filename = base_name
            suffix = 1
            while filename in used_filenames:
                stem = base_name.removesuffix(".json")
                filename = f"{stem}__{suffix}.json"
                suffix += 1
            used_filenames.add(filename)

            session_payload = {
                "haystack_date": haystack_date,
                "haystack_session_id": haystack_session_id,
                "messages": messages,
            }
            write_json(session_dir / filename, session_payload)
            session_count += 1

    return {
        "samples": len(samples),
        "sessions": session_count,
        "output_dir": str(output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export LongMemEval JSON into per-query benchmark files."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete the output directory before exporting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = export_dataset(args.input, args.output, clean=not args.no_clean)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
