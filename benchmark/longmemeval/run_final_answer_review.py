#!/usr/bin/env python3
"""Resolve every disputed LongMemEval answer with the configured Claude Code job.

The two input JSONL files are merged by ``question_id``.  Each selected question
is mapped to its numeric ``datasets/longmemeval/<idx>`` workspace and processed
sequentially with:

    reme start config=jinli_lme job=final_answer_review

The job returns a plain four-field JSON object with ``reason``,
``golden_answer_correct``, ``answer``, and ``is_session_time_wrong``.  After
every new success, this driver atomically rewrites the complete accumulated
output JSONL so an interrupted run can safely resume.

Examples:
    python benchmark/longmemeval/run_final_answer_review.py
    python benchmark/longmemeval/run_final_answer_review.py --limit 3
    python benchmark/longmemeval/run_final_answer_review.py --no-resume
    python benchmark/longmemeval/run_final_answer_review.py --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "datasets" / "longmemeval"
DEFAULT_REFERENCES = (
    REPO / "benchmark" / "longmemeval" / "golden_check_list_false.jsonl",
    REPO / "benchmark" / "longmemeval" / "merge_confirm_jinli_false.jsonl",
)
DEFAULT_OUTPUT = REPO / "benchmark" / "longmemeval" / "final_answer_review.jsonl"
DEFAULT_LOG_DIR = REPO / "logs" / "final_answer_review"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output JSONL (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="directory for per-question logs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="process only the first N pending questions (0 = all)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore existing output and rerun every selected question",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the merged cases without invoking ReMe",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and reject malformed or non-object rows."""
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object at {path}:{line_number}")
                rows.append(row)
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read JSONL file: {path}") from exc
    return rows


def merge_references(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    """Merge reference rows by question ID, preserving file and row order."""
    merged: dict[str, list[dict[str, Any]]] = {}
    seen_sources: set[tuple[str, str]] = set()
    for path in paths:
        for row in _read_jsonl(path):
            question_id = str(row.get("question_id") or "").strip()
            if not question_id:
                raise ValueError(f"Reference row in {path} has no question_id")
            source_key = (question_id, str(path.resolve()))
            if source_key in seen_sources:
                raise ValueError(f"Duplicate question_id={question_id!r} within {path}")
            seen_sources.add(source_key)
            merged.setdefault(question_id, []).append({"source": path.name, **row})
    if not merged:
        raise ValueError("No reference answers found")
    return merged


def workspace_map() -> dict[str, Path]:
    """Map every dataset question ID to its numeric sample workspace."""
    mapping: dict[str, Path] = {}
    for workspace in sorted(
        (path for path in DATA.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda p: int(p.name),
    ):
        query_path = workspace / "query.json"
        if not query_path.is_file():
            continue
        try:
            with query_path.open(encoding="utf-8") as file:
                query = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot parse {query_path}") from exc
        if not isinstance(query, dict):
            raise ValueError(f"Expected a JSON object in {query_path}")
        question_id = str(query.get("question_id") or "").strip()
        if not question_id:
            raise ValueError(f"Missing question_id in {query_path}")
        if question_id in mapping:
            raise ValueError(
                f"Duplicate dataset question_id={question_id!r}: {mapping[question_id]} and {workspace}",
            )
        mapping[question_id] = workspace
    return mapping


def _validate_result(value: Any, *, source: str) -> dict[str, Any]:
    """Validate the final four-field answer contract."""
    expected_keys = {"reason", "golden_answer_correct", "answer", "is_session_time_wrong"}
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(
            f"{source} must contain exactly 'reason', 'golden_answer_correct', 'answer', "
            "and 'is_session_time_wrong'",
        )
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError(f"{source} has an invalid reason")
    if not isinstance(value["golden_answer_correct"], bool):
        raise ValueError(f"{source} has an invalid golden_answer_correct")
    if not isinstance(value["answer"], str):
        raise ValueError(f"{source} has an invalid answer")
    answer = value["answer"].strip()
    if value["golden_answer_correct"] and answer:
        raise ValueError(f"{source} answer must be empty when golden_answer_correct is true")
    if not value["golden_answer_correct"] and not answer:
        raise ValueError(f"{source} answer must be non-empty when golden_answer_correct is false")
    if not isinstance(value["is_session_time_wrong"], bool):
        raise ValueError(f"{source} has an invalid is_session_time_wrong")
    return {
        "reason": value["reason"].strip(),
        "golden_answer_correct": value["golden_answer_correct"],
        "answer": answer,
        "is_session_time_wrong": value["is_session_time_wrong"],
    }


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    """Load resumable output, rejecting duplicate or malformed rows."""
    if not path.exists():
        return {}
    results: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        question_id = str(row.get("question_id") or "").strip()
        if not question_id:
            raise ValueError(f"Existing output row in {path} has no question_id")
        if question_id in results:
            raise ValueError(
                f"Duplicate question_id={question_id!r} in existing output {path}",
            )
        results[question_id] = _validate_result(
            {key: value for key, value in row.items() if key != "question_id"},
            source=f"existing result for {question_id}",
        )
    return results


def atomic_write_results(
    path: Path,
    order: list[str],
    results: dict[str, dict[str, Any]],
) -> None:
    """Atomically rewrite all accumulated rows in stable merged-input order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as file:
            temp_path = Path(file.name)
            for question_id in order:
                if question_id not in results:
                    continue
                row = {"question_id": question_id, **results[question_id]}
                file.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n",
                )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def run_one(question_id: str, workspace: Path, log_dir: Path) -> dict[str, Any]:
    """Run the configured one-shot job and validate its stdout JSON."""
    env = dict(os.environ, LME_WORKSPACE_DIR=str(workspace.relative_to(REPO)))
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from reme.reme import main; main()",
            "start",
            "config=jinli_lme",
            "job=final_answer_review",
        ],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{question_id}.log"
    log_text = (
        f"workspace={workspace}\nreturncode={completed.returncode}\n\n"
        f"[stdout]\n{completed.stdout}\n[stderr]\n{completed.stderr}"
    )
    log_path.write_text(
        log_text,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Job failed for {question_id} with rc={completed.returncode}; see {log_path}",
        )
    try:
        value = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Job stdout is not JSON for {question_id}; see {log_path}",
        ) from exc
    return _validate_result(value, source=f"job result for {question_id}")


def main() -> int:
    """Merge, run, and checkpoint every disputed case sequentially."""
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")

    references = merge_references(list(DEFAULT_REFERENCES))
    mapping = workspace_map()
    missing = [question_id for question_id in references if question_id not in mapping]
    if missing:
        raise ValueError(f"No dataset workspace for question IDs: {', '.join(missing)}")

    order = list(references)
    results = {} if args.no_resume else load_existing(args.output.resolve())
    pending = [question_id for question_id in order if question_id not in results]
    if args.limit:
        pending = pending[: args.limit]

    one_reference = sum(len(items) == 1 for items in references.values())
    two_references = sum(len(items) == 2 for items in references.values())
    print(
        f"merged={len(order)} one_reference={one_reference} two_references={two_references} "
        f"existing={len(results)} pending={len(pending)} output={args.output.resolve()}",
        flush=True,
    )

    if args.dry_run:
        for question_id in pending:
            print(
                f"[would-run] question_id={question_id} workspace={mapping[question_id].name} "
                f"references={len(references[question_id])}",
            )
        return 0

    for position, question_id in enumerate(pending, start=1):
        workspace = mapping[question_id]
        print(
            f"[start {position}/{len(pending)}] question_id={question_id} workspace={workspace.name} "
            f"references={len(references[question_id])}",
            flush=True,
        )
        result = run_one(question_id, workspace, args.log_dir.resolve())
        results[question_id] = result
        atomic_write_results(args.output.resolve(), order, results)
        print(
            f"[saved {position}/{len(pending)}] question_id={question_id}",
            flush=True,
        )

    print(
        f"ALL FINISHED total_saved={sum(question_id in results for question_id in order)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
