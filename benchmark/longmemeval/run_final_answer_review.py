#!/usr/bin/env python3
"""Review every LongMemEval golden answer with the configured Claude Code job.

Every numeric ``datasets/longmemeval/<idx>`` workspace is processed sequentially.
The reference JSONL files are merged by ``question_id`` and supplied only when
they contain an alternative answer for that sample:

    reme start config=jinli_lme job=final_answer_review

The job returns a plain four-field JSON object with ``reason``,
``golden_answer_correct``, ``answer``, and ``is_session_time_wrong``.  After
every new success, this driver atomically rewrites the complete accumulated
output JSONL so an interrupted run can safely resume.

Examples:
    python benchmark/longmemeval/run_final_answer_review.py
    python benchmark/longmemeval/run_final_answer_review.py --exclude-reference-question-ids
    python benchmark/longmemeval/run_final_answer_review.py --only-reference-question-ids --rerun-selected
    python benchmark/longmemeval/run_final_answer_review.py --concurrency 2 --submit-interval-seconds 6
    python benchmark/longmemeval/run_final_answer_review.py --question-id e47becba
    python benchmark/longmemeval/run_final_answer_review.py --reference path/to/results.jsonl
    python benchmark/longmemeval/run_final_answer_review.py --limit 3
    python benchmark/longmemeval/run_final_answer_review.py --no-resume
    python benchmark/longmemeval/run_final_answer_review.py --dry-run
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
import time
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
REFERENCE_PATHS_ENV = "LME_FINAL_ANSWER_REFERENCE_PATHS"
MAX_CONCURRENCY = 3
MIN_SUBMIT_INTERVAL_SECONDS = 5.0
DEFAULT_SUBMIT_INTERVAL_SECONDS = 5.1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--question-id",
        dest="question_ids",
        action="append",
        help="process only this dataset question ID; repeat for multiple IDs (default: all)",
    )
    reference_selection = parser.add_mutually_exclusive_group()
    reference_selection.add_argument(
        "--exclude-reference-question-ids",
        action="store_true",
        help="skip question IDs found in the selected reference-answer JSONL files",
    )
    reference_selection.add_argument(
        "--only-reference-question-ids",
        action="store_true",
        help="process only question IDs found in the selected reference-answer JSONL files",
    )
    parser.add_argument(
        "--reference",
        dest="references",
        action="append",
        type=Path,
        help="reference-answer JSONL; repeat for multiple files (default: built-in disputed results)",
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
        "--concurrency",
        type=int,
        default=MAX_CONCURRENCY,
        help=f"maximum concurrent jobs, from 1 to {MAX_CONCURRENCY} (default: {MAX_CONCURRENCY})",
    )
    parser.add_argument(
        "--submit-interval-seconds",
        type=float,
        default=DEFAULT_SUBMIT_INTERVAL_SECONDS,
        help=f"minimum time between job submissions; must be > {MIN_SUBMIT_INTERVAL_SECONDS:g} "
        f"(default: {DEFAULT_SUBMIT_INTERVAL_SECONDS:g})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="process only the first N pending questions (0 = all)",
    )
    resume_mode = parser.add_mutually_exclusive_group()
    resume_mode.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore existing output and rerun every selected question",
    )
    resume_mode.add_argument(
        "--rerun-selected",
        action="store_true",
        help="rerun every selected question while preserving existing results until replacements finish",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the selected cases without invoking ReMe",
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


def select_question_ids(
    mapping: dict[str, Path],
    requested: list[str] | None,
    excluded: set[str] | None = None,
) -> list[str]:
    """Return all dataset IDs or validate an explicitly requested subset."""
    excluded = excluded or set()
    if not requested:
        return [question_id for question_id in mapping if question_id not in excluded]
    selected: list[str] = []
    seen: set[str] = set()
    for raw_question_id in requested:
        question_id = raw_question_id.strip()
        if not question_id:
            raise ValueError("--question-id must not be empty")
        if question_id in seen:
            raise ValueError(f"Duplicate --question-id: {question_id}")
        if question_id not in mapping:
            raise ValueError(f"No dataset workspace for question ID: {question_id}")
        if question_id not in excluded:
            selected.append(question_id)
        seen.add(question_id)
    return selected


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
        "is_session_time_wrong": False,
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


def run_one(
    question_id: str,
    workspace: Path,
    log_dir: Path,
    reference_paths: list[Path],
) -> dict[str, Any]:
    """Run the configured one-shot job and validate its stdout JSON."""
    env = dict(os.environ, LME_WORKSPACE_DIR=str(workspace.relative_to(REPO)))
    env[REFERENCE_PATHS_ENV] = json.dumps(
        [str(path.resolve()) for path in reference_paths],
        ensure_ascii=False,
    )
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
    """Review and checkpoint the selected dataset cases sequentially."""
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if not 1 <= args.concurrency <= MAX_CONCURRENCY:
        raise ValueError(f"--concurrency must be between 1 and {MAX_CONCURRENCY}")
    if args.submit_interval_seconds <= MIN_SUBMIT_INTERVAL_SECONDS:
        raise ValueError(
            f"--submit-interval-seconds must be > {MIN_SUBMIT_INTERVAL_SECONDS:g}",
        )

    reference_paths = [path.resolve() for path in (args.references or DEFAULT_REFERENCES)]
    mapping = workspace_map()
    references = merge_references(reference_paths)
    missing = [question_id for question_id in references if question_id not in mapping]
    if missing:
        raise ValueError(f"No dataset workspace for question IDs: {', '.join(missing)}")

    full_order = list(mapping)
    excluded = set(references) if args.exclude_reference_question_ids else set()
    order = select_question_ids(mapping, args.question_ids, excluded)
    if args.only_reference_question_ids:
        order = [question_id for question_id in order if question_id in references]
    results = {} if args.no_resume else load_existing(args.output.resolve())
    pending = (
        list(order) if args.rerun_selected else [question_id for question_id in order if question_id not in results]
    )
    if args.limit:
        pending = pending[: args.limit]

    no_reference = sum(question_id not in references for question_id in order)
    one_reference = sum(len(references.get(question_id, [])) == 1 for question_id in order)
    multiple_references = sum(len(references.get(question_id, [])) > 1 for question_id in order)
    print(
        f"total={len(order)} no_reference={no_reference} one_reference={one_reference} "
        f"multiple_references={multiple_references} "
        f"excluded={len(excluded)} "
        f"only_reference_questions={args.only_reference_question_ids} "
        f"concurrency={args.concurrency} submit_interval={args.submit_interval_seconds:g}s "
        f"existing={len(results)} pending={len(pending)} output={args.output.resolve()}",
        flush=True,
    )

    if args.dry_run:
        for question_id in pending:
            print(
                f"[would-run] question_id={question_id} workspace={mapping[question_id].name} "
                f"references={len(references.get(question_id, []))}",
            )
        return 0

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency)
    active: dict[concurrent.futures.Future[dict[str, Any]], tuple[int, str]] = {}
    next_position = 0
    saved_count = 0
    next_submit_at = 0.0
    try:
        while next_position < len(pending) or active:
            can_submit = next_position < len(pending) and len(active) < args.concurrency
            if can_submit and time.monotonic() >= next_submit_at:
                question_id = pending[next_position]
                position = next_position + 1
                workspace = mapping[question_id]
                print(
                    f"[submit {position}/{len(pending)}] question_id={question_id} "
                    f"workspace={workspace.name} references={len(references.get(question_id, []))}",
                    flush=True,
                )
                future = executor.submit(
                    run_one,
                    question_id,
                    workspace,
                    args.log_dir.resolve(),
                    reference_paths,
                )
                active[future] = (position, question_id)
                next_position += 1
                next_submit_at = time.monotonic() + args.submit_interval_seconds
                continue

            if not active:
                time.sleep(max(0.0, next_submit_at - time.monotonic()))
                continue

            timeout = None
            if can_submit:
                timeout = max(0.0, next_submit_at - time.monotonic())
            done, _ = concurrent.futures.wait(
                active,
                timeout=timeout,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                position, question_id = active.pop(future)
                results[question_id] = future.result()
                atomic_write_results(args.output.resolve(), full_order, results)
                saved_count += 1
                print(
                    f"[saved {saved_count}/{len(pending)}] submitted_position={position} " f"question_id={question_id}",
                    flush=True,
                )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    print(
        f"ALL FINISHED total_saved={sum(question_id in results for question_id in order)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
