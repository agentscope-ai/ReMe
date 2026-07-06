#!/usr/bin/env python3
"""Cache reme longmemeval_session_answer outputs for LongMemEval queries."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_DATASET_DIR = Path("/Users/yuli/workspace/ReMe/longmemeval")
DEFAULT_OUTPUT_NAME = "longmemeval_session_answer.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
        tmp_path = Path(f.name)
    tmp_path.replace(path)


def iter_question_dirs(dataset_dir: Path) -> list[Path]:
    question_dirs = [
        path for path in dataset_dir.iterdir() if path.is_dir() and path.name.isdigit()
    ]
    return sorted(question_dirs, key=lambda path: int(path.name))


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


def parse_reme_stdout(stdout: str) -> str:
    answers: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("✅"):
            line = line.removeprefix("✅").strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            answer = payload.get("answer")
            if isinstance(answer, str) and answer.strip():
                answers.append(answer.strip())
                continue
            nested = payload.get("longmemeval_session_answer")
            if isinstance(nested, str) and nested.strip():
                answers.append(nested.strip())
                continue
        answers.append(line)
    return answers[-1] if answers else ""


def call_reme(
    query_path: Path,
    *,
    config: str | None,
    timeout: int,
) -> dict[str, Any]:
    cmd = ["reme", "longmemeval_session_answer"]
    if config:
        cmd.append(f"config={config}")
    cmd.extend(
        [
            f"timeout={timeout}",
            f"query_path={query_path}",
        ],
    )

    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout + 30,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout
        raise RuntimeError(f"reme exited {completed.returncode}: {detail}")

    return {
        "command": cmd,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "answer": parse_reme_stdout(completed.stdout) or "unknown",
    }


def cache_one(
    question_dir: Path,
    *,
    output_name: str,
    refresh: bool,
    config: str | None,
    timeout: int,
    include_raw: bool,
) -> dict[str, Any]:
    output_path = question_dir / output_name
    if output_path.exists() and not refresh:
        return {"index": question_dir.name, "status": "skipped", "path": str(output_path)}

    query_path = question_dir / "query.json"
    query_payload = load_json(query_path)
    answer_payload = load_json(question_dir / "answer.json") if (question_dir / "answer.json").exists() else {}
    reme_result = call_reme(query_path, config=config, timeout=timeout)

    payload = {
        "question_index": int(question_dir.name),
        "question_id": query_payload.get("question_id"),
        "question": query_payload.get("question"),
        "question_date": query_payload.get("question_date"),
        "query_path": str(query_path),
        "golden_answer": answer_payload.get("answer"),
        "answer_session_ids": answer_payload.get("answer_session_ids"),
        "longmemeval_session_answer": reme_result["answer"],
    }
    if include_raw:
        payload.update(
            {
                "reme_stdout": reme_result["stdout"],
                "reme_stderr": reme_result["stderr"],
                "reme_command": reme_result["command"],
            },
        )
    write_json_atomic(output_path, payload)
    return {
        "index": question_dir.name,
        "status": "cached",
        "path": str(output_path),
        "answer": reme_result["answer"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use reme longmemeval_session_answer to answer LongMemEval queries from sibling session dirs.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument(
        "--config",
        default="demo",
        help="ReMe config passed as config=VALUE. Use an empty string to omit it.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--end-index", type=int)
    parser.add_argument(
        "--indices",
        default="",
        help="Comma-separated indices or ranges, e.g. 0,3,10-20. Applied after start/end filters.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-run reme even if the output file already exists.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include raw reme stdout/stderr and the full command in output JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question_dirs = iter_question_dirs(args.dataset_dir)
    if args.start_index is not None:
        question_dirs = [
            path for path in question_dirs if int(path.name) >= args.start_index
        ]
    if args.end_index is not None:
        question_dirs = [path for path in question_dirs if int(path.name) <= args.end_index]
    if args.indices:
        selected = parse_indices(args.indices)
        question_dirs = [path for path in question_dirs if int(path.name) in selected]
    if args.limit:
        question_dirs = question_dirs[: args.limit]

    config = args.config or None
    counts = {"cached": 0, "skipped": 0, "failed": 0}
    failures: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                cache_one,
                question_dir,
                output_name=args.output_name,
                refresh=args.refresh,
                config=config,
                timeout=args.timeout,
                include_raw=args.include_raw,
            ): question_dir
            for question_dir in question_dirs
        }
        total = len(futures)
        for done, future in enumerate(as_completed(futures), start=1):
            question_dir = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                counts["failed"] += 1
                failures.append({"index": question_dir.name, "error": str(exc)})
                print(f"[{done}/{total}] failed {question_dir.name}: {exc}", flush=True)
                continue

            status = str(result["status"])
            counts[status] += 1
            answer = result.get("answer", "")
            print(
                f"[{done}/{total}] {status} {result['index']} "
                f"{json.dumps(answer, ensure_ascii=False)}",
                flush=True,
            )

    summary = {
        "dataset_dir": str(args.dataset_dir),
        "output_name": args.output_name,
        "queries": len(question_dirs),
        "counts": counts,
        "failures": failures,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
