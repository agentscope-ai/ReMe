#!/usr/bin/env python3
"""Cache reme memory_time_range outputs into LongMemEval query.json files."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_DATASET_DIR = Path("/Users/yuli/workspace/ReMe/longmemeval")


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


def clean_date(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    value = value.strip().strip("\"'")
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None
    return value


def normalize_memory_time_range(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}

    normalized: dict[str, str] = {}
    thinking = value.get("thinking")
    if isinstance(thinking, str) and thinking.strip():
        normalized["thinking"] = thinking.strip()
    for key in ("start_dt", "end_dt"):
        cleaned = clean_date(value.get(key))
        if cleaned:
            normalized[key] = cleaned
    return normalized


def parse_reme_stdout(stdout: str) -> dict[str, str]:
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
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get("memory_time_range"), dict):
            return normalize_memory_time_range(payload["memory_time_range"])
        if isinstance(payload.get("structured_output"), dict):
            return normalize_memory_time_range(payload["structured_output"])
        if "start_dt" in payload or "end_dt" in payload:
            return normalize_memory_time_range(payload)
    return {}


def call_reme(question: str, question_date: str, timeout: int) -> dict[str, str]:
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
        timeout=timeout,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"reme exited {completed.returncode}: {stderr}")
    return parse_reme_stdout(completed.stdout)


def iter_query_paths(dataset_dir: Path) -> list[Path]:
    question_dirs = [
        path for path in dataset_dir.iterdir() if path.is_dir() and path.name.isdigit()
    ]
    question_dirs.sort(key=lambda path: int(path.name))
    return [path / "query.json" for path in question_dirs]


def cache_one(query_path: Path, *, refresh: bool, timeout: int) -> dict[str, Any]:
    query = load_json(query_path)
    existing = query.get("memory_time_range")
    if isinstance(existing, dict) and not refresh:
        normalized = normalize_memory_time_range(existing)
        if normalized != existing:
            query["memory_time_range"] = normalized
            write_json_atomic(query_path, query)
            return {"path": str(query_path), "status": "normalized", "range": normalized}
        return {"path": str(query_path), "status": "skipped", "range": existing}

    memory_time_range = call_reme(
        str(query["question"]), str(query["question_date"]), timeout
    )
    query["memory_time_range"] = memory_time_range
    write_json_atomic(query_path, query)
    return {"path": str(query_path), "status": "cached", "range": memory_time_range}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache reme memory_time_range into every LongMemEval query.json."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-run reme even if query.json already has memory_time_range.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N query files, useful for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query_paths = iter_query_paths(args.dataset_dir)
    if args.limit:
        query_paths = query_paths[: args.limit]

    counts = {"cached": 0, "skipped": 0, "normalized": 0, "failed": 0}
    failures: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                cache_one,
                query_path,
                refresh=args.refresh,
                timeout=args.timeout,
            ): query_path
            for query_path in query_paths
        }
        total = len(futures)
        for done, future in enumerate(as_completed(futures), start=1):
            query_path = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                counts["failed"] += 1
                failures.append({"path": str(query_path), "error": str(exc)})
                print(f"[{done}/{total}] failed {query_path}: {exc}", flush=True)
                continue

            status = str(result["status"])
            counts[status] += 1
            print(
                f"[{done}/{total}] {status} {query_path.parent.name} "
                f"{json.dumps(result['range'], ensure_ascii=False)}",
                flush=True,
            )

    summary = {
        "dataset_dir": str(args.dataset_dir),
        "queries": len(query_paths),
        "counts": counts,
        "failures": failures,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
