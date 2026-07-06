#!/usr/bin/env python3
"""Judge LongMemEval context_answer outputs with reme answer_judge."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_DATASET_DIR = Path("/Users/yuli/workspace/ReMe/longmemeval")
DEFAULT_CONTEXT_ANSWER_NAME = "context_answer.json"
DEFAULT_OUTPUT_NAME = "answer_judge.json"
DEFAULT_SUMMARY_NAME = "answer_judge_summary.json"


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


def normalize_answer_judgement(value: Any) -> dict[str, str | bool]:
    if not isinstance(value, dict):
        return {}

    normalized: dict[str, str | bool] = {}
    thinking = value.get("thinking")
    if isinstance(thinking, str) and thinking.strip():
        normalized["thinking"] = thinking.strip()

    answer = value.get("answer")
    if isinstance(answer, bool):
        normalized["answer"] = answer
    elif isinstance(answer, str):
        cleaned = answer.strip().lower()
        if cleaned in {"true", "yes", "correct", "对", "正确"}:
            normalized["answer"] = True
        elif cleaned in {"false", "no", "incorrect", "错", "错误", "不正确"}:
            normalized["answer"] = False
    return normalized


def parse_reme_stdout(stdout: str) -> dict[str, str | bool]:
    candidates: list[Any] = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("✅"):
            line = line.removeprefix("✅").strip()
        if not line.startswith("{"):
            continue
        try:
            candidates.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    structured_outputs: list[dict[str, str | bool]] = []
    direct_outputs: list[dict[str, str | bool]] = []
    for payload in candidates:
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get("answer_judgement"), dict):
            structured_outputs.append(
                normalize_answer_judgement(payload["answer_judgement"])
            )
            continue
        if isinstance(payload.get("structured_output"), dict):
            structured_outputs.append(
                normalize_answer_judgement(payload["structured_output"])
            )
            continue
        answer = payload.get("answer")
        if isinstance(answer, str):
            try:
                parsed_answer = json.loads(answer)
            except json.JSONDecodeError:
                parsed_answer = {"answer": answer}
            normalized = normalize_answer_judgement(parsed_answer)
            if normalized:
                direct_outputs.append(normalized)
                continue
        normalized = normalize_answer_judgement(payload)
        if normalized:
            direct_outputs.append(normalized)

    for normalized in [*structured_outputs, *direct_outputs]:
        if isinstance(normalized.get("answer"), bool):
            return normalized
    return {}


def call_reme(
    *,
    query: str,
    agent_answer: str,
    golden_answer: str,
    config: str | None,
    timeout: int,
) -> dict[str, Any]:
    cmd = ["reme", "answer_judge"]
    if config:
        cmd.append(f"config={config}")
    cmd.extend(
        [
            f"timeout={timeout}",
            f"query={json.dumps(query, ensure_ascii=False)}",
            f"agent_answer={json.dumps(agent_answer, ensure_ascii=False)}",
            f"golden_answer={json.dumps(golden_answer, ensure_ascii=False)}",
        ]
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
        "answer_judgement": parse_reme_stdout(completed.stdout),
    }


def read_agent_answer(context_answer_path: Path) -> str:
    payload = load_json(context_answer_path)
    context_answer = payload.get("context_answer")
    if isinstance(context_answer, dict):
        answer = context_answer.get("answer")
    else:
        answer = payload.get("longmemeval_session_answer", payload.get("answer"))
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(
            f"{context_answer_path} has no context_answer.answer, "
            "longmemeval_session_answer, or answer"
        )
    return answer.strip()


def judge_one(
    question_dir: Path,
    *,
    context_answer_name: str,
    output_name: str,
    refresh: bool,
    config: str | None,
    timeout: int,
    include_raw: bool,
) -> dict[str, Any]:
    output_path = question_dir / output_name
    if output_path.exists() and not refresh:
        payload = load_json(output_path)
        judgement = payload.get("answer_judgement")
        correct = judgement.get("answer") if isinstance(judgement, dict) else None
        return {
            "index": question_dir.name,
            "status": "skipped",
            "path": str(output_path),
            "correct": correct,
        }

    context_answer_path = question_dir / context_answer_name
    if not context_answer_path.exists():
        return {
            "index": question_dir.name,
            "status": "missing_context_answer",
            "path": str(context_answer_path),
        }

    query_payload = load_json(question_dir / "query.json")
    answer_payload = load_json(question_dir / "answer.json")
    query = str(query_payload["question"])
    agent_answer = read_agent_answer(context_answer_path)
    golden_answer = str(answer_payload["answer"])

    reme_result = call_reme(
        query=query,
        agent_answer=agent_answer,
        golden_answer=golden_answer,
        config=config,
        timeout=timeout,
    )
    judgement = reme_result["answer_judgement"]

    payload = {
        "question_index": int(question_dir.name),
        "question_id": query_payload.get("question_id"),
        "question": query,
        "agent_answer": agent_answer,
        "golden_answer": golden_answer,
        "answer_judgement": judgement,
    }
    if include_raw:
        payload.update(
            {
                "reme_stdout": reme_result["stdout"],
                "reme_stderr": reme_result["stderr"],
                "reme_command": reme_result["command"],
            }
        )
    write_json_atomic(output_path, payload)

    return {
        "index": question_dir.name,
        "status": "judged",
        "path": str(output_path),
        "correct": judgement.get("answer") if isinstance(judgement, dict) else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge LongMemEval context_answer.json files with reme answer_judge."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--context-answer-name", default=DEFAULT_CONTEXT_ANSWER_NAME)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument(
        "--config",
        default="demo",
        help="ReMe config passed as config=VALUE. Use an empty string to omit it.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--end-index", type=int)
    parser.add_argument(
        "--indices",
        default="",
        help="Comma-separated indices or ranges, e.g. 0,3,10-20.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-run reme even if the answer_judge output already exists.",
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
    counts = {
        "judged": 0,
        "skipped": 0,
        "missing_context_answer": 0,
        "failed": 0,
        "correct": 0,
        "incorrect": 0,
        "unknown": 0,
    }
    failures: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                judge_one,
                question_dir,
                context_answer_name=args.context_answer_name,
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
            correct = result.get("correct")
            if correct is True:
                counts["correct"] += 1
            elif correct is False:
                counts["incorrect"] += 1
            elif status not in {"missing_context_answer"}:
                counts["unknown"] += 1

            print(
                f"[{done}/{total}] {status} {result['index']} "
                f"{json.dumps(correct, ensure_ascii=False)}",
                flush=True,
            )

    judged_total = counts["correct"] + counts["incorrect"]
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "context_answer_name": args.context_answer_name,
        "output_name": args.output_name,
        "queries": len(question_dirs),
        "counts": counts,
        "accuracy": (counts["correct"] / judged_total) if judged_total else None,
        "failures": failures,
    }
    summary_path = args.dataset_dir / args.summary_name
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
