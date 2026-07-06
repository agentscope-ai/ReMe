#!/usr/bin/env python3
"""Cache reme context_answer outputs for LongMemEval sessions."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_DATASET_DIR = Path("/Users/yuli/workspace/ReMe/longmemeval")
DEFAULT_OUTPUT_NAME = "context_answer.json"
DEFAULT_AGENT_KEY = "memory_time_range_soft_boost"


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


def find_session_file(question_dir: Path, session_id: str) -> Path:
    matches = [
        path
        for path in (question_dir / "session").glob("*.json")
        if load_json(path).get("haystack_session_id") == session_id
    ]
    if not matches:
        raise FileNotFoundError(
            f"Could not find session_id={session_id!r} under {question_dir / 'session'}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Found multiple files for session_id={session_id!r}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def format_session_context(session_payloads: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for payload in session_payloads:
        session_id = payload.get("haystack_session_id", "")
        session_date = payload.get("haystack_date", "")
        chunks.append(f"[session_id: {session_id} | date: {session_date}]")
        messages = payload.get("messages", [])
        if not isinstance(messages, list):
            continue
        for idx, message in enumerate(messages, start=1):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "unknown"))
            content = str(message.get("content", ""))
            has_answer = message.get("has_answer")
            suffix = " | has_answer: true" if has_answer is True else ""
            chunks.append(f"{idx}. {role}{suffix}: {content}")
    return "\n".join(chunks).strip()


def read_answer_session_ids(question_dir: Path) -> list[str]:
    answer_payload = load_json(question_dir / "answer.json")
    answer_session_ids = answer_payload.get("answer_session_ids")
    if not isinstance(answer_session_ids, list) or not answer_session_ids:
        raise ValueError(f"{question_dir / 'answer.json'} has no answer_session_ids")
    return [str(session_id) for session_id in answer_session_ids]


def read_agent_session_ids(question_dir: Path, agent_key: str) -> list[str]:
    agent_payload = load_json(question_dir / "agent.json")
    source: Any = agent_payload
    if "predicted_haystack_session_ids" not in agent_payload:
        source = agent_payload.get(agent_key)
    if not isinstance(source, dict):
        available = ", ".join(
            key
            for key, value in agent_payload.items()
            if isinstance(value, dict) and "predicted_haystack_session_ids" in value
        )
        raise ValueError(
            f"{question_dir / 'agent.json'} has no predictions for agent_key={agent_key!r}. "
            f"Available keys: {available or 'none'}"
        )

    session_ids = source.get("predicted_haystack_session_ids")
    if not isinstance(session_ids, list) or not session_ids:
        raise ValueError(
            f"{question_dir / 'agent.json'} {agent_key!r} has no "
            "predicted_haystack_session_ids"
        )
    return [str(session_id) for session_id in session_ids]


def read_session_ids(
    question_dir: Path,
    *,
    session_id_source: str,
    agent_key: str,
) -> tuple[list[str], dict[str, Any]]:
    if session_id_source == "answer":
        session_ids = read_answer_session_ids(question_dir)
        return session_ids, {"session_id_source": "answer"}
    if session_id_source == "agent":
        session_ids = read_agent_session_ids(question_dir, agent_key)
        return session_ids, {"session_id_source": "agent", "agent_key": agent_key}
    raise ValueError(f"Unsupported session_id_source={session_id_source!r}")


def normalize_context_answer(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, str] = {}
    for key in ("thinking", "answer"):
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            normalized[key] = item.strip()
    return normalized


def parse_reme_stdout(stdout: str) -> dict[str, Any]:
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

    structured_outputs: list[dict[str, str]] = []
    direct_outputs: list[dict[str, str]] = []
    for payload in candidates:
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get("context_answer"), dict):
            structured_outputs.append(normalize_context_answer(payload["context_answer"]))
            continue
        if isinstance(payload.get("structured_output"), dict):
            structured_outputs.append(normalize_context_answer(payload["structured_output"]))
            continue
        answer = payload.get("answer")
        if isinstance(answer, str):
            try:
                parsed_answer = json.loads(answer)
            except json.JSONDecodeError:
                parsed_answer = {"answer": answer}
            normalized = normalize_context_answer(parsed_answer)
            if normalized:
                direct_outputs.append(normalized)
                continue
        normalized = normalize_context_answer(payload)
        if normalized:
            direct_outputs.append(normalized)

    for normalized in [*structured_outputs, *direct_outputs]:
        if normalized.get("thinking") and normalized.get("answer"):
            return normalized
    for normalized in [*structured_outputs, *direct_outputs]:
        if normalized:
            return normalized
    return {}


def call_reme(
    query: str,
    session_context: str,
    *,
    config: str | None,
    timeout: int,
) -> dict[str, Any]:
    cmd = ["reme", "context_answer"]
    if config:
        cmd.append(f"config={config}")
    cmd.extend(
        [
            f"timeout={timeout}",
            f"query={json.dumps(query, ensure_ascii=False)}",
            f"session_context={json.dumps(session_context, ensure_ascii=False)}",
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
        "context_answer": parse_reme_stdout(completed.stdout),
    }


def cache_one(
    question_dir: Path,
    *,
    output_name: str,
    refresh: bool,
    config: str | None,
    timeout: int,
    include_raw: bool,
    session_id_source: str,
    agent_key: str,
) -> dict[str, Any]:
    output_path = question_dir / output_name
    if output_path.exists() and not refresh:
        return {"index": question_dir.name, "status": "skipped", "path": str(output_path)}

    query_payload = load_json(question_dir / "query.json")
    answer_payload = load_json(question_dir / "answer.json")
    query = str(query_payload["question"])
    session_ids, session_source_metadata = read_session_ids(
        question_dir,
        session_id_source=session_id_source,
        agent_key=agent_key,
    )

    session_payloads: list[dict[str, Any]] = []
    session_files: list[str] = []
    for session_id in session_ids:
        session_file = find_session_file(question_dir, str(session_id))
        session_files.append(str(session_file))
        session_payloads.append(load_json(session_file))

    session_context = format_session_context(session_payloads)
    reme_result = call_reme(
        query,
        session_context,
        config=config,
        timeout=timeout,
    )

    payload = {
        "question_index": int(question_dir.name),
        "question_id": query_payload.get("question_id"),
        "question": query,
        "golden_answer": answer_payload.get("answer"),
        "answer_session_ids": answer_payload.get("answer_session_ids"),
        "context_session_ids": session_ids,
        "session_files": session_files,
        "session_context": session_context,
        "context_answer": reme_result["context_answer"],
        **session_source_metadata,
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
        "status": "cached",
        "path": str(output_path),
        "context_answer": reme_result["context_answer"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use reme context_answer to answer LongMemEval queries from their "
            "gold answer sessions or agent-predicted sessions."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument(
        "--config",
        default="demo",
        help="ReMe config passed as config=VALUE. Use an empty string to omit it.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--session-id-source",
        choices=("answer", "agent"),
        default="answer",
        help=(
            "Read context sessions from answer.json answer_session_ids or "
            "agent.json predicted_haystack_session_ids."
        ),
    )
    parser.add_argument(
        "--agent-key",
        default=DEFAULT_AGENT_KEY,
        help=(
            "agent.json key to read when --session-id-source=agent. Ignored if "
            "predicted_haystack_session_ids exists at the top level."
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--end-index", type=int)
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
                session_id_source=args.session_id_source,
                agent_key=args.agent_key,
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
            answer = result.get("context_answer", {}).get("answer", "")
            print(
                f"[{done}/{total}] {status} {result['index']} "
                f"{json.dumps(answer, ensure_ascii=False)}",
                flush=True,
            )

    summary = {
        "dataset_dir": str(args.dataset_dir),
        "output_name": args.output_name,
        "session_id_source": args.session_id_source,
        "agent_key": args.agent_key if args.session_id_source == "agent" else None,
        "queries": len(question_dirs),
        "counts": counts,
        "failures": failures,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
