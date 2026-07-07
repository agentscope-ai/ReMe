#!/usr/bin/env python3
"""Run LongMemEval agentic answer-and-judge jobs over dataset indices."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "datasets" / "longmemeval"


@dataclass
class IndexResult:
    index: int
    ok: bool
    update_returncode: int
    answer_returncode: int | None
    message: str


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(cmd, 124, exc.stdout or "", exc.stderr or f"timeout after {timeout}s")
    except OSError as exc:
        return subprocess.CompletedProcess(cmd, 127, "", str(exc))


def run_index(index: int, args: argparse.Namespace) -> IndexResult:
    workspace_dir = DATASET_ROOT / str(index)
    if not workspace_dir.is_dir():
        return IndexResult(index, False, 0, None, f"missing workspace: {workspace_dir}")

    workspace_arg = f"workspace_dir={workspace_dir}"
    update_cmd = [
        args.reme_bin,
        "start",
        f"config={args.config}",
        "job=update_index",
        workspace_arg,
    ]
    update = _run(update_cmd, args.update_timeout)
    if update.returncode != 0:
        message = (update.stderr or update.stdout).strip()
        return IndexResult(index, False, update.returncode, None, message[-2000:])

    answer_cmd = [
        args.reme_bin,
        "start",
        f"config={args.config}",
        "job=agentic_answer_and_judge",
        workspace_arg,
        f"answer_id={args.answer_id}",
    ]
    answer = _run(answer_cmd, args.answer_timeout)
    if answer.returncode != 0:
        message = (answer.stderr or answer.stdout).strip()
        return IndexResult(index, False, update.returncode, answer.returncode, message[-2000:])

    result_path = workspace_dir / "result" / f"{args.answer_id}.json"
    return IndexResult(index, result_path.is_file(), update.returncode, answer.returncode, str(result_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-id", required=True, help="shared answer id, saved as result/{answer_id}.json")
    parser.add_argument("--config", default="jinli_lme", help="ReMe config name or path")
    parser.add_argument("--reme-bin", default="reme", help="reme executable")
    parser.add_argument("--start-index", type=int, default=0, help="first dataset index, inclusive")
    parser.add_argument("--end-index", type=int, default=499, help="last dataset index, inclusive")
    parser.add_argument("--workers", type=int, default=4, help="number of indices to run concurrently")
    parser.add_argument("--update-timeout", type=int, default=600, help="seconds for each update_index job")
    parser.add_argument("--answer-timeout", type=int, default=3600, help="seconds for each answer-and-judge job")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    indices = list(range(args.start_index, args.end_index + 1))
    failures: list[IndexResult] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_index, index, args): index for index in indices}
        for future in as_completed(futures):
            result = future.result()
            status = "ok" if result.ok else "fail"
            print(f"[{status}] {result.index}: {result.message}", flush=True)
            if not result.ok:
                failures.append(result)

    if failures:
        print(f"{len(failures)} / {len(indices)} indices failed", file=sys.stderr)
        return 1
    print(f"completed {len(indices)} indices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
