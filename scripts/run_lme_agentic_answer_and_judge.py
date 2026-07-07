#!/usr/bin/env python3
"""Run LongMemEval agentic answer-and-judge jobs over dataset indices."""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "datasets" / "longmemeval"
SEARCH_LIMIT_ENV = "REME_SEARCH_LIMIT"


@dataclass
class IndexResult:
    index: int
    ok: bool
    update_returncode: int
    answer_returncode: int | None
    message: str


def _run(cmd: list[str], timeout: int, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
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


def _answer_env(args: argparse.Namespace) -> dict[str, str] | None:
    if args.search_limit is None:
        return None
    env = os.environ.copy()
    env[SEARCH_LIMIT_ENV] = str(args.search_limit)
    return env


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
    answer = _run(answer_cmd, args.answer_timeout, env=_answer_env(args))
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
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="shuffle dataset indices before submitting jobs; use --no-shuffle to keep numeric order",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed used with --shuffle")
    parser.add_argument("--workers", type=int, default=4, help="number of indices to run concurrently")
    parser.add_argument("--update-timeout", type=int, default=600, help="seconds for each update_index job")
    parser.add_argument("--answer-timeout", type=int, default=3600, help="seconds for each answer-and-judge job")
    parser.add_argument(
        "--search-limit",
        type=int,
        help=(
            f"set {SEARCH_LIMIT_ENV} for answer jobs, overriding search.py's default limit; "
            "omit to use the code default"
        ),
    )
    args = parser.parse_args()
    if args.search_limit is not None and args.search_limit <= 0:
        parser.error("--search-limit must be positive")
    return args


def main() -> int:
    args = parse_args()
    indices = list(range(args.start_index, args.end_index + 1))
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(indices)
        seed_text = "none" if args.seed is None else str(args.seed)
        print(f"shuffled {len(indices)} indices with seed={seed_text}", flush=True)
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
