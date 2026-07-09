#!/usr/bin/env python3
"""Run LongMemEval ``session_review`` serially across samples.

For every workspace under ``datasets/longmemeval/<idx>`` in the selected numeric
range, this launches:

    reme start config=jinli_lme job=session_review

with ``LME_WORKSPACE_DIR`` pointed at that sample. Runs are strictly serial: the
next sample starts only after the previous process exits. Each sample's
stdout/stderr goes to ``logs/session_review/<idx>.log``.

By default the script processes samples 0..499 inclusive and reruns every sample
in that range. Pass ``--resume`` to skip samples whose ``session_review.json``
already exists.

Examples:
    python benchmark/longmemeval/run_session_review.py
    python benchmark/longmemeval/run_session_review.py --start 187 --end 499
    python benchmark/longmemeval/run_session_review.py --resume
    python benchmark/longmemeval/run_session_review.py --limit 5 --dry-run
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "datasets" / "longmemeval"
LOGDIR = REPO / "logs" / "session_review"
OUTPUT_FILENAME = "session_review.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=int, default=0, help="first numeric sample id to process, inclusive (default 0)")
    p.add_argument("--end", type=int, default=499, help="last numeric sample id to process, inclusive (default 499)")
    p.add_argument("--limit", type=int, default=0, help="only process the first N selected samples (0 = all)")
    p.add_argument(
        "--resume",
        action="store_true",
        help=f"skip samples whose {OUTPUT_FILENAME} already exists",
    )
    p.add_argument("--dry-run", action="store_true", help="list what would run, launch nothing")
    p.add_argument("--stop-on-fail", action="store_true", help="stop immediately after the first failed sample")
    return p.parse_args()


def sample_ids() -> list[str]:
    """List all sample IDs (numeric workspace dirs), numerically sorted."""
    ids = [p.name for p in DATA.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(ids, key=int)


def output_exists(idx: str) -> bool:
    """Return True when the sample already has a session review artifact."""
    return (DATA / idx / OUTPUT_FILENAME).exists()


def run_one(idx: str) -> bool:
    """Run ``session_review`` for one sample. Returns True on success."""
    log = LOGDIR / f"{idx}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, LME_WORKSPACE_DIR=f"datasets/longmemeval/{idx}")

    started = time.strftime("%H:%M:%S")
    print(f"[start {started}] {idx}", flush=True)
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            ["reme", "start", "config=jinli_lme", "job=session_review"],
            cwd=REPO,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )

    ok = proc.returncode == 0 and output_exists(idx)
    tag = "done" if ok else "fail"
    print(f"[{tag}] {idx} rc={proc.returncode} log={log}", flush=True)
    return ok


def main() -> int:
    """Run the serial driver."""
    args = parse_args()
    if args.end < args.start:
        raise ValueError(f"--end ({args.end}) must be >= --start ({args.start})")

    LOGDIR.mkdir(parents=True, exist_ok=True)

    ids = [i for i in sample_ids() if args.start <= int(i) <= args.end]
    if args.limit:
        ids = ids[: args.limit]

    pending = [i for i in ids if not (args.resume and output_exists(i))]
    print(
        f"job=session_review samples total={len(ids)} pending={len(pending)} "
        f"range={args.start}..{args.end} resume={args.resume}",
        flush=True,
    )

    if args.dry_run:
        for idx in pending:
            print(f"[would-run] {idx}")
        return 0

    counters = {"done": 0, "fail": 0, "skip": 0}
    for idx in ids:
        if args.resume and output_exists(idx):
            counters["skip"] += 1
            print(f"[skip] {idx} ({OUTPUT_FILENAME} exists)", flush=True)
            continue

        if run_one(idx):
            counters["done"] += 1
        else:
            counters["fail"] += 1
            if args.stop_on_fail:
                break

    print(
        f"ALL FINISHED done={counters['done']} fail={counters['fail']} skip={counters['skip']}",
        flush=True,
    )
    return 0 if counters["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
